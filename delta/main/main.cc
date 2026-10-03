// Copyright (C) 2019 Force67

#include "host_memory/host_memory.h"
#include "io/path.h"
#include "logger/logger.h"
#include "options/options.h"
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak_util.h>
#endif

#include "base/containers/vector.h"
#include "base/strings/xstring.h"

#include <cstring>

#if defined(__linux__)
#include <sys/prctl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <VersionHelpers.h>
#include <Windows.h>
#endif

#include "base/threading/thread.h"
#include "cpu/backend.h"
#include "gpu/render/renderer.h"
#include "guest/session.h"
#include "host/window.h"
#include "kern/guest_va_space.h"
#include "main/command_line.h"
#include "main/firmware_config.h"
#include "main/home_screen.h"
#include "main/launcher.h"
#include "main/recent_games.h"
#include "ui/overlay_log.h"
#include "ui/pause_menu.h"

static bool VerifyViability() {
#ifdef _WIN32
  if (!IsWindows8OrGreater()) {
    LOG_ERROR(
        "Your operating system is outdated. Please update to windows 8 "
        "or newer.");
    return false;
  }
#endif

  constexpr size_t kOneMb = 1024ull * 1024ull;
  constexpr size_t kEightGb = 8ull * 1024ull * kOneMb;

  if (host_memory::GetAvailableMem() < kEightGb) {
    LOG_ERROR("Your system doesn't have enough physical memory to run " FXNAME);
    return false;
  }

#if defined(DELTA_BACKEND_NATIVE)
  // Native x86 host: the guest runs directly on this CPU, so it must itself
  // expose the instruction set PS4 code expects.
  base::String missingFeatures;
  Xbyak::util::Cpu cpu;

#define CHECK_FEATURE(x, y)               \
  if (!cpu.has(Xbyak::util::Cpu::t##x)) { \
    missingFeatures += y;                 \
    missingFeatures += ";";               \
  }

  CHECK_FEATURE(SSE, "SSE");
  CHECK_FEATURE(SSE2, "SSE2");
  CHECK_FEATURE(SSE3, "SSE3");
  CHECK_FEATURE(SSSE3, "SSSE3");
  CHECK_FEATURE(SSE41, "SSE4.1");
  CHECK_FEATURE(SSE42, "SSE4.2");
  CHECK_FEATURE(AESNI, "AES");
  CHECK_FEATURE(AVX, "AVX");
  CHECK_FEATURE(PCLMULQDQ, "CLMUL");
  CHECK_FEATURE(F16C, "F16C");
  CHECK_FEATURE(BMI1, "BM1");

  if (!missingFeatures.empty()) {
    LOG_ERROR("Your cpu is missing the following instructions: {}",
              missingFeatures.c_str());
    return false;
  }
#else
  // aarch64 host: guest x86-64 runs in the FEXCore JIT, which synthesises the
  // expected instruction set regardless of the host CPU.
  LOG_INFO("FEX backend: skipping host x86 feature probe");
#endif

  return true;
}

#ifdef _WIN32
// Associate .pkg with this executable under HKCU (no admin needed, idempotent)
// so double-clicking a package in Explorer launches us with its path.
static void registerPkgAssociation() {
  wchar_t exe[MAX_PATH]{};
  if (!GetModuleFileNameW(nullptr, exe, MAX_PATH))
    return;

  auto writeKey = [](const wchar_t* sub, const base::StringW& value) {
    HKEY key;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, sub, 0, nullptr, 0, KEY_WRITE,
                        nullptr, &key, nullptr) != ERROR_SUCCESS)
      return;
    RegSetValueExW(key, nullptr, 0, REG_SZ,
                   reinterpret_cast<const BYTE*>(value.c_str()),
                   static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
  };

  writeKey(L"Software\\Classes\\.pkg", L"PS4Delta.pkg");
  base::StringW cmd = L"\"";
  cmd += exe;
  cmd += L"\" \"%1\"";
  writeKey(L"Software\\Classes\\PS4Delta.pkg\\shell\\open\\command", cmd);
}

static void win32PostInit() {
  using NtQueryTimerResolution_t = LONG(WINAPI*)(PULONG, PULONG, PULONG);
  using NtSetTimerResolution_t = LONG(WINAPI*)(ULONG, BOOLEAN, PULONG);

  auto hNtLib = GetModuleHandleW(L"ntdll.dll");
  auto NtQueryTimerResolution_f = reinterpret_cast<NtQueryTimerResolution_t>(
      GetProcAddress(hNtLib, "NtQueryTimerResolution"));
  auto NtSetTimerResolution_f = reinterpret_cast<NtSetTimerResolution_t>(
      GetProcAddress(hNtLib, "NtSetTimerResolution"));

  ULONG min_res, max_res, orig_res, new_res;
  if (NtQueryTimerResolution_f(&min_res, &max_res, &orig_res) == 0)
    NtSetTimerResolution_f(max_res, TRUE, &new_res);

  registerPkgAssociation();
}
#endif

int main(int argc, char** argv) {
  auto command = cli::Parse(argc, argv);
  if (command.exit)
    return command.exit_code;
  if (!cli::ConfigureFirmware(command))
    return 2;

#if defined(__linux__)
  // Let a debugger attach to a run that is already going. Under the default
  // yama ptrace_scope=1 only an ancestor may attach, and a stuck title is
  // exactly the case where starting over under gdb changes the timing that
  // produced it.
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif
  logger::CreateLogger(true);
  logger::RouteBaseLogging();
  // Before anything logs: the on-screen panel shows the tail of the log, and
  // the boot lines are the ones worth seeing before a title even presents.
#if !defined(__ANDROID__)
  ui::OverlayLogAttach();
#endif
  // Before anything else: every subsystem below reads its knobs from here, and
  // most latch the value the first time they run.
  base::Vector<char*> option_argv{argv[0]};
  for (auto& option : command.options)
    option_argv.push_back(option.data());
  int option_argc = static_cast<int>(option_argv.size());
  option_argv.push_back(nullptr);
  options::Init(option_argc, option_argv.data());
  if (command.game.empty()) {
    if (command.dump_options || !command.configure_ps4_fw.empty() ||
        !command.configure_ps5_fw.empty())
      return 0;
#if defined(__linux__) && !defined(__ANDROID__)
    command.game = cli::ShowHomeScreen(cli::ReadRecentGames());
#endif
    if (command.game.empty())
      return 0;
  }
  // Bring the render Vulkan device up NOW, before any guest memory is mapped:
  // initialized lazily (first Gnm submit), the NVIDIA driver fails its
  // in-process setup once the guest's huge MAP_FIXED mappings exist
  // (vk_icdGetInstanceProcAddr returns NULL) and enumeration silently falls
  // back to the llvmpipe software rasteriser, ~30 ms/frame instead of a real
  // GPU. Harmless when only llvmpipe exists (same device either way).
  gpu::render::Init(gpu::render::DefaultRenderer());
  // Claim the addresses the guest MAP_FIXEDs before anything host-side can be
  // handed them, in particular before the CPU backend reserves its JIT heap.
  kern::ReserveGuestVaSpace();
  cpu::EarlyInit();  // segregate guest/JIT memory before guest modules map

  if (!VerifyViability())
    return -1;

  Launcher core;

  if (!core.Init())
    return -1;

#ifdef _WIN32
  win32PostInit();
#endif

  if (!command.guest_args.empty()) {
    core.argv.reserve(command.guest_args.size() + 1);
    core.argv.emplace_back();
    for (const auto& arg : command.guest_args)
      core.argv.emplace_back(arg.c_str());
  }
  for (;;) {
    core.Boot(base::String(command.game.c_str()));
    while (!ui::PauseMenuExitRequested() && !ui::PauseMenuReturnRequested() &&
           !guest::Stopping())
      base::SleepForMilliseconds(16);
    const bool exit = ui::PauseMenuExitRequested();
    core.Stop();
    if (exit || ui::PauseMenuExitRequested())
      break;
#if defined(__linux__) && !defined(__ANDROID__)
    command.game = cli::ShowHomeScreen(cli::ReadRecentGames()).c_str();
#else
    command.game.clear();
#endif
    if (command.game.empty())
      break;
  }
  host::Shutdown();

  return 0;
}
