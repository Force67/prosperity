// Copyright (C) 2019 Force67

#include <base.h>
#include <logger/logger.h>
#include <utl/mem.h>
#include <utl/path.h>
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak_util.h>
#endif

#include <base/strings/xstring.h>
#include <base/containers/vector.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#include <VersionHelpers.h>
#endif

#include "dcore.h"
#include "cpu/cpu_backend.h"

static bool verifyViablity() {
#ifdef _WIN32
  if (!IsWindows8OrGreater()) {
    LOG_ERROR("Your operating system is outdated. Please update to windows 8 "
              "or newer.");
    return false;
  }
#endif

  constexpr size_t one_mb = 1024ull * 1024ull;
  constexpr size_t eight_gb = 8ull * 1024ull * one_mb;

  if (utl::getAvailableMem() < eight_gb) {
    LOG_ERROR("Your system doesn't have enough physical memory to run " FXNAME);
    return false;
  }

#if defined(DELTA_BACKEND_NATIVE)
  // Native x86 host: the guest runs directly on this CPU, so it must itself
  // expose the instruction set PS4 code expects.
  base::String missingFeatures;
  Xbyak::util::Cpu cpu;

#define CHECK_FEATURE(x, y)                                                    \
  if (!cpu.has(Xbyak::util::Cpu::t##x)) {                                      \
    missingFeatures += y;                                                      \
    missingFeatures += ";";                                                    \
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

  auto writeKey = [](const wchar_t *sub, const std::wstring &value) {
    HKEY key;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, sub, 0, nullptr, 0, KEY_WRITE,
                        nullptr, &key, nullptr) != ERROR_SUCCESS)
      return;
    RegSetValueExW(key, nullptr, 0, REG_SZ,
                   reinterpret_cast<const BYTE *>(value.c_str()),
                   static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
  };

  writeKey(L"Software\\Classes\\.pkg", L"PS4Delta.pkg");
  std::wstring cmd = L"\"";
  cmd += exe;
  cmd += L"\" \"%1\"";
  writeKey(L"Software\\Classes\\PS4Delta.pkg\\shell\\open\\command", cmd);
}

static void win32PostInit() {
  using NtQueryTimerResolution_t = LONG(WINAPI *)(PULONG, PULONG, PULONG);
  using NtSetTimerResolution_t = LONG(WINAPI *)(ULONG, BOOLEAN, PULONG);

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

EXPORT int dcoreMain(int argc, char **argv) {
  cpu::earlyInit(); // segregate guest/JIT memory before anything maps (FEX path)
  utl::createLogger(true);

  if (!verifyViablity())
    return -1;

  deltaCore core;

  if (!core.init())
    return -1;

#ifdef _WIN32
  win32PostInit();
#endif

  if (argc > 1) {
    if (argc > 2) {
      core.argv.reserve(argc - 1);
      core.argv.emplace_back();
      for (int i = 2; i < argc; ++i) {
        core.argv.emplace_back(argv[i]);
      }
    }

    base::String path(argv[1]);
    core.boot(path);
  }

  // Block forever; proc runs on a detached thread.
  // TODO: replace with proper shutdown signal once we have an event loop.
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return 0;
}
