// Runs a module through load -> relocate -> start, one stage at a time, so each
// layer can be brought up on its own. Usage: modexec <main-module.sprx> [run]
#define _GNU_SOURCE
#include <cstdio>
#include "base/arch.h"
#include <cstring>

#include <csignal>
#include <cstdlib>
#include <ucontext.h>

#include <logger/logger.h>
#include <host_memory/host_memory.h>

#include "kern/lv2/sys_dynlib.h"
#include "kern/module.h"
#include "kern/process.h"
#include "kern/vfs.h"

#include <base/strings/string_ref.h>
#include <base/strings/xstring.h>

// Resolve a host address to "<module>+0x<off> (<seg>)" by scanning the loaded
// module images, so a guest fault points straight at a guest module offset.
static void symbolize(uintptr_t addr, char* out, size_t n) {
  auto* proc = krnl::Proc::GetActive();
  if (proc) {
    for (auto& mod : proc->GetModuleList()) {
      auto& mi = mod->GetInfo();
      auto* t = mi.text_seg.addr;
      auto* d = mi.data_seg.addr;
      if (t && addr >= (uintptr_t)t && addr < (uintptr_t)t + mi.text_seg.size) {
        std::snprintf(out, n, "%s+%#lx (.text)", mi.name.c_str(),
                      addr - (uintptr_t)t);
        return;
      }
      if (d && addr >= (uintptr_t)d && addr < (uintptr_t)d + mi.data_seg.size) {
        std::snprintf(out, n, "%s+%#lx (.data)", mi.name.c_str(),
                      addr - (uintptr_t)d);
        return;
      }
    }
  }
  std::snprintf(out, n, "%#lx (??)", addr);
}

static void crashHandler(int sig, siginfo_t* si, void* ucv) {
  auto* uc = static_cast<ucontext_t*>(ucv);
  auto* gr = uc->uc_mcontext.gregs;
  char rip[256], fault[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  symbolize((uintptr_t)si->si_addr, fault, sizeof(fault));
  std::fprintf(stderr, "\n=== GUEST FAULT: %s (signal %d) ===\n",
               strsignal(sig), sig);
  std::fprintf(stderr, "  rip   = %016llx  %s\n", (unsigned long long)gr[REG_RIP], rip);
  std::fprintf(stderr, "  fault = %016llx  %s\n", (unsigned long long)si->si_addr, fault);
  std::fprintf(stderr, "  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n",
               (unsigned long long)gr[REG_RAX], (unsigned long long)gr[REG_RBX],
               (unsigned long long)gr[REG_RCX], (unsigned long long)gr[REG_RDX]);
  std::fprintf(stderr, "  rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n",
               (unsigned long long)gr[REG_RSI], (unsigned long long)gr[REG_RDI],
               (unsigned long long)gr[REG_RBP], (unsigned long long)gr[REG_RSP]);
  std::fprintf(stderr, "  r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n",
               (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
               (unsigned long long)gr[REG_R10], (unsigned long long)gr[REG_R11]);
  std::fprintf(stderr, "  r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
               (unsigned long long)gr[REG_R12], (unsigned long long)gr[REG_R13],
               (unsigned long long)gr[REG_R14], (unsigned long long)gr[REG_R15]);
  // Dump the guest TLS state so we can see why __tls_get_addr returns null.
  // TCB = fs base; DTV = *(TCB+8); DTV[0]=generation, per-module block pointers
  // at DTV+0x10 + id*8 (see libkernel __tls_get_addr at 0x289c0).
  if (auto* proc = krnl::Proc::GetActive()) {
    auto* tcb = reinterpret_cast<u64*>(proc->GetEnv().fs_base);
    std::fprintf(stderr, "  --- TLS ---\n  tcb(fs)=%p\n", (void*)tcb);
    if (tcb) {
      auto* dtv = reinterpret_cast<u64*>(tcb[1]);
      std::fprintf(stderr, "  dtv=%p", (void*)dtv);
      if (dtv) {
        std::fprintf(stderr, " gen=%llu count=%llu\n", (unsigned long long)dtv[0],
                     (unsigned long long)dtv[1]);
        for (int i = 0; i < 8; i++)
          std::fprintf(stderr, "    dtv[%d] block=%016llx\n", i,
                       (unsigned long long)dtv[2 + i]);
      } else {
        std::fprintf(stderr, "\n");
      }
    }
  }

  std::fflush(stderr);
  std::_Exit(128 + sig);
}

static void InstallCrashHandler() {
  struct sigaction sa = {};
  sa.sa_sigaction = crashHandler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
}

// SCOUT: patch a guest function to `xor eax,eax; ret` (return 0). Used to step
// over libkernel-internal validation that rejects our externally-loaded module
// set, so we can see how much further the boot gets.
static void forceReturn0(krnl::Proc& proc, const char* mod, u32 off) {
  auto m = proc.GetModule(base::StringRef(mod));
  if (!m)
    return;
  u8* p = m->GetInfo().base + off;
  // mprotect needs a page-aligned base.
  auto page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~0xFFFull);
  host_memory::ProtectMem(page, 0x1000, host_memory::PageProtection::kRwx);
  p[0] = 0x31;  // xor eax, eax
  p[1] = 0xC0;
  p[2] = 0xC3;  // ret
  std::printf("[modexec] SCOUT patched %s+%#x -> return 0\n", mod, off);
}

// Redirect libkernel's __tls_get_addr (NID vNe1w4diLCs) to our host HLE. Found
// by NID so it's firmware-independent. libkernel's own dynamic-TLS allocator
// leaves DTV entries null, so general-dynamic __thread access faults; ours
// hands back a real per-module block. Overwrites the entry with
// `movabs rax, &fn; jmp rax`.
static void patchTlsGetAddr(krnl::Proc& proc) {
  auto k = proc.GetModule(base::StringRef("libkernel"));
  if (!k)
    return;
  uintptr_t addr = k->GetSymbolByNid("vNe1w4diLCs");
  if (!addr) {
    std::printf("[modexec] __tls_get_addr export not found\n");
    return;
  }
  auto* p = reinterpret_cast<u8*>(addr);
  auto page = reinterpret_cast<void*>(addr & ~0xFFFull);
  host_memory::ProtectMem(page, 0x2000, host_memory::PageProtection::kRwx);
  p[0] = 0x48;  // movabs rax, imm64
  p[1] = 0xB8;
  *reinterpret_cast<u64*>(p + 2) =
      reinterpret_cast<u64>(&krnl::GuestTlsGetAddr);
  p[10] = 0xFF;  // jmp rax
  p[11] = 0xE0;
  std::printf("[modexec] patched libkernel __tls_get_addr @%p -> host HLE\n", p);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <main-module.sprx>\n", argv[0]);
    return 1;
  }

  std::setvbuf(stdout, nullptr, _IONBF, 0);  // don't lose scout prints on crash
  InstallCrashHandler();

  logger::CreateLogger(true);

  // Mount /app0 onto the directory the main module lives in, so the game's
  // runtime file opens resolve to the extracted disc image.
  {
    base::String p(argv[1]);
    auto slash = p.find_last_of('/');
    base::String dir = slash == base::String::npos ? "." : p.substr(0, slash);
    krnl::vfs::Mount("/app0", dir.c_str());
    std::printf("[modexec] mounted /app0 -> %s\n", dir.c_str());
  }

  krnl::Proc proc;

  // stage 1: load. create() preloads libkernel + libSceLibcInternal, then the
  // main module and its DT_NEEDED tree.
  std::printf("[modexec] === stage 1: load ===\n");
  if (!proc.Create(base::String(argv[1]))) {
    std::printf("[modexec] proc::create FAILED\n");
    return 2;
  }
  auto& mods = proc.GetModuleList();
  std::printf("[modexec] loaded %zu modules\n", mods.size());
  for (auto& m : mods) {
    auto& mi = m->GetInfo();
    std::printf("[modexec]   %-24s base=%p init=+%#lx\n", mi.name.c_str(),
                (void*)mi.base,
                mi.init_addr ? (uintptr_t)(mi.init_addr - mi.base) : 0);
  }

  // stage 2: resolve imports + relocate every module (what guest libkernel
  // triggers via syscall 599 at startup).
  std::printf("[modexec] === stage 2: relocate ===\n");
  int rc = krnl::sys_dynlib_process_needed_and_relocate();
  std::printf("[modexec] relocate -> %d\n", rc);
  if (rc != 0)
    return 3;

  // A real eboot carries its own SCE process param (PT_SCE_PROCPARAM). Only when
  // it's missing (e.g. running a bare lib as main) do we fake a minimal one so
  // libkernel's _start clears its size + "ORBI" magic check.
  auto& m0 = mods[0]->GetInfo();
  if (!m0.proc_param) {
    static u8 procParam[0x50] = {};
    *reinterpret_cast<u64*>(procParam + 0x00) = sizeof(procParam);
    *reinterpret_cast<u32*>(procParam + 0x08) = 0x4942524F;  // "ORBI"
    *reinterpret_cast<u32*>(procParam + 0x0C) = 1;           // entry count (!= 0)
    *reinterpret_cast<u32*>(procParam + 0x10) = 0x11000000;  // sdk version
    m0.proc_param = procParam;
    m0.proc_param_size = sizeof(procParam);
    std::printf("[modexec] (using synthetic proc param)\n");
  } else {
    std::printf("[modexec] using module's own proc param (%u bytes)\n", m0.proc_param_size);
  }

  // stage 3 (opt-in): jump into the guest. proc::start enters libkernel's entry
  // with modules[0] as the main program.
  if (argc > 2 && std::strcmp(argv[2], "run") == 0) {
    // SCOUT patches for libkernel-internal module bookkeeping (11.00 offsets).
    forceReturn0(proc, "libkernel", 0x287e0);  // module-gen lib-id validator
    // AppContent's module_start eagerly creates an IPMI client to the SceAppContent
    // system service. We don't emulate the service process, so the client is NULL
    // and a virtual call faults. Neuter the singleton-init helper so init no-ops.
    forceReturn0(proc, "libSceAppContentUtil", 0x1a00);
    patchTlsGetAddr(proc);
    std::printf("[modexec] === stage 3: execute (jumping to guest entry) ===\n");
    std::fflush(stdout);
    proc.Start();
    std::printf("[modexec] returned from guest entry\n");
  }

  return 0;
}
