/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#define _GNU_SOURCE
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ucontext.h>
#include <unistd.h>

#include "crash.h"
#include "module.h"
#include "proc.h"
#include "cpu/cpu_backend.h"

namespace krnl {
const char *syscall_getname(uint32_t idx); // name_table.cpp

// Resolve a host address to "<module>+0x<off> (<seg>)" by scanning loaded module
// images, so a guest fault points straight at a guest module offset.
static void symbolize(uintptr_t addr, char *out, size_t n) {
  if (auto *proc = proc::getActive()) {
    for (auto &mod : proc->getModuleList()) {
      auto &mi = mod->getInfo();
      auto *t = mi.textSeg.addr;
      auto *d = mi.dataSeg.addr;
      if (t && addr >= (uintptr_t)t && addr < (uintptr_t)t + mi.textSeg.size) {
        std::snprintf(out, n, "%s+%#lx (.text)", mi.name.c_str(),
                      addr - (uintptr_t)t);
        return;
      }
      if (d && addr >= (uintptr_t)d && addr < (uintptr_t)d + mi.dataSeg.size) {
        std::snprintf(out, n, "%s+%#lx (.data)", mi.name.c_str(),
                      addr - (uintptr_t)d);
        return;
      }
    }
  }
  std::snprintf(out, n, "%#lx (??)", addr);
}

// Walk the rbp frame chain (guest code keeps frame pointers) and symbolize each
// return address. Bounded and range-checked so a bad frame can't loop or fault.
// Works for both backends: guest frames are x86-64 frames in identity-mapped
// memory, so the walk is plain pointer reads on either host arch.
static void backtrace(uintptr_t rbp) {
  std::fprintf(stderr, "  --- backtrace ---\n");
  for (int i = 0; i < 32; i++) {
    if (rbp < 0x10000 || (rbp & 7))
      break;
    auto *frame = reinterpret_cast<uintptr_t *>(rbp);
    uintptr_t next = frame[0];
    uintptr_t ret = frame[1];
    if (!ret)
      break;
    char sym[256];
    symbolize(ret, sym, sizeof(sym));
    std::fprintf(stderr, "  #%-2d %016lx  %s\n", i, ret, sym);
    if (next <= rbp)  // frames grow upward; stop if it doesn't
      break;
    rbp = next;
  }
}

// Per-syscall call counter, filled by the lv2 trampoline under DELTA_SCHIST.
extern "C" uint64_t g_sysHist[1024];

// SIGUSR1 probe: dump the receiving thread's current guest RIP + a stack scan of
// return addresses in loaded modules. Sent to every thread (one per /proc task)
// to find what a wedged title's threads are blocked on. x86-native only.
#if defined(__x86_64__)
static void probeHandler(int, siginfo_t *, void *ucv) {
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  std::fprintf(stderr, "[probe] tid=%ld rip=%016llx %s\n", (long)gettid(),
               (unsigned long long)gr[REG_RIP], rip);
  uintptr_t rsp = gr[REG_RSP];
  if (rsp >= 0x10000) {
    auto *sp = reinterpret_cast<uintptr_t *>(rsp);
    int printed = 0;
    for (int i = 0; i < 512 && printed < 6; i++) {
      char sym[256];
      symbolize(sp[i], sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        std::fprintf(stderr, "[probe]   sp+%-4x %s\n", i * 8, sym);
        printed++;
      }
    }
  }
  // DELTA_SCHIST syscall histogram (lv2.cpp counts each syscall in its trampoline).
  // Dump the non-zero counts so a slow/wedged title's hammered syscalls are visible
  // -- the only profiler available (perf/strace/proc-mem are yama-blocked here).
  // Signal ONE thread to avoid interleaved output from concurrent handlers.
  bool any = false;
  for (int i = 0; i < 1024; i++) {
    if (g_sysHist[i] > 100) {  // skip noise
      if (!any) { std::fprintf(stderr, "[schist] syscall counts:\n"); any = true; }
      std::fprintf(stderr, "[schist]   %4d %-28s %llu\n", i, syscall_getname(i),
                   (unsigned long long)g_sysHist[i]);
    }
  }
  std::fflush(stderr);
}
#endif

static void crashHandler(int sig, siginfo_t *si, void *ucv) {
  // Let the CPU backend handle JIT-internal signals (e.g. FEX unaligned-atomic
  // SIGBUS) and resume; only a genuinely fatal fault falls through to the dump.
  if (cpu::tryHandleJitSignal(sig, si, ucv))
    return;

  // Only the first faulting thread prints. A second concurrent fault (common at
  // teardown) would interleave the dump and can itself core-dump, truncating it.
  static std::atomic<bool> s_dumping{false};
  if (s_dumping.exchange(true)) {
    for (;;) pause();  // park until the first thread's _Exit ends the process
  }

#if defined(__x86_64__)
  // Guest SDK assert/__debugbreak is `int 0x41` (cd 41); in userspace it raises
  // SIGSEGV (#GP). Real hardware with no kernel debugger attached just steps over
  // it and the assert handler's caller continues, so do the same: skip the 2-byte
  // instruction and resume. Otherwise every guest assertion would kill the boot.
  if (sig == SIGSEGV && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *ip = reinterpret_cast<const uint8_t *>(uc->uc_mcontext.gregs[REG_RIP]);
    if (ip && ip[0] == 0xcd && ip[1] == 0x41) {
      static std::atomic<int> n{0};
      if (n.fetch_add(1) < 20) {
        char sym[256];
        symbolize(uc->uc_mcontext.gregs[REG_RIP], sym, sizeof(sym));
        auto *g = uc->uc_mcontext.gregs;
        std::fprintf(stderr,
                     "[assert] skipped guest int 0x41 @ %s "
                     "rsi=%lld rdx=%lld r15=%lld rax=%lld rcx=%lld\n",
                     sym, (long long)g[REG_RSI], (long long)g[REG_RDX],
                     (long long)g[REG_R15], (long long)g[REG_RAX],
                     (long long)g[REG_RCX]);
      }
      uc->uc_mcontext.gregs[REG_RIP] += 2;
      return;
    }
  }
#endif

  char fault[256];
  symbolize((uintptr_t)si->si_addr, fault, sizeof(fault));
  std::fprintf(stderr, "\n=== GUEST FAULT: %s (signal %d) ===\n",
               strsignal(sig), sig);
  if (int sc = cpu::faultingSyscall(); sc >= 0)
    std::fprintf(stderr, "  in syscall %d (%s)\n", sc, syscall_getname((uint32_t)sc));
  std::fprintf(stderr, "  fault = %016llx  %s\n",
               (unsigned long long)si->si_addr, fault);
#if defined(__x86_64__)
  // Native x86 host: the host signal context IS the guest context.
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  std::fprintf(stderr, "  rip   = %016llx  %s\n",
               (unsigned long long)gr[REG_RIP], rip);
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
  if (gr[REG_RIP]) {
    auto *b = reinterpret_cast<const uint8_t *>(gr[REG_RIP]);
    std::fprintf(stderr, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      std::fprintf(stderr, " %02x", b[i]);
    std::fprintf(stderr, "\n");
  }
  backtrace(gr[REG_RBP]);
#else
  // aarch64 host: guest x86 state lives in the FEXCore CPUState, not the host
  // ARM signal context. Reconstruct the precise guest RIP from the host JIT PC
  // (CPUState.rip alone is only block-accurate and multiblock hides the site).
  uint64_t hostpc = 0;
#if defined(__aarch64__)
  if (ucv)
    hostpc = static_cast<ucontext_t *>(ucv)->uc_mcontext.pc;
#endif
  uint64_t recon = cpu::reconstructGuestRip(hostpc);
  uint64_t grip = recon ? recon : cpu::currentGuestRip(); // fall back to block rip
  std::fprintf(stderr, "  host pc in JIT: %s\n", recon ? "yes" : "no (FEX/HLE C++)");
  char ripsym[256];
  symbolize(grip, ripsym, sizeof(ripsym));
  std::fprintf(stderr, "  host pc   = %016llx\n", (unsigned long long)hostpc);
  std::fprintf(stderr, "  guest rip = %016llx  %s\n",
               (unsigned long long)grip, ripsym);
  if (grip) {
    auto *b = reinterpret_cast<const uint8_t *>(grip);
    std::fprintf(stderr, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      std::fprintf(stderr, " %02x", b[i]);
    std::fprintf(stderr, "\n");
  }
  // Guest GPR dump + rbp backtrace (parity with the native x86 dump above).
  // gregs order is FEXCore::X86State::REG_* (RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,
  // R8..R15); mirror it locally so this TU needs no FEXCore headers.
  if (const uint64_t *g = cpu::currentGuestGregs()) {
    enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
    std::fprintf(stderr, "  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n",
                 (unsigned long long)g[RAX], (unsigned long long)g[RBX],
                 (unsigned long long)g[RCX], (unsigned long long)g[RDX]);
    std::fprintf(stderr, "  rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n",
                 (unsigned long long)g[RSI], (unsigned long long)g[RDI],
                 (unsigned long long)g[RBP], (unsigned long long)g[RSP]);
    std::fprintf(stderr, "  r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n",
                 (unsigned long long)g[R8], (unsigned long long)g[R9],
                 (unsigned long long)g[R10], (unsigned long long)g[R11]);
    std::fprintf(stderr, "  r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
                 (unsigned long long)g[R12], (unsigned long long)g[R13],
                 (unsigned long long)g[R14], (unsigned long long)g[R15]);
    // Print the boundary-call trace before the (best-effort, occasionally
    // out-of-bounds) stack scan so it survives even if the scan faults.
    cpu::dumpThreadTrace(stderr);
    std::fflush(stderr);
    backtrace(g[RBP]);
    // Raw stack scan: optimised guest code omits frame pointers, so the rbp
    // chain above misses frames. Scan the guest stack for any value that lands
    // in a loaded module's .text; that's the (super)set of return
    // addresses, i.e. the real call chain.
    std::fprintf(stderr, "  --- stack scan ---\n");
    auto *sp = reinterpret_cast<uintptr_t *>(g[RSP]);
    if (g[RSP] >= 0x10000) {
      for (int i = 0; i < 256; i++) {
        uintptr_t v = sp[i];
        char sym[256];
        symbolize(v, sym, sizeof(sym));
        if (std::strstr(sym, "(.text)"))
          std::fprintf(stderr, "  sp+%-4x %016lx  %s\n", i * 8, v, sym);
      }
    }
  }
#endif
  std::fflush(stderr);
  std::fflush(stdout);  // _Exit won't flush; keep the guest trace up to the fault
  std::_Exit(128 + sig);
}

void installCrashHandler() {
  struct sigaction sa = {};
  sa.sa_sigaction = crashHandler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
#if defined(__x86_64__)
  struct sigaction pa = {};
  pa.sa_sigaction = probeHandler;
  pa.sa_flags = SA_SIGINFO | SA_RESTART;  // don't abort the thread's blocking call
  sigemptyset(&pa.sa_mask);
  sigaction(SIGUSR1, &pa, nullptr);
#endif
}
}  // namespace krnl
