/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#define _GNU_SOURCE
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ucontext.h>

#include "crash.h"
#include "module.h"
#include "proc.h"

namespace krnl {
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

static void crashHandler(int sig, siginfo_t *si, void *ucv) {
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256], fault[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  symbolize((uintptr_t)si->si_addr, fault, sizeof(fault));
  std::fprintf(stderr, "\n=== GUEST FAULT: %s (signal %d) ===\n",
               strsignal(sig), sig);
  std::fprintf(stderr, "  rip   = %016llx  %s\n",
               (unsigned long long)gr[REG_RIP], rip);
  std::fprintf(stderr, "  fault = %016llx  %s\n",
               (unsigned long long)si->si_addr, fault);
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
  backtrace(gr[REG_RBP]);
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
}
}  // namespace krnl
