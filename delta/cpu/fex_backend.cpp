/*
 * PS4Delta : PS4 emulation and research project
 *
 * FexBackend (aarch64 host). Guest PS4 x86-64 code runs inside an embedded
 * FEXCore JIT. Nothing in the guest image is rewritten: the JIT decodes
 * `syscall` itself and hands it to HandleSyscall (dispatched to lv2), and
 * emulates fs/gs from the guest CPUState. Each guest thread is a FEXCore thread
 * pinned 1:1 to a host thread, so the "current FEXCore thread" is a host
 * thread_local and guest TLS (fs base) is that thread's CPUState.fs_cached.
 *
 * Mirrors the standalone harness (see memory fex-arm-embedding
 * and tools fex-embed/harness.cpp).
 */
#if defined(DELTA_BACKEND_FEX)

#include <base.h>
#include <logger/logger.h>

#include <atomic>
#include <thread>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <ucontext.h>
#include <vector>
#include <sys/mman.h>

#include <FEXCore/Utils/ArchHelpers/Arm64.h>
#include <FEXCore/Utils/Allocator.h>

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/HLE/SyscallHandler.h>
#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/HostFeatures.h>
#include <FEXCore/Debug/InternalThreadState.h>

#include "Common/HostFeatures.h" // FEX::FetchHostFeatures

#include "cpu_backend.h"
#include "kern/module.h"
#include "kern/proc.h"

namespace krnl {
uintptr_t lv2_get(uint32_t sysIndex);
const char *syscall_getname(uint32_t idx);
struct tls_index;
void *PS4ABI guest_tls_get_addr(tls_index *ti); // HLE dynamic-TLS resolver
}

namespace cpu {

// The FEXCore thread executing on this host thread (1:1). Set on entry so the
// syscall handler and setThreadFsBase can reach the live guest CPUState.
// Internal linkage at namespace scope so krnl::setThreadFsBase (same TU) reaches it.
static thread_local FEXCore::Core::InternalThreadState *t_curThread = nullptr;

// Raw context pointer for the signal-path helpers (reconstructGuestRip); the
// owning unique_ptr lives in FexBackend.
static FEXCore::Context::Context *g_ctxPtr = nullptr;

// Last syscall this host thread entered (and whether it returned), so the crash
// handler can name the syscall a fault occurred inside.
static thread_local uint32_t t_lastSyscall = 0xFFFFFFFFu;
static thread_local bool t_inSyscall = false;

// Per-thread ring of recent guest->host boundary crossings (syscalls + HLE
// thunk calls), dumped by the crash handler for the faulting thread. Kept tiny
// and lock-free (thread_local) so it is safe to touch from a signal handler.
struct TraceEvt {
  char kind;        // 's' syscall, 'h' HLE thunk, 0 = empty
  uint32_t id;      // syscall number / thunk index
  uint64_t a0, a1, a2, a3;
  uint64_t ret;     // result (or sentinel before return)
  uint64_t caller;  // guest caller PC (HLE only)
  const char *name; // static HLE name pointer (HLE only), else nullptr
};
static constexpr uint32_t kTraceRing = 96;
static thread_local TraceEvt t_trace[kTraceRing];
static thread_local uint32_t t_tracePos = 0;
static inline TraceEvt &traceNext() {
  TraceEvt &e = t_trace[t_tracePos % kTraceRing];
  t_tracePos++;
  return e;
}

// Set around CTX->ExecuteThread so thr_exit can bail out of the JIT (longjmp)
// instead of returning into guest code (which libkernel treats as fatal).
static thread_local std::jmp_buf t_exitJmp;
static thread_local bool t_exitJmpValid = false;

namespace {

// Executable guest ranges registered by the loader, queried by the JIT.
struct ExecRange {
  uint64_t base, size;
};
std::mutex g_rangeMutex;
std::vector<ExecRange> g_ranges;

// Per-thunk HLE name (libname!NID), parallel to g_hostThunks, for DELTA_HLE_TRACE.
std::vector<std::string> g_thunkNames;

// Named module ranges, for the deadlock watchdog's symbolization.
struct NamedRange { uint64_t base, size; std::string name; };
std::mutex g_namedMutex;
std::vector<NamedRange> g_named;
static void symRange(uint64_t a, char *out, size_t n) {
  std::lock_guard lk(g_namedMutex);
  for (auto &r : g_named)
    if (a >= r.base && a < r.base + r.size) {
      std::snprintf(out, n, "%s[%#llx]+%#llx",
                    r.name.empty() ? "?" : r.name.c_str(),
                    (unsigned long long)r.base,
                    (unsigned long long)(a - r.base));
      return;
    }
  std::snprintf(out, n, "%#llx", (unsigned long long)a);
}

// Live guest threads, for the DELTA_WATCHDOG=secs deadlock dump: after N seconds
// it prints every live thread's current guest RIP so a stalled boot's blocking
// site can be symbolized to a module+offset without a debugger.
struct LiveThread { FEXCore::Core::InternalThreadState *thread; uint32_t id; };
std::mutex g_liveMutex;
std::vector<LiveThread> g_live;
std::atomic<uint32_t> g_liveSeq{0};
static void startWatchdog() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char *e = std::getenv("DELTA_WATCHDOG");
    if (!e) return;
    int secs = std::atoi(e);
    if (secs <= 0) secs = 20;
    std::thread([secs] {
      for (int round = 0;; round++) {
        std::this_thread::sleep_for(std::chrono::seconds(secs));
        std::lock_guard lk(g_liveMutex);
        std::fprintf(stderr, "=== WATCHDOG round %d: %zu live guest threads ===\n",
                     round, g_live.size());
        for (auto &t : g_live) {
          auto &S = t.thread->CurrentFrame->State;
          char sym[256];
          symRange(S.rip, sym, sizeof(sym));
          std::fprintf(stderr, "  tid=%u rip=%#llx (%s)\n", t.id,
                       (unsigned long long)S.rip, sym);
          // Scan the stack upward for return addresses into known modules (the
          // wait stub omits frame pointers, so a raw scan beats an rbp walk) to
          // reveal which subsystem this thread is parked inside.
          uint64_t rsp = S.gregs[FEXCore::X86State::REG_RSP];
          int shown = 0;
          for (int i = 0; i < 1024 && shown < 12; i++) {
            uint64_t a = rsp + (uint64_t)i * 8;
            if (a < 0x1000) break;
            uint64_t v = 0;
            std::memcpy(&v, reinterpret_cast<void *>(a), 8);
            // a plausible code return address that lands in a named module range
            if (v < 0x200000000000ull || v >= 0x210000000000ull) continue;
            char s2[256];
            symRange(v, s2, sizeof(s2));
            if (s2[0] == '0') continue;  // unnamed range -> skip noise
            std::fprintf(stderr, "      stk+%#x %#llx (%s)\n", i * 8,
                         (unsigned long long)v, s2);
            shown++;
          }
        }
        std::fflush(stderr);
      }
    }).detach();
  });
}

// Host-thunk table: index -> native HLE function, dispatched from a guest
// trampoline via the kHostThunkSyscallBase magic syscall. See makeHostThunk.
std::mutex g_thunkMutex;
std::vector<void *> g_hostThunks;
// Bump-allocated pool of guest-executable trampolines (one per bound HLE export).
uint8_t *g_thunkPool = nullptr;
size_t g_thunkPoolUsed = 0;
constexpr size_t g_thunkPoolSize = 0x100000; // 1 MiB -> ~95k trampolines
constexpr size_t kThunkStride = 16;

class FexSyscallHandler final : public FEXCore::HLE::SyscallHandler {
public:
  FexSyscallHandler() { OSABI = FEXCore::HLE::SyscallOSABI::OS_LINUX64; }

  uint64_t HandleSyscall(FEXCore::Core::CpuStateFrame *Frame,
                         FEXCore::HLE::SyscallArguments *Args) override {
    // Args->Argument[0] = syscall number (RAX); [1..6] = RDI,RSI,RDX,R10,R8,R9.
    const uint32_t num = static_cast<uint32_t>(Args->Argument[0]);

    // Dynamic-TLS bridge: the patched guest __tls_get_addr issues this magic
    // syscall with the tls_index pointer in rdi (Argument[1]).
    if (num == kTlsGetAddrSyscall) {
      uint64_t r = reinterpret_cast<uint64_t>(krnl::guest_tls_get_addr(
          reinterpret_cast<krnl::tls_index *>(Args->Argument[1])));
      if (g_ctxPtr) {
        uint32_t ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
        g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef & ~1u);
      }
      return r;
    }

    // Host-thunk bridge: a guest trampoline (planted by makeHostThunk) issued
    // this magic syscall to invoke a native HLE function. Reconstruct the SysV
    // call arguments: the trampoline did `mov r10,rcx` so the original 4th arg
    // (rcx, which `syscall` clobbers) is in Argument[4]; args 7-8 sit on the
    // guest stack just above the return address.
    if ((num & 0xFF000000u) == kHostThunkSyscallBase) {
      const uint32_t idx = num & 0x00FFFFFFu;
      void *fn = nullptr;
      {
        std::lock_guard lk(g_thunkMutex);
        if (idx < g_hostThunks.size())
          fn = g_hostThunks[idx];
      }
      uint64_t ret = 0;
      if (fn) {
        // Reconstruct SysV args 7..14 from the guest stack just above the return
        // address (the trampoline pushed nothing). Passing extra args a callee
        // ignores is harmless; this covers up to 14-arg Sce exports such as
        // sceGnmSubmitAndFlipCommandBuffers (9 args).
        const uint64_t rsp = Frame->State.gregs[FEXCore::X86State::REG_RSP];
        uint64_t s[8] = {};
        if (rsp)
          for (int i = 0; i < 8; i++)
            s[i] = reinterpret_cast<uint64_t *>(rsp)[i + 1];
        using Fn = uint64_t(PS4ABI *)(uint64_t, uint64_t, uint64_t, uint64_t,
                                      uint64_t, uint64_t, uint64_t, uint64_t,
                                      uint64_t, uint64_t, uint64_t, uint64_t,
                                      uint64_t, uint64_t);
        static const bool hleTrace = std::getenv("DELTA_HLE_TRACE") != nullptr;
        uint64_t caller = (rsp) ? reinterpret_cast<uint64_t *>(rsp)[0] : 0;
        const char *evName = nullptr;
        { std::lock_guard lk(g_thunkMutex);
          if (idx < g_thunkNames.size()) evName = g_thunkNames[idx].c_str(); }
        TraceEvt &ev = traceNext();
        ev = {'h', idx, Args->Argument[1], Args->Argument[2], Args->Argument[3],
              Args->Argument[4], ~0ull, caller, evName};
        ret = reinterpret_cast<Fn>(fn)(
            Args->Argument[1], Args->Argument[2], Args->Argument[3],
            Args->Argument[4], Args->Argument[5], Args->Argument[6], s[0], s[1],
            s[2], s[3], s[4], s[5], s[6], s[7]);
        ev.ret = ret;
        if (hleTrace) {
          char cs[256]; symRange(caller, cs, sizeof(cs));
          const char *nm = "";
          { std::lock_guard lk(g_thunkMutex);
            if (idx < g_thunkNames.size()) nm = g_thunkNames[idx].c_str(); }
          std::fprintf(stderr, "[hle] %s thunk#%u(%#lx,%#lx,%#lx,%#lx) -> %#lx  from %s\n",
                       nm, idx, Args->Argument[1], Args->Argument[2],
                       Args->Argument[3], Args->Argument[4],
                       (unsigned long)ret, cs);
        }
      }
      if (g_ctxPtr) {
        uint32_t ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
        g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef & ~1u);
      }
      return ret;
    }

    const uintptr_t handler = krnl::lv2_get(num);
    if (!handler)
      return 0;

    // Optional syscall trace: FEX_SCTRACE=1.
    static const bool trace = std::getenv("FEX_SCTRACE") != nullptr;
    if (trace)
      std::fprintf(stderr, "[sc] %3u %-22s (%#lx, %#lx, %#lx, %#lx, %#lx, %#lx)\n",
                   num, krnl::syscall_getname(num), Args->Argument[1],
                   Args->Argument[2], Args->Argument[3], Args->Argument[4],
                   Args->Argument[5], Args->Argument[6]);

    // The lv2 handlers are plain AArch64 functions (PS4ABI is empty off-x86);
    // call with the six GPR args and return the result into RAX.
    // TODO(boot): PS4/BSD return convention reports error via carry flag + errno, and
    // a few syscalls return a second value in RDX. We currently surface only
    // RAX, matching the native path's simplified handlers.
    using Fn = uint64_t(PS4ABI *)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    auto fn = reinterpret_cast<Fn>(handler);
    t_lastSyscall = num;
    t_inSyscall = true;
    TraceEvt &ev = traceNext();
    ev = {'s', num, Args->Argument[1], Args->Argument[2], Args->Argument[3],
          Args->Argument[4], ~0ull, 0, nullptr};
    uint64_t ret = fn(Args->Argument[1], Args->Argument[2], Args->Argument[3],
                      Args->Argument[4], Args->Argument[5], Args->Argument[6]);
    ev.ret = ret;
    t_inSyscall = false;
    if (trace)
      std::fprintf(stderr, "    -> %#lx\n", ret);

    // FreeBSD/PS4 syscall convention: carry clear = success. Our HLE handlers
    // return the result in rax and don't signal errors via carry; the native
    // lifted path leaves CF clear (handler epilogue / the calltrace clc()).
    // FEX's OS_LINUX64 path reports errors as negative rax and leaves the
    // pre-syscall flags untouched, so the guest's `jb cerror` reads a stale
    // carry and treats a good call as failure (e.g. ScePthread aborting on a
    // perfectly good sysctl(kern.usrstack)). CF isn't stored directly in
    // flags[]; clear it through FEX's compacted-EFLAGS API so it sticks.
    if (g_ctxPtr) {
      uint32_t ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
      ef &= ~1u; // EFLAGS.CF (bit 0)
      g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef);
    }
    return ret;
  }

  FEXCore::HLE::ExecutableRangeInfo
  QueryGuestExecutableRange(FEXCore::Core::InternalThreadState *, uint64_t Address) override {
    std::lock_guard lk(g_rangeMutex);
    for (auto &r : g_ranges)
      if (Address >= r.base && Address < r.base + r.size)
        return {r.base, r.size, false};
    return {0, 0, false};
  }

  std::optional<FEXCore::ExecutableFileSectionInfo>
  LookupExecutableFileSection(FEXCore::Core::InternalThreadState *, uint64_t) override {
    return std::nullopt;
  }
};

// Minimal signal delegator. Sufficient for fault-free guest code; a real game
// that self-modifies or faults will need the host SIGSEGV/SIGILL plumbing.
// TODO(boot): real signal handling (SMC write-protect faults, guest signals).
class FexSignalDelegator final : public FEXCore::SignalDelegator {};

// Return target for runGuestFunction: a synchronously-called guest function rets
// here, and we longjmp out of the JIT just like thr_exit. Dispatched as a host
// thunk, so its signature matches the thunk call path (extra args ignored).
static uint64_t PS4ABI guestFnReturnExit() {
  exitGuestThread();
  return 0;  // unreachable (exitGuestThread longjmps)
}

class FexBackend final : public ICpuBackend {
public:
  void onImageMapped(krnl::moduleInfo &info) override {
    ensureInit();
    std::lock_guard lk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<uint64_t>(info.base), info.codeSize});
    {
      std::lock_guard nk(g_namedMutex);
      g_named.push_back({reinterpret_cast<uint64_t>(info.base), info.codeSize,
                         std::string(info.name.c_str())});
    }
    LOG_INFO("fex: registered exec range {} +{:#x}", (void *)info.base, info.codeSize);
  }

  // Per-guest-thread bookkeeping. The gdt lives here (FEX tracks GDT/LDT per
  // thread; sharing one array across threads is incorrect) alongside the guest
  // stack and call-ret stack so they can be freed when the thread finishes.
  struct FexThread {
    FEXCore::Core::InternalThreadState *thread;
    void *stack;
    size_t stackSize;
    void *callret;
    size_t callretSize;
    FEXCore::Core::CPUState::gdt_segment gdt[32];
  };

  void *createGuestThread(uintptr_t entry, void *arg, uint64_t fsbase) override {
    ensureInit();
    auto *h = new FexThread{};

    // Guest stack (the guest's own RSP); HLE handlers run on the host thread
    // stack, so this only needs to satisfy guest code.
    h->stackSize = 8ull * 1024 * 1024;
    h->stack = mmap(nullptr, h->stackSize, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint64_t rsp = (reinterpret_cast<uint64_t>(h->stack) + h->stackSize - 0x200) & ~0xFULL;

    // IMPORTANT: create on the calling (parent) thread, as FEX's ThreadManager
    // does, never on the freshly spawned worker while other guest threads run.
    auto *thread = CTX->CreateThread(entry, rsp, nullptr);
    h->thread = thread;
    auto &S = thread->CurrentFrame->State;

    // FEX call/return prediction stack (guard-paged), seeded to Base + SIZE/4.
    {
      constexpr size_t kSize = FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
      constexpr size_t kPage = 0x1000;
      h->callretSize = kSize + 2 * kPage;
      void *alloc = mmap(nullptr, h->callretSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (alloc != MAP_FAILED) {
        h->callret = alloc;
        void *crBase = reinterpret_cast<uint8_t *>(alloc) + kPage;
        mprotect(crBase, kSize, PROT_READ | PROT_WRITE);
        thread->CallRetStackBase = crBase;
        S.callret_sp = reinterpret_cast<uint64_t>(crBase) + kSize / 4;
      }
    }

    // Per-thread 64-bit segments (the decoder reads CS.L).
    S.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_GDT] = &h->gdt[0];
    S.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_LDT] = &h->gdt[0];
    S.cs_idx = FEXCore::Core::CPUState::DEFAULT_USER_CS << 3;
    auto *gdt = FEXCore::Core::CPUState::GetSegmentFromIndex(S, S.cs_idx);
    FEXCore::Core::CPUState::SetGDTBase(gdt, 0);
    FEXCore::Core::CPUState::SetGDTLimit(gdt, 0xFFFFFU);
    S.cs_cached = FEXCore::Core::CPUState::CalculateGDTBase(*gdt);
    gdt->L = 1;
    gdt->D = 0;

    // PS4 entry convention: argument block pointer in RDI.
    S.gregs[FEXCore::X86State::REG_RDI] = reinterpret_cast<uint64_t>(arg);

    // Seed guest TLS (fs base). Until the guest installs its own via sysarch, an
    // unset (0) base would fault early TLS reads at fs+disp; give a scratch TLS
    // region with a TCB self-pointer at [fs:0], mirroring native's valid host fs.
    uint64_t fs = fsbase;
    if (fs == 0) {
      constexpr size_t kTls = 0x10000;
      void *t = mmap(nullptr, kTls, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (t != MAP_FAILED) {
        fs = reinterpret_cast<uint64_t>(t) + kTls / 2;
        *reinterpret_cast<uint64_t *>(fs) = fs;
      }
    }
    S.fs_cached = fs;
    S.gs_cached = fs;
    return h;
  }

  void runGuestThread(void *handle) override {
    auto *h = static_cast<FexThread *>(handle);
    t_curThread = h->thread;
    FEXCore::Allocator::RegisterTLSData(h->thread); // FEX per-thread registration
    startWatchdog();
    uint32_t myId = g_liveSeq.fetch_add(1);
    uint64_t entryRip = h->thread->CurrentFrame->State.rip;
    { std::lock_guard lk(g_liveMutex); g_live.push_back({h->thread, myId}); }
    LOG_INFO("fex: running guest thread rip={:#x} (watchdog tid={})",
             h->thread->CurrentFrame->State.rip, myId);
    // thr_exit (cpu::exitGuestThread) longjmps here to leave the JIT without
    // returning to guest code. The thread is being torn down regardless, so
    // abandoning the JIT dispatcher's host frame is safe.
    if (setjmp(t_exitJmp) == 0) {
      t_exitJmpValid = true;
      CTX->ExecuteThread(h->thread);
    } else {
      LOG_INFO("fex: guest thread exited via thr_exit");
    }
    t_exitJmpValid = false;
    auto &endS = h->thread->CurrentFrame->State;
    LOG_INFO("fex: guest thread returned rip={:#x}", (unsigned long)endS.rip);
    if (std::getenv("DELTA_WATCHDOG")) {
      char es[256]; symRange(entryRip, es, sizeof(es));
      char rs[256]; symRange(endS.rip, rs, sizeof(rs));
      std::fprintf(stderr, "=== THREAD tid=%u RETURNED entry=%#llx (%s) ret=%#llx (%s) ===\n",
                   myId, (unsigned long long)entryRip, es,
                   (unsigned long long)endS.rip, rs);
      if (endS.rip >= 0x200000000000ull && endS.rip < 0x210000000000ull) {
        const uint8_t *b = reinterpret_cast<const uint8_t *>(endS.rip);
        std::fprintf(stderr, "      bytes@rip: %02x %02x %02x %02x %02x %02x\n",
                     b[0], b[1], b[2], b[3], b[4], b[5]);
      }
      uint64_t rsp = endS.gregs[FEXCore::X86State::REG_RSP];
      int shown = 0;
      for (int i = 0; i < 2048 && shown < 20; i++) {
        uint64_t a = rsp + (uint64_t)i * 8, v = 0;
        if (a < 0x1000) break;
        std::memcpy(&v, reinterpret_cast<void *>(a), 8);
        if (v < 0x200000000000ull || v >= 0x210000000000ull) continue;
        char s2[256]; symRange(v, s2, sizeof(s2));
        if (s2[0] == '0') continue;
        std::fprintf(stderr, "      stk+%#x %#llx (%s)\n", i * 8,
                     (unsigned long long)v, s2);
        shown++;
      }
    }
    // Diagnostic: a guest thread that "returns" to a tiny rip jumped through a
    // bad/unset function pointer (e.g. a GPU thread with no real GPU backend).
    // Dump its registers + a module-resolved stack scan to pin the culprit.
    if (endS.rip < 0x100000ull) {
      std::fprintf(stderr, "=== BOGUS THREAD RETURN rip=%#lx ===\n",
                   (unsigned long)endS.rip);
      const char *rn[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
      for (int i = 0; i < 16; i++)
        std::fprintf(stderr, "  %s=%#lx\n", rn[i], (unsigned long)endS.gregs[i]);
      uint64_t rsp = endS.gregs[FEXCore::X86State::REG_RSP];
      if (rsp) {
        for (int i = 0; i < 64; i++) {
          uint64_t v = reinterpret_cast<uint64_t *>(rsp)[i];
          if (v >= 0x200000000000ull && v < 0x206000000000ull)
            std::fprintf(stderr, "  stk+%#x = %#lx\n", i * 8, (unsigned long)v);
        }
      }
    }
    { std::lock_guard lk(g_liveMutex);
      for (size_t i = 0; i < g_live.size(); i++)
        if (g_live[i].thread == h->thread) { g_live.erase(g_live.begin() + i); break; } }
    FEXCore::Allocator::UninstallTLSData(h->thread);
    CTX->DestroyThread(h->thread);
    t_curThread = nullptr;
    if (h->stack) munmap(h->stack, h->stackSize);
    if (h->callret) munmap(h->callret, h->callretSize);
    delete h;
  }

  uint64_t runGuestFunction(uintptr_t fn, uint64_t a0, uint64_t a1,
                            uint64_t a2) override {
    // A guest function that returns must land somewhere; point its return address
    // at a host thunk that calls exitGuestThread, so the JIT unwinds cleanly.
    static uintptr_t exitThunk =
        makeHostThunk(reinterpret_cast<void *>(&guestFnReturnExit));

    // Inherit the caller's guest TLS (fs base): module init calls into libkernel,
    // which reads thread-local state. The caller (blocked on join below) isn't
    // touching its TLS meanwhile, so sharing it for this synchronous call is safe
    // and avoids faulting on the scratch-TLS a fresh thread would otherwise get.
    uint64_t fsbase = t_curThread ? t_curThread->CurrentFrame->State.fs_cached : 0;

    // createGuestThread sets RDI=arg; add RSI/RDX for the 2nd/3rd SysV args.
    auto *h = static_cast<FexThread *>(
        createGuestThread(fn, reinterpret_cast<void *>(a0), fsbase));
    auto &S = h->thread->CurrentFrame->State;
    S.gregs[FEXCore::X86State::REG_RSI] = a1;
    S.gregs[FEXCore::X86State::REG_RDX] = a2;
    // Push the return address. After the implicit `call`, x86 wants rsp%16==8 at
    // the callee's first instruction, so 16-align then subtract 8.
    uint64_t rsp = S.gregs[FEXCore::X86State::REG_RSP] & ~0xFULL;
    rsp -= 8;
    *reinterpret_cast<uint64_t *>(rsp) = exitThunk;
    S.gregs[FEXCore::X86State::REG_RSP] = rsp;
    // Run on a fresh host thread (never nest ExecuteThread on the caller's host
    // thread) and block until it finishes. fn returns -> exitThunk -> longjmp.
    std::thread([this, h] { runGuestThread(h); }).join();
    return 0;
  }

private:
  void ensureInit() {
    std::call_once(initFlag, [this] {
      FEXCore::Config::Initialize();
      FEXCore::Config::ReloadMetaLayer();
      FEXCore::Config::Set(FEXCore::Config::CONFIG_IS64BIT_MODE, "1");

      auto HostFeatures = FEX::FetchHostFeatures();
      CTX = FEXCore::Context::Context::CreateNewContext(HostFeatures);
      g_ctxPtr = CTX.get();
      CTX->SetSignalDelegator(&sigDelegator);
      CTX->SetSyscallHandler(&syscallHandler);
      CTX->EnableExitOnHLT();
      if (!CTX->InitCore())
        LOG_ERROR("fex: FEXCore InitCore failed");
      else
        LOG_INFO("fex: FEXCore context initialised");
    });
  }

  std::once_flag initFlag;
  fextl::unique_ptr<FEXCore::Context::Context> CTX;
  FexSyscallHandler syscallHandler;
  FexSignalDelegator sigDelegator;
};

FexBackend g_backend;

} // namespace

namespace {
// Dedicated VA region for FEXCore's internal allocations (JIT code buffers,
// block-link maps, lookup caches). Kept disjoint from guest memory: FEX
// identity-maps the guest into the host VA, and the PS4 guest reserves large
// MAP_FIXED ranges high in the address space (~0xfcxx_xxxx_xxxx); if FEX's
// kernel-chosen ::mmap internals land there too, a guest fixed mapping clobbers
// them (zero-fills the JIT's block-link map -> the null-node crash). We route
// all of FEXCore's allocations into this reserved window instead.
#ifdef __ANDROID__
// 39-bit user VA: pin above the guest arena (sys_mem kCeil = 384 GiB) and below
// where bionic's mmap_base/stack live (~448 GiB+). Reserved first in earlyInit.
constexpr uintptr_t kFexHeapBase = 0x0000'0060'0000'0000ull; // 384 GiB
constexpr size_t kFexHeapSize = 32ull * 1024 * 1024 * 1024;  // 32 GiB
#else
constexpr uintptr_t kFexHeapBase = 0x0000'5000'0000'0000ull; // 80 TiB
constexpr size_t kFexHeapSize = 96ull * 1024 * 1024 * 1024;  // 96 GiB
#endif
std::atomic<uintptr_t> g_fexHeapNext{0};
uintptr_t g_fexHeapEnd = 0;

void *fexInternalMmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  // Honor explicit addresses (incl. MAP_FIXED); FEX knows what it's doing.
  if (addr || !g_fexHeapEnd)
    return ::mmap(addr, len, prot, flags, fd, off);
  // Bump-allocate anonymous requests from the reserved window with MAP_FIXED so
  // they can never overlap guest memory.
  const size_t alen = (len + 0xFFFull) & ~0xFFFull;
  uintptr_t base = g_fexHeapNext.fetch_add(alen, std::memory_order_relaxed);
  if (base + alen > g_fexHeapEnd)
    return ::mmap(nullptr, len, prot, flags, fd, off); // window exhausted: fall back
  return ::mmap(reinterpret_cast<void *>(base), len, prot, flags | MAP_FIXED, fd, off);
}
int fexInternalMunmap(void *addr, size_t len) { return ::munmap(addr, len); }
} // namespace

void earlyInit() {
  // Reserve the window PROT_NONE so the kernel won't hand any of it to guest
  // mmaps, then point FEXCore's allocator hooks at it. Done before any context
  // or guest mapping exists.
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#ifdef MAP_FIXED_NOREPLACE
  void *r = ::mmap(reinterpret_cast<void *>(kFexHeapBase), kFexHeapSize, PROT_NONE,
                   flags | MAP_FIXED_NOREPLACE, -1, 0);
  if (r == MAP_FAILED || r != reinterpret_cast<void *>(kFexHeapBase))
    r = ::mmap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#else
  void *r = ::mmap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#endif
  if (r == MAP_FAILED) {
    LOG_WARNING("fex: could not reserve internal heap window; JIT memory shares "
                "guest VA (may corrupt under heavy guest mmap use)");
    return;
  }
  g_fexHeapNext.store(reinterpret_cast<uintptr_t>(r), std::memory_order_relaxed);
  g_fexHeapEnd = reinterpret_cast<uintptr_t>(r) + kFexHeapSize;
  FEXCore::Allocator::mmap = fexInternalMmap;
  FEXCore::Allocator::munmap = fexInternalMunmap;
  LOG_INFO("fex: reserved internal heap {} +{:#x}", r, kFexHeapSize);
}

ICpuBackend &backend() { return g_backend; }

void exitGuestThread() {
  if (t_exitJmpValid) {
    t_exitJmpValid = false;
    std::longjmp(t_exitJmp, 1);
  }
  // Not in a guest thread context: nothing to unwind.
}

// Plant a guest x86 trampoline that bounces into the native HLE function `hostFn`
// via the kHostThunkSyscallBase magic syscall. The trampoline preserves the 4th
// arg (rcx) into r10 before `syscall` clobbers rcx, matching the dispatch above.
uintptr_t makeHostThunk(void *hostFn, const char *name) {
  std::lock_guard lk(g_thunkMutex);
  g_thunkNames.resize(g_hostThunks.size() + 1);
  g_thunkNames[g_hostThunks.size()] = name ? name : "";
  if (!g_thunkPool) {
    g_thunkPool = static_cast<uint8_t *>(
        mmap(nullptr, g_thunkPoolSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunkPool == MAP_FAILED) {
      g_thunkPool = nullptr;
      LOG_ERROR("fex: host-thunk pool mmap failed");
      return 0;
    }
    // FEX won't JIT code outside a registered executable range.
    std::lock_guard rk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<uint64_t>(g_thunkPool), g_thunkPoolSize});
  }
  if (g_thunkPoolUsed + kThunkStride > g_thunkPoolSize) {
    LOG_ERROR("fex: host-thunk pool exhausted");
    return 0;
  }
  const uint32_t idx = static_cast<uint32_t>(g_hostThunks.size());
  g_hostThunks.push_back(hostFn);

  uint8_t *t = g_thunkPool + g_thunkPoolUsed;
  g_thunkPoolUsed += kThunkStride;
  const uint32_t sc = kHostThunkSyscallBase | idx;
  uint8_t *p = t;
  *p++ = 0x49; *p++ = 0x89; *p++ = 0xCA;           // mov r10, rcx
  *p++ = 0xB8;                                       // mov eax, imm32
  std::memcpy(p, &sc, 4); p += 4;
  *p++ = 0x0F; *p++ = 0x05;                          // syscall
  *p++ = 0xC3;                                       // ret
  return reinterpret_cast<uintptr_t>(t);
}

uint64_t currentGuestRip() {
  return t_curThread ? t_curThread->CurrentFrame->State.rip : 0;
}

const uint64_t *currentGuestGregs() {
  return t_curThread ? t_curThread->CurrentFrame->State.gregs : nullptr;
}

int faultingSyscall() { return t_inSyscall ? static_cast<int>(t_lastSyscall) : -1; }

void dumpThreadTrace(void *fileStar) {
  auto *f = static_cast<std::FILE *>(fileStar);
  if (!f)
    return;
  std::fprintf(f, "  --- last guest->host calls (this thread, oldest first) ---\n");
  uint32_t count = t_tracePos < kTraceRing ? t_tracePos : kTraceRing;
  uint32_t start = t_tracePos - count;
  for (uint32_t i = 0; i < count; i++) {
    const TraceEvt &e = t_trace[(start + i) % kTraceRing];
    if (e.kind == 's') {
      std::fprintf(f, "  sc  %3u %-22s (%#llx,%#llx,%#llx,%#llx) -> %#llx\n",
                   e.id, krnl::syscall_getname(e.id),
                   (unsigned long long)e.a0, (unsigned long long)e.a1,
                   (unsigned long long)e.a2, (unsigned long long)e.a3,
                   (unsigned long long)e.ret);
    } else if (e.kind == 'h') {
      char cs[256];
      symRange(e.caller, cs, sizeof(cs));
      std::fprintf(f, "  hle %s(%#llx,%#llx,%#llx,%#llx) -> %#llx  from %s\n",
                   e.name ? e.name : "?", (unsigned long long)e.a0,
                   (unsigned long long)e.a1, (unsigned long long)e.a2,
                   (unsigned long long)e.a3, (unsigned long long)e.ret, cs);
    }
  }
}

uint64_t reconstructGuestRip(uint64_t hostPC) {
  if (!g_ctxPtr || !t_curThread)
    return 0;
  if (!g_ctxPtr->IsAddressInCodeBuffer(t_curThread, hostPC))
    return 0;
  return g_ctxPtr->RestoreRIPFromHostPC(t_curThread, hostPC);
}

bool tryHandleJitSignal(int sig, void *infop, void *ucv) {
#if defined(__aarch64__)
  // FEX raises SIGBUS(BUS_ADRALN) from the JIT for unaligned atomic accesses
  // and backpatches them. Mirror FEX's frontend SIGBUS handler. (FEX's own
  // SignalDelegator owns the richer SIGSEGV/SMC path; we only need this one
  // case.)
  if (sig != SIGBUS || !g_ctxPtr || !t_curThread || !ucv || !infop)
    return false;
  auto *uc = static_cast<ucontext_t *>(ucv);
  const uint64_t pc = uc->uc_mcontext.pc;
  if (!g_ctxPtr->IsAddressInCodeBuffer(t_curThread, pc))
    return false;
  if (static_cast<siginfo_t *>(infop)->si_code != BUS_ADRALN)
    return false;
  auto result = FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(
      t_curThread, FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier,
      pc, reinterpret_cast<uint64_t *>(&uc->uc_mcontext.regs[0]));
  uc->uc_mcontext.pc = pc + result.value_or(0);
  return result.has_value();
#else
  (void)sig; (void)infop; (void)ucv;
  return false;
#endif
}

} // namespace cpu

// Guest fs base (TLS) is the current FEXCore thread's CPUState.fs_cached. Called
// by the guest via sys_sysarch(AMD64_SET_FSBASE) and on thread spawn.
namespace krnl {
void setThreadFsBase(uint64_t v) {
  if (cpu::t_curThread)
    cpu::t_curThread->CurrentFrame->State.fs_cached = v;
}
} // namespace krnl

#endif // DELTA_BACKEND_FEX
