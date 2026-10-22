#include <pthread.h>
#include <sys/mman.h>
#include <csetjmp>

#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/HostFeatures.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Debug/InternalThreadState.h>
#include <FEXCore/Utils/Allocator.h>

#include "Common/HostFeatures.h"

#include "base/functional/function.h"
#include "base/logging.h"
#include "base/memory/move.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "cpu/aarch64/code_map.h"
#include "cpu/aarch64/cpu_map.h"
#include "cpu/aarch64/fex_context.h"
#include "cpu/aarch64/host_thunks.h"
#include "cpu/aarch64/live_threads.h"
#include "cpu/aarch64/monitors.h"
#include "cpu/aarch64/stack_pool.h"
#include "cpu/aarch64/syscall_handler.h"
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "kern/crash.h"
#include "kern/module.h"
#include "logger/logger.h"
#include "memory_debug/memory_debug.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kVectorTso, "DELTA_FEX_VECTOR_TSO", false);
DELTA_OPTION(bool, kMemcpyTso, "DELTA_FEX_MEMCPY_TSO", false);
}  // namespace

namespace cpu {

using fex::CpuMap;
using fex::CpuUnmap;
using fex::g_context;
using fex::t_thread;

static thread_local std::jmp_buf t_exit_jmp;
static thread_local bool t_exit_jmp_valid = false;

namespace {

constexpr size_t kScratchTlsSize = 0x10000;

class FexSignalDelegator final : public FEXCore::SignalDelegator {};

u64 PS4ABI GuestFnReturnExit() {
  ExitGuestThread();
  return 0;
}

class FexBackend final : public Backend {
 public:
  void OnImageMapped(kern::ModuleInfo& info) override {
    EnsureInit();
    fex::AddExecRange(reinterpret_cast<u64>(info.base), info.code_size);
    fex::AddModuleRange(reinterpret_cast<u64>(info.base), info.code_size,
                        info.name.c_str());
    LOG_INFO("fex: registered exec range {} +{:#x}", (void*)info.base,
             info.code_size);
  }

  void* CreateGuestThread(uintptr_t entry, void* arg, u64 fsbase) override {
    const memory_debug::Scope memory_scope(memory_debug::Bucket::kThreadStacks);
    EnsureInit();
    auto* h = new FexThread{};

    h->stack_size = 8ull * 1024 * 1024;
    h->stack = stack_pool_.Take(h->stack_size);
    if (!h->stack)
      h->stack = CpuMap(nullptr, h->stack_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    u64 rsp =
        (reinterpret_cast<u64>(h->stack) + h->stack_size - 0x200) & ~0xFULL;

    auto* thread = ctx_->CreateThread();
    h->thread = thread;
    auto& s = thread->CurrentFrame->State;
    s.rip = entry;
    s.gregs[FEXCore::X86State::REG_RSP] = rsp;

    {
      constexpr size_t kSize =
          FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
      constexpr size_t kPage = 0x1000;
      h->callret_size = kSize + 2 * kPage;
      void* alloc = callret_pool_.Take(h->callret_size);
      if (!alloc)
        alloc = CpuMap(nullptr, h->callret_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (alloc != MAP_FAILED) {
        h->callret = alloc;
        void* cr_base = reinterpret_cast<u8*>(alloc) + kPage;
        if (mprotect(cr_base, kSize, PROT_READ | PROT_WRITE) == 0)
          memory_debug::TrackMapping(cr_base, kSize,
                                     memory_debug::Bucket::kThreadStacks);
        thread->CallRetStackBase = cr_base;
        s.callret_sp = reinterpret_cast<u64>(cr_base) + kSize / 4;
      }
    }

    // The decoder reads CS.L.
    s.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_GDT] =
        &h->gdt[0];
    s.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_LDT] =
        &h->gdt[0];
    s.cs_idx = FEXCore::Core::CPUState::DEFAULT_USER_CS << 3;
    auto* gdt = FEXCore::Core::CPUState::GetSegmentFromIndex(s, s.cs_idx);
    FEXCore::Core::CPUState::SetGDTBase(gdt, 0);
    FEXCore::Core::CPUState::SetGDTLimit(gdt, 0xFFFFFU);
    s.cs_cached = FEXCore::Core::CPUState::CalculateGDTBase(*gdt);
    gdt->L = 1;
    gdt->D = 0;

    s.gregs[FEXCore::X86State::REG_RDI] = reinterpret_cast<u64>(arg);

    // Early TLS reads need a TCB self-pointer at [fs:0].
    u64 fs = fsbase;
    if (fs == 0) {
      void* t = CpuMap(nullptr, kScratchTlsSize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (t != MAP_FAILED) {
        h->scratch_tls = t;
        fs = reinterpret_cast<u64>(t) + kScratchTlsSize / 2;
        *reinterpret_cast<u64*>(fs) = fs;
      }
    }
    s.fs_cached = fs;
    s.gs_cached = fs;
    return h;
  }

  void DiscardGuestThread(void* handle) override {
    auto* h = static_cast<FexThread*>(handle);
    ctx_->DestroyThread(h->thread);
    if (h->stack)
      CpuUnmap(h->stack, h->stack_size);
    if (h->callret)
      CpuUnmap(h->callret, h->callret_size);
    if (h->scratch_tls)
      CpuUnmap(h->scratch_tls, kScratchTlsSize);
    delete h;
  }

  void RunGuestThread(void* handle) override {
    const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
    auto* h = static_cast<FexThread*>(handle);
    t_thread = h->thread;
    guest::ThreadRegistration registration([](uintptr_t pc) {
      return g_context && t_thread &&
             g_context->IsAddressInCodeBuffer(t_thread, pc);
    });
    kern::InstallSigAltStack();
    // FEXCore init may have replaced the SIGSEGV/SIGILL handlers.
    kern::InstallCrashHandler();
    FEXCore::Allocator::RegisterTLSData(h->thread);
    fex::StartMonitors();
    u64 entry_rip = h->thread->CurrentFrame->State.rip;
    {
      pthread_attr_t at;
      void* hsp = nullptr;
      size_t hsz = 0;
      if (pthread_getattr_np(pthread_self(), &at) == 0) {
        pthread_attr_getstack(&at, &hsp, &hsz);
        pthread_attr_destroy(&at);
      }
      BASE_LOGI(
          "fex", "gthread rip={:#x} gstack=[{:p}+{:#x}] hoststack=[{:p}+{:#x}]",
          (unsigned long long)entry_rip, h->stack, h->stack_size, hsp, hsz);
    }
    const u32 my_id = fex::AddCurrentLiveThread(h->thread);
    LOG_INFO("fex: running guest thread rip={:#x} (watchdog tid={})",
             h->thread->CurrentFrame->State.rip, my_id);
    if (setjmp(t_exit_jmp) == 0) {
      t_exit_jmp_valid = true;
      guest::SetThreadExitHandler(&ExitGuestThread);
      guest::PausePoint();
      ctx_->ExecuteThread(h->thread);
    } else {
      LOG_INFO("fex: guest thread exited via thr_exit");
    }
    t_exit_jmp_valid = false;
    guest::SetThreadExitHandler(nullptr);
    auto& end_s = h->thread->CurrentFrame->State;
    LOG_INFO("fex: guest thread returned rip={:#x}", (unsigned long)end_s.rip);
    fex::ReportThreadReturn(my_id, entry_rip, end_s);
    fex::RemoveLiveThread(h->thread);
    FEXCore::Allocator::UninstallTLSData(h->thread);
    ctx_->DestroyThread(h->thread);
    t_thread = nullptr;
    if (h->scratch_tls)
      CpuUnmap(h->scratch_tls, kScratchTlsSize);
    if (h->stack)
      stack_pool_.Put(h->stack, h->stack_size);
    if (h->callret)
      callret_pool_.Put(h->callret, h->callret_size);
    delete h;
  }

  u64 RunGuestFunction(uintptr_t fn, u64 a0, u64 a1, u64 a2, u64 a3) override {
    const uintptr_t exit_thunk =
        MakeHostThunk(reinterpret_cast<void*>(&GuestFnReturnExit));

    // Module init reads TLS through libkernel; the caller waits meanwhile.
    u64 fsbase = t_thread ? t_thread->CurrentFrame->State.fs_cached : 0;

    auto* h = static_cast<FexThread*>(
        CreateGuestThread(fn, reinterpret_cast<void*>(a0), fsbase));
    auto& s = h->thread->CurrentFrame->State;
    s.gregs[FEXCore::X86State::REG_RSI] = a1;
    s.gregs[FEXCore::X86State::REG_RDX] = a2;
    s.gregs[FEXCore::X86State::REG_RCX] = a3;
    u64 rsp = s.gregs[FEXCore::X86State::REG_RSP] & ~0xFULL;
    rsp -= 8;
    *reinterpret_cast<u64*>(rsp) = exit_thunk;
    s.gregs[FEXCore::X86State::REG_RSP] = rsp;
    RunOnInitWorker([this, h] { RunGuestThread(h); });
    return 0;
  }

  void ResetSession() override {
    init_worker_started_ = init_worker_done_ = false;
    init_worker_job_ = nullptr;
    stack_pool_.Release();
    callret_pool_.Release();
    g_context = nullptr;
    ctx_.reset();
    init_done_ = false;
    fex::ResetHostThunks();
    fex::ResetCodeMap();
  }

 private:
  struct FexThread {
    FEXCore::Core::InternalThreadState* thread;
    void* stack;
    size_t stack_size;
    void* callret;
    size_t callret_size;
    void* scratch_tls;
    FEXCore::Core::CPUState::gdt_segment gdt[32];
  };

  void EnsureInit() {
    const memory_debug::Scope memory_scope(memory_debug::Bucket::kJit);
    base::LockGuard<base::Mutex> lk(init_m_);
    if (init_done_)
      return;
    init_done_ = true;
    FEXCore::Config::Initialize();
    FEXCore::Config::ReloadMetaLayer();
    FEXCore::Config::Set(FEXCore::Config::CONFIG_IS64BIT_MODE, "1");
    // The dual-CAS emulation of unaligned LOCK RMWs tears on ARM.
    FEXCore::Config::Set(FEXCore::Config::CONFIG_STRICTINPROCESSSPLITLOCKS,
                         "1");
    // SIMD copies and REP MOVS reorder on ARM; SotC publishes job descriptors
    // with vmovups pairs.
    if (kVectorTso)
      FEXCore::Config::Set(FEXCore::Config::CONFIG_VECTORTSOENABLED, "1");
    if (kMemcpyTso)
      FEXCore::Config::Set(FEXCore::Config::CONFIG_MEMCPYSETTSOENABLED, "1");

    auto host_features = FEX::FetchHostFeatures();
    ctx_ = FEXCore::Context::Context::CreateNewContext(host_features);
    g_context = ctx_.get();
    ctx_->SetSignalDelegator(&sig_delegator_);
    ctx_->SetSyscallHandler(&syscall_handler_);
    ctx_->EnableExitOnHLT();
    if (!ctx_->InitCore())
      LOG_ERROR("fex: FEXCore InitCore failed");
    else
      LOG_INFO("fex: FEXCore context initialised");
  }

  // Module inits keep pointers into their host thread's TCB (SotC FIOS2), so
  // they all run on one thread that never exits, as on the console.
  void RunOnInitWorker(base::Function<void()> job) {
    base::UniqueLock<base::Mutex> lk(init_worker_m_);
    if (!init_worker_started_) {
      init_worker_started_ = true;
      guest::SpawnThread("fex_backend", [this] {
        for (;;) {
          base::Function<void()> next;
          {
            base::UniqueLock<base::Mutex> wl(init_worker_m_);
            guest::Wait(init_worker_cv_, wl,
                        [this] { return (bool)init_worker_job_; });
            if (!init_worker_job_)
              return;
            next = base::move(init_worker_job_);
            init_worker_job_ = nullptr;
          }
          next();
          {
            base::LockGuard<base::Mutex> wl(init_worker_m_);
            init_worker_done_ = true;
          }
          init_worker_cv_.NotifyAll();
        }
      });
    }
    init_worker_done_ = false;
    init_worker_job_ = base::move(job);
    init_worker_cv_.NotifyAll();
    init_worker_cv_.Wait(lk, [this] { return init_worker_done_; });
  }

  base::Mutex init_m_;
  bool init_done_ = false;
  fextl::unique_ptr<FEXCore::Context::Context> ctx_;
  fex::FexSyscallHandler syscall_handler_;
  FexSignalDelegator sig_delegator_;

  fex::StackPool stack_pool_;
  fex::StackPool callret_pool_;

  base::Mutex init_worker_m_;
  base::ConditionVariable init_worker_cv_;
  bool init_worker_started_ = false;
  base::Function<void()> init_worker_job_;
  bool init_worker_done_ = false;
};

FexBackend g_backend;

}  // namespace

Backend& GetBackend() {
  return g_backend;
}

void ExitGuestThread() {
  if (t_exit_jmp_valid) {
    t_exit_jmp_valid = false;
    std::longjmp(t_exit_jmp, 1);
  }
}

u64 CurrentGuestRip() {
  return t_thread ? t_thread->CurrentFrame->State.rip : 0;
}

u64 CurrentGuestFsBase() {
  return t_thread ? t_thread->CurrentFrame->State.fs_cached : 0;
}

const u64* CurrentGuestGregs() {
  return t_thread ? reinterpret_cast<const u64*>(
                        t_thread->CurrentFrame->State.gregs)
                  : nullptr;
}

void SetThreadFsBase(u64 v) {
  if (t_thread)
    t_thread->CurrentFrame->State.fs_cached = v;
}

u64 ThreadFsBase() {
  return t_thread ? t_thread->CurrentFrame->State.fs_cached : 0;
}

}  // namespace cpu
