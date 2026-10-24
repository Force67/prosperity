/*
 * PS4Delta : PS4 emulation and research project
 *
 * User contexts. See sys_context.h.
 */

#include "kern/lv2/sys_context.h"

#include <cstring>
#include "base/arch.h"
#include "cpu/backend.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_thread.h"
#include "kern/process.h"

namespace kern {
namespace {
constexpr u32 kSysSigreturn = 417;
constexpr u32 kSysGetcontext = 421;
constexpr u32 kSysSetcontext = 422;
constexpr u32 kSysSwapcontext = 423;

// ucontext_t: the signal mask, the mcontext at 0x40, then uc_link and the
// rest, which getcontext does not copy out.
constexpr size_t kUcMcontext = 0x40;
constexpr size_t kUcCopySize = 0x4c0;
constexpr size_t kUcSize = 0x500;

// mcontext_t.
enum : size_t {
  kMcRdi = 0x08,
  kMcRsi = 0x10,
  kMcRdx = 0x18,
  kMcRcx = 0x20,
  kMcR8 = 0x28,
  kMcR9 = 0x30,
  kMcRax = 0x38,
  kMcRbx = 0x40,
  kMcRbp = 0x48,
  kMcR10 = 0x50,
  kMcR11 = 0x58,
  kMcR12 = 0x60,
  kMcR13 = 0x68,
  kMcR14 = 0x70,
  kMcR15 = 0x78,
  kMcFlags = 0x90,
  kMcEs = 0x94,
  kMcDs = 0x96,
  kMcRip = 0xa0,
  kMcCs = 0xa8,
  kMcRflags = 0xb0,
  kMcRsp = 0xb8,
  kMcSs = 0xc0,
  kMcLen = 0xc8,
  kMcFpformat = 0xd0,
  kMcOwnedfp = 0xd8,
  kMcFpstate = 0x100,
  kMcFsbase = 0x440,
  kMcXfpustate = 0x450,
  kMcXfpustateLen = 0x458,
};
constexpr u64 kMcSize = 0x480;
constexpr u32 kMcHasSegs = 1;
constexpr u32 kMcHasBases = 2;
constexpr u32 kMcHasFpxstate = 4;
constexpr u64 kFpFmtNodev = 0x10000;
constexpr u64 kFpFmtXmm = 0x10002;
constexpr u64 kFpOwnedNone = 0x20000;
constexpr u64 kFpOwnedFpu = 0x20001;
constexpr u64 kFpOwnedPcb = 0x20002;
constexpr u16 kUserCodeSel = 0x43;
constexpr u16 kUserDataSel = 0x3b;

// The rflags bits a context may set (the kernel's PSL_USERCHANGE), less TF:
// a single-step trap would take the emulator down, not hand the guest SIGTRAP.
constexpr u64 kRflagsUser = 0x254dd5 & ~u64{0x100};
constexpr u64 kRflagsCarry = 1;
constexpr u64 kRflagsReserved = 2;

// The XSAVE image: MXCSR and its mask in the legacy area, then the header.
constexpr size_t kXsMxcsr = 0x18;
constexpr size_t kXsMxcsrMask = 0x1c;
constexpr size_t kXsHeader = 0x200;
constexpr size_t kXsHeaderSize = 0x40;

template <typename T>
T Get(const u8* p, size_t off) {
  T v;
  std::memcpy(&v, p + off, sizeof(T));
  return v;
}

template <typename T>
void Put(u8* p, size_t off, T v) {
  std::memcpy(p + off, &v, sizeof(T));
}

bool IsPs5() {
  auto* proc = Process::GetActive();
  return proc && proc->GetPlatform() == Process::Platform::kPs5;
}

// The components XRSTOR may load. A translating backend restores the image
// itself and takes all three.
u64 HostXcr0() {
#if defined(__x86_64__)
  u32 lo, hi;
  asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (u64{hi} << 32) | lo;
#else
  return 7;
#endif
}

// A guest image XRSTOR accepts: MXCSR within the host mask, only components
// the host enables, and a clean header.
void SanitizeXsave(u8* image, u32 mxcsr_mask) {
  Put<u32>(image, kXsMxcsr, Get<u32>(image, kXsMxcsr) & mxcsr_mask);
  const u64 bv = Get<u64>(image, kXsHeader) & HostXcr0() & 7;
  std::memset(image + kXsHeader, 0, kXsHeaderSize);
  Put<u64>(image, kXsHeader, bv);
}

// The guest's view of the syscall's result: rax and the carry flag.
void Return(GuestFrame* f, int result) {
  if (result < 0) {
    f->rax = static_cast<u64>(-result);
    f->rflags |= kRflagsCarry;
  } else {
    f->rax = static_cast<u64>(result);
    f->rflags &= ~kRflagsCarry;
  }
}

}  // namespace

void FillUcontext(const GuestFrame& f,
                  const u8* xsave,
                  const GuestSigset& mask,
                  bool as_syscall_return,
                  u8* uc) {
  std::memset(uc, 0, kUcCopySize);
  std::memcpy(uc, &mask, sizeof(mask));
  u8* mc = uc + kUcMcontext;
  Put(mc, kMcRdi, f.rdi);
  Put(mc, kMcRsi, f.rsi);
  Put(mc, kMcRcx, f.rcx);
  Put(mc, kMcR8, f.r8);
  Put(mc, kMcR9, f.r9);
  Put(mc, kMcRbx, f.rbx);
  Put(mc, kMcRbp, f.rbp);
  Put(mc, kMcR10, f.r10);
  Put(mc, kMcR11, f.r11);
  Put(mc, kMcR12, f.r12);
  Put(mc, kMcR13, f.r13);
  Put(mc, kMcR14, f.r14);
  Put(mc, kMcR15, f.r15);
  Put(mc, kMcRip, f.rip);
  Put(mc, kMcRsp, f.rsp);
  // getcontext's context resumes as the syscall returning 0.
  Put(mc, kMcRax, as_syscall_return ? u64{0} : f.rax);
  Put(mc, kMcRdx, as_syscall_return ? u64{0} : f.rdx);
  Put(mc, kMcRflags, as_syscall_return ? f.rflags & ~kRflagsCarry : f.rflags);
  Put<u32>(mc, kMcFlags, kMcHasSegs);
  Put<u16>(mc, kMcEs, kUserDataSel);
  Put<u16>(mc, kMcDs, kUserDataSel);
  Put<u64>(mc, kMcCs, kUserCodeSel);
  Put<u64>(mc, kMcSs, kUserDataSel);
  Put(mc, kMcLen, kMcSize);
  Put(mc, kMcFpformat, kFpFmtXmm);
  Put(mc, kMcOwnedfp, kFpOwnedFpu);
  std::memcpy(mc + kMcFpstate, xsave, kGuestXsaveSize);
  Put(mc, kMcFsbase, cpu::CurrentGuestFsBase());
}

namespace {
// set_mcontext: validates `mc` and loads it into the frame and image. The fs
// base, when the context carries one, goes to `fsbase`.
int LoadContext(GuestFrame* f,
                u8* xsave,
                const u8* mc,
                bool* has_fsbase,
                u64* fsbase) {
  if (Get<u64>(mc, kMcLen) != kMcSize)
    return -SysError::eINVAL;
  const u32 flags = Get<u32>(mc, kMcFlags);
  if (flags > (IsPs5() ? 7u : 3u))
    return -SysError::eINVAL;

  alignas(64) u8 image[kGuestXsaveSize];
  std::memcpy(image, xsave, sizeof(image));
  const u32 mxcsr_mask = Get<u32>(xsave, kXsMxcsrMask);
  const u64 format = Get<u64>(mc, kMcFpformat);
  if (format == kFpFmtXmm) {
    const u64 owned = Get<u64>(mc, kMcOwnedfp);
    if (owned == kFpOwnedFpu || owned == kFpOwnedPcb) {
      std::memcpy(image, mc + kMcFpstate, sizeof(image));
    } else if (owned == kFpOwnedNone) {
      Put<u64>(image, kXsHeader, 0);  // every component to its init state
    } else {
      return -SysError::eINVAL;
    }
  } else if (format != kFpFmtNodev) {
    return -SysError::eINVAL;
  }
  if (flags & kMcHasFpxstate) {
    // Extended state past the legacy area, by pointer.
    const auto* ext = Get<const u8*>(mc, kMcXfpustate);
    const u64 len = Get<u64>(mc, kMcXfpustateLen);
    if (len > kGuestXsaveSize - kXsHeader)
      return -SysError::eINVAL;
    if (len && !host_memory::IsMemoryRangeMapped(ext, len))
      return -SysError::eFAULT;
    std::memcpy(image + kXsHeader, ext, len);
  }
  SanitizeXsave(image, mxcsr_mask ? mxcsr_mask : 0xffbf);

  std::memcpy(xsave, image, sizeof(image));
  f->rdi = Get<u64>(mc, kMcRdi);
  f->rsi = Get<u64>(mc, kMcRsi);
  f->rdx = Get<u64>(mc, kMcRdx);
  f->rcx = Get<u64>(mc, kMcRcx);
  f->r8 = Get<u64>(mc, kMcR8);
  f->r9 = Get<u64>(mc, kMcR9);
  f->rax = Get<u64>(mc, kMcRax);
  f->rbx = Get<u64>(mc, kMcRbx);
  f->rbp = Get<u64>(mc, kMcRbp);
  f->r10 = Get<u64>(mc, kMcR10);
  f->r11 = Get<u64>(mc, kMcR11);
  f->r12 = Get<u64>(mc, kMcR12);
  f->r13 = Get<u64>(mc, kMcR13);
  f->r14 = Get<u64>(mc, kMcR14);
  f->r15 = Get<u64>(mc, kMcR15);
  f->rip = Get<u64>(mc, kMcRip);
  f->rsp = Get<u64>(mc, kMcRsp);
  f->rflags = (f->rflags & ~kRflagsUser) |
              (Get<u64>(mc, kMcRflags) & kRflagsUser) | kRflagsReserved;
  *has_fsbase = flags & kMcHasBases;
  *fsbase = Get<u64>(mc, kMcFsbase);
  return 0;
}

}  // namespace

int LoadMcontext(GuestFrame* f,
                 u8* xsave,
                 const u8* mc,
                 bool* has_fsbase,
                 u64* fsbase) {
  return LoadContext(f, xsave, mc, has_fsbase, fsbase);
}

namespace {
bool IsGuestRange(const void* p, size_t len) {
  return p && host_memory::IsMemoryRangeMapped(p, len);
}

// Switches the calling thread to the context in `uc`.
int SetContext(GuestFrame* f, u8* xsave, const u8* uc) {
  if (!IsGuestRange(uc, kUcCopySize))
    return -SysError::eFAULT;
  bool has_fsbase;
  u64 fsbase;
  if (const int err =
          LoadContext(f, xsave, uc + kUcMcontext, &has_fsbase, &fsbase))
    return err;
  if (has_fsbase)
    cpu::SetThreadFsBase(fsbase);
  GuestSigset mask;
  std::memcpy(&mask, uc, sizeof(mask));
  SetCurrentSigmask(mask);
  return 0;
}
}  // namespace

bool IsContextSyscall(u32 sid) {
  return sid == kSysSigreturn || sid == kSysGetcontext ||
         sid == kSysSetcontext || sid == kSysSwapcontext;
}

void PS4ABI sys_context(u32 sid, GuestFrame* f, u8* xsave) {
  f->rsp += 8;  // past the lifted call's return address, as after `ret`
  const auto* a0 = reinterpret_cast<u8*>(f->rdi);
  const auto* a1 = reinterpret_cast<u8*>(f->rsi);
  switch (sid) {
    case kSysGetcontext:
      if (!a0)
        return Return(f, -SysError::eINVAL);
      if (!IsGuestRange(a0, kUcCopySize))
        return Return(f, -SysError::eFAULT);
      FillUcontext(*f, xsave, CurrentSigmask(), true, const_cast<u8*>(a0));
      return Return(f, 0);
    case kSysSetcontext:
    case kSysSigreturn:
      if (!a0)
        return Return(f, -SysError::eINVAL);
      if (const int err = SetContext(f, xsave, a0))
        return Return(f, err);
      DeliverSignalToFrame(f, xsave);  // the restored mask may admit one
      return;  // the frame is the new context: no return value
    case kSysSwapcontext:
      if (!a0 || !a1)
        return Return(f, -SysError::eINVAL);
      if (!IsGuestRange(a0, kUcCopySize))
        return Return(f, -SysError::eFAULT);
      FillUcontext(*f, xsave, CurrentSigmask(), true, const_cast<u8*>(a0));
      if (const int err = SetContext(f, xsave, a1))
        return Return(f, err);
      DeliverSignalToFrame(f, xsave);  // the new mask may admit one
      return;
  }
  Return(f, -SysError::eNOSYS);
}

// A new thread that starts in `ucontext`. It keeps the creator's fs base
// unless the context names one.
int PS4ABI sys_thr_create(const void* ucontext, i64* tid, int /*flags*/) {
  const uintptr_t entry = Lv2ContextEntry();
  if (!entry)
    return -SysError::eNOSYS;
  const auto* uc = static_cast<const u8*>(ucontext);
  if (!IsGuestRange(uc, kUcSize))
    return -SysError::eFAULT;

  auto* image = new GuestContextImage{};
  image->frame.rflags = kRflagsReserved;
  Put<u16>(image->xsave, 0, 0x37f);  // x87 control word, as after FNINIT
  Put<u32>(image->xsave, kXsMxcsr, 0x1f80);
  Put<u32>(image->xsave, kXsMxcsrMask, 0xffbf);
  bool has_fsbase;
  u64 fsbase;
  if (const int err = LoadContext(&image->frame, image->xsave, uc + kUcMcontext,
                                  &has_fsbase, &fsbase)) {
    delete image;
    return err;
  }

  thr_param p{};
  p.start_func = reinterpret_cast<void(PS4ABI*)(void*)>(entry);
  p.arg = image;
  p.tls_base =
      reinterpret_cast<u8*>(has_fsbase ? fsbase : cpu::CurrentGuestFsBase());
  p.child_tid = tid;
  const int err = sys_thr_new(&p, sizeof(p));
  if (err)
    delete image;
  return err;
}

}  // namespace kern
