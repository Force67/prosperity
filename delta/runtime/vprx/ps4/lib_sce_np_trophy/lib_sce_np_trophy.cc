/*
 * HLE libSceNpTrophy. See lib_sce_np_trophy.h.
 *
 * Stateful + synchronous: contexts and handles are allocated from small fixed
 * pools (id = slot + 1, matching the real lib); calls validate their ids and
 * return the NpTrophy error codes the SDK uses. We hold no real trophy data, so
 * registration just flips a flag and queries report an empty, all-locked set.
 */

#include "runtime/vprx/ps4/lib_sce_np_trophy/lib_sce_np_trophy.h"
#include "base/arch.h"
#include "guest/session.h"
#include "guest_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/logging.h"

#include "base/containers/array.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kTrophyTrace, "DELTA_TROPHY_TRACE", false);
}  // namespace

namespace {

// NpTrophy error codes (the 0x8055160x family).
constexpr int kOk = 0;
constexpr int kErrInvalidArgument = 0x80551604;
constexpr int kErrInvalidHandle = 0x80551608;
constexpr int kErrInvalidContext = 0x80551609;
constexpr int kErrExceedsMax = 0x8055160B;  // context/handle pool full
constexpr int kErrNotRegistered = 0x8055160F;

constexpr i32 kInvalid = -1;
constexpr int kMaxContexts = 8;  // real-lib ceilings
constexpr int kMaxHandles = 4;
constexpr i32 kInvalidTrophyId = -1;

base::Mutex g_mtx;
base::Array<bool, kMaxContexts> g_ctx_used{};
base::Array<bool, kMaxContexts> g_ctx_reg{};
base::Array<bool, kMaxHandles> g_hnd_used{};

bool CtxValid(i32 c) {
  return c >= 1 && c <= kMaxContexts && g_ctx_used[c - 1];
}
bool HndValid(i32 h) {
  return h >= 1 && h <= kMaxHandles && g_hnd_used[h - 1];
}

// The game-info / trophy-info structs begin with a caller-set `size`; the
// caller fills it to sizeof(struct) and the kernel writes up to that many
// bytes. Zero the record (an empty/idle set) without overrunning, keeping the
// size field.
void ZeroSized(void* out) {
  if (!out)
    return;
  u64 size = *static_cast<u64*>(out);
  if (size < sizeof(u64) || size > 0x10000)
    return;
  std::memset(out, 0, size);
  *static_cast<u64*>(out) = size;
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_ctx_used);
  guest::ResetResource(g_ctx_reg);
  guest::ResetResource(g_hnd_used);
});
}  // namespace

}  // namespace

extern "C" {

int PS4ABI sceNpTrophyCreateContext(i32* context,
                                    i32 user_id,
                                    u32 service_label,
                                    u64 options) {
  if (!context || options != 0ull)
    return kErrInvalidArgument;
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (int i = 0; i < kMaxContexts; i++) {
    if (!g_ctx_used[i]) {
      g_ctx_used[i] = true;
      g_ctx_reg[i] = false;
      *context = i + 1;
      if (kTrophyTrace)
        BASE_LOGI("trophy", "CreateContext user={} label={:#x} -> ctx={}",
                  user_id, service_label, i + 1);
      return kOk;
    }
  }
  return kErrExceedsMax;
}

int PS4ABI sceNpTrophyCreateHandle(i32* handle) {
  if (!handle)
    return kErrInvalidArgument;
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (int i = 0; i < kMaxHandles; i++) {
    if (!g_hnd_used[i]) {
      g_hnd_used[i] = true;
      *handle = i + 1;
      if (kTrophyTrace)
        BASE_LOGI("trophy", "CreateHandle -> handle={}", i + 1);
      return kOk;
    }
  }
  return kErrExceedsMax;
}

int PS4ABI sceNpTrophyDestroyContext(i32 context) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  g_ctx_used[context - 1] = false;
  g_ctx_reg[context - 1] = false;
  return kOk;
}

int PS4ABI sceNpTrophyDestroyHandle(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!HndValid(handle))
    return kErrInvalidHandle;
  g_hnd_used[handle - 1] = false;
  return kOk;
}

int PS4ABI sceNpTrophyAbortHandle(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!HndValid(handle))
    return kErrInvalidHandle;
  return kOk;
}

int PS4ABI sceNpTrophyRegisterContext(i32 context, i32 handle, u64 options) {
  if (options != 0ull)
    return kErrInvalidArgument;
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  g_ctx_reg[context - 1] = true;
  if (kTrophyTrace)
    BASE_LOGI("trophy", "RegisterContext ctx={} handle={} -> OK", context,
              handle);
  return kOk;
}

int PS4ABI sceNpTrophyUnlockTrophy(i32 context,
                                   i32 handle,
                                   i32 trophy_id,
                                   i32* platinum_id) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  if (!g_ctx_reg[context - 1])
    return kErrNotRegistered;
  if (!platinum_id)
    return kErrInvalidArgument;
  *platinum_id = kInvalidTrophyId;  // no platinum awarded (we persist nothing)
  return kOk;
}

int PS4ABI sceNpTrophyGetTrophyUnlockState(i32 context,
                                           i32 handle,
                                           void* flags,
                                           u32* count) {
  if (!flags || !count)
    return kErrInvalidArgument;
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  if (!g_ctx_reg[context - 1])
    return kErrNotRegistered;
  std::memset(flags, 0, 16);  // OrbisNpTrophyFlagArray: 128 bits, none unlocked
  *count = 0;                 // empty trophy set
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetTrophyUnlockState ctx={} -> count=0", context);
  return kOk;
}

int PS4ABI sceNpTrophyGetGameInfo(i32 context,
                                  i32 handle,
                                  void* details,
                                  void* data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  if (!details || !data)
    return kErrInvalidArgument;
  if (!g_ctx_reg[context - 1])
    return kErrNotRegistered;
  ZeroSized(details);  // OrbisNpTrophyGameDetails (0x4A0): 0 trophies
  ZeroSized(data);     // OrbisNpTrophyGameData (0x20): 0 unlocked
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetGameInfo ctx={} -> OK", context);
  return kOk;
}

int PS4ABI sceNpTrophyGetTrophyInfo(i32 context,
                                    i32 handle,
                                    i32 trophy_id,
                                    void* details,
                                    void* data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  if (!details || !data)
    return kErrInvalidArgument;
  if (!g_ctx_reg[context - 1])
    return kErrNotRegistered;
  ZeroSized(details);
  ZeroSized(data);
  return kOk;
}

int PS4ABI sceNpTrophyGetGroupInfo(i32 context,
                                   i32 handle,
                                   i32 group_id,
                                   void* details,
                                   void* data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  if (!details || !data)
    return kErrInvalidArgument;
  if (!g_ctx_reg[context - 1])
    return kErrNotRegistered;
  ZeroSized(details);
  ZeroSized(data);
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetGroupInfo ctx={} group={} -> OK", context,
              group_id);
  return kOk;
}

// Icon getters: we ship no trophy icons. The two-call protocol is size-query
// (buffer == null -> write the byte size) then fetch (buffer != null -> fill).
// Report a 0-byte icon so callers display nothing instead of erroring/looping.
static int TrophyIcon(void* buffer, u64* size) {
  if (!size)
    return kErrInvalidArgument;
  if (!buffer)
    *size = 0;
  return kOk;
}

int PS4ABI sceNpTrophyGetGameIcon(i32 context,
                                  i32 handle,
                                  void* buffer,
                                  u64* size) {
  if (!CtxValid(context))
    return kErrInvalidContext;
  return TrophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetGroupIcon(i32 context,
                                   i32 handle,
                                   i32 group_id,
                                   void* buffer,
                                   u64* size) {
  if (!CtxValid(context))
    return kErrInvalidContext;
  return TrophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetTrophyIcon(i32 context,
                                    i32 handle,
                                    i32 trophy_id,
                                    void* buffer,
                                    u64* size) {
  if (!CtxValid(context))
    return kErrInvalidContext;
  return TrophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyCaptureScreenshot(i32 a, void* b, void* c) {
  return kOk;  // no screenshot pipeline; accept and drop
}

int PS4ABI sceNpTrophyShowTrophyList(i32 context, i32 handle) {
  if (!CtxValid(context))
    return kErrInvalidContext;
  if (!HndValid(handle))
    return kErrInvalidHandle;
  return kOk;  // no trophy-list UI; treat as shown
}

}  // extern "C"
