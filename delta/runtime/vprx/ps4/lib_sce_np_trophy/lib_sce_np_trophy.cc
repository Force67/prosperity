/*
 * HLE libSceNpTrophy. See lib_sce_np_trophy.h.
 *
 * Stateful + synchronous: contexts and handles are allocated from small fixed
 * pools (id = slot + 1, matching the real lib); calls validate their ids and
 * return the NpTrophy error codes the SDK uses. We hold no real trophy data, so
 * registration just flips a flag and queries report an empty, all-locked set.
 */

#include "guest_abi.h"
#include "lib_sce_np_trophy.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/logging.h>

#include <options/options.h>
#include <base/containers/array.h>
#include <base/threading/lock_guard.h>
#include <base/threading/mutex.h>

namespace {
DELTA_OPTION(bool, kTrophyTrace, "DELTA_TROPHY_TRACE", false);
}  // namespace

namespace {

// NpTrophy error codes (the 0x8055160x family).
constexpr int OK = 0;
constexpr int ERR_INVALID_ARGUMENT = 0x80551604;
constexpr int ERR_INVALID_HANDLE = 0x80551608;
constexpr int ERR_INVALID_CONTEXT = 0x80551609;
constexpr int ERR_EXCEEDS_MAX = 0x8055160B;  // context/handle pool full
constexpr int ERR_NOT_REGISTERED = 0x8055160F;

constexpr i32 kInvalid = -1;
constexpr int kMaxContexts = 8;  // real-lib ceilings
constexpr int kMaxHandles = 4;
constexpr i32 kInvalidTrophyId = -1;

base::Mutex g_mtx;
base::Array<bool, kMaxContexts> g_ctxUsed{};
base::Array<bool, kMaxContexts> g_ctxReg{};
base::Array<bool, kMaxHandles> g_hndUsed{};


bool ctxValid(i32 c) {
  return c >= 1 && c <= kMaxContexts && g_ctxUsed[c - 1];
}
bool hndValid(i32 h) {
  return h >= 1 && h <= kMaxHandles && g_hndUsed[h - 1];
}

// The game-info / trophy-info structs begin with a caller-set `size`; the caller
// fills it to sizeof(struct) and the kernel writes up to that many bytes. Zero
// the record (an empty/idle set) without overrunning, keeping the size field.
void zeroSized(void *out) {
  if (!out)
    return;
  u64 size = *static_cast<u64 *>(out);
  if (size < sizeof(u64) || size > 0x10000)
    return;
  std::memset(out, 0, size);
  *static_cast<u64 *>(out) = size;
}

}  // namespace

extern "C" {

int PS4ABI sceNpTrophyCreateContext(i32 *context, i32 userId,
                                    u32 serviceLabel, u64 options) {
  if (!context || options != 0ull)
    return ERR_INVALID_ARGUMENT;
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (int i = 0; i < kMaxContexts; i++) {
    if (!g_ctxUsed[i]) {
      g_ctxUsed[i] = true;
      g_ctxReg[i] = false;
      *context = i + 1;
      if (kTrophyTrace)
        BASE_LOGI("trophy", "CreateContext user={} label={:#x} -> ctx={}",
                  userId, serviceLabel, i + 1);
      return OK;
    }
  }
  return ERR_EXCEEDS_MAX;
}

int PS4ABI sceNpTrophyCreateHandle(i32 *handle) {
  if (!handle)
    return ERR_INVALID_ARGUMENT;
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (int i = 0; i < kMaxHandles; i++) {
    if (!g_hndUsed[i]) {
      g_hndUsed[i] = true;
      *handle = i + 1;
      if (kTrophyTrace)
        BASE_LOGI("trophy", "CreateHandle -> handle={}", i + 1);
      return OK;
    }
  }
  return ERR_EXCEEDS_MAX;
}

int PS4ABI sceNpTrophyDestroyContext(i32 context) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  g_ctxUsed[context - 1] = false;
  g_ctxReg[context - 1] = false;
  return OK;
}

int PS4ABI sceNpTrophyDestroyHandle(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  g_hndUsed[handle - 1] = false;
  return OK;
}

int PS4ABI sceNpTrophyAbortHandle(i32 handle) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  return OK;
}

int PS4ABI sceNpTrophyRegisterContext(i32 context, i32 handle,
                                      u64 options) {
  if (options != 0ull)
    return ERR_INVALID_ARGUMENT;
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  g_ctxReg[context - 1] = true;
  if (kTrophyTrace)
    BASE_LOGI("trophy", "RegisterContext ctx={} handle={} -> OK", context,
              handle);
  return OK;
}

int PS4ABI sceNpTrophyUnlockTrophy(i32 context, i32 handle,
                                   i32 trophyId, i32 *platinumId) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  if (!platinumId)
    return ERR_INVALID_ARGUMENT;
  *platinumId = kInvalidTrophyId;  // no platinum awarded (we persist nothing)
  return OK;
}

int PS4ABI sceNpTrophyGetTrophyUnlockState(i32 context, i32 handle,
                                           void *flags, u32 *count) {
  if (!flags || !count)
    return ERR_INVALID_ARGUMENT;
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  std::memset(flags, 0, 16);  // OrbisNpTrophyFlagArray: 128 bits, none unlocked
  *count = 0;                 // empty trophy set
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetTrophyUnlockState ctx={} -> count=0", context);
  return OK;
}

int PS4ABI sceNpTrophyGetGameInfo(i32 context, i32 handle, void *details,
                                  void *data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!details || !data)
    return ERR_INVALID_ARGUMENT;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  zeroSized(details);  // OrbisNpTrophyGameDetails (0x4A0): 0 trophies
  zeroSized(data);     // OrbisNpTrophyGameData (0x20): 0 unlocked
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetGameInfo ctx={} -> OK", context);
  return OK;
}

int PS4ABI sceNpTrophyGetTrophyInfo(i32 context, i32 handle,
                                    i32 trophyId, void *details,
                                    void *data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!details || !data)
    return ERR_INVALID_ARGUMENT;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  zeroSized(details);
  zeroSized(data);
  return OK;
}

int PS4ABI sceNpTrophyGetGroupInfo(i32 context, i32 handle,
                                   i32 groupId, void *details, void *data) {
  base::LockGuard<base::Mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!details || !data)
    return ERR_INVALID_ARGUMENT;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  zeroSized(details);
  zeroSized(data);
  if (kTrophyTrace)
    BASE_LOGI("trophy", "GetGroupInfo ctx={} group={} -> OK", context, groupId);
  return OK;
}

// Icon getters: we ship no trophy icons. The two-call protocol is size-query
// (buffer == null -> write the byte size) then fetch (buffer != null -> fill).
// Report a 0-byte icon so callers display nothing instead of erroring/looping.
static int trophyIcon(void *buffer, u64 *size) {
  if (!size)
    return ERR_INVALID_ARGUMENT;
  if (!buffer)
    *size = 0;
  return OK;
}

int PS4ABI sceNpTrophyGetGameIcon(i32 context, i32 handle, void *buffer,
                                  u64 *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetGroupIcon(i32 context, i32 handle,
                                   i32 groupId, void *buffer,
                                   u64 *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetTrophyIcon(i32 context, i32 handle,
                                    i32 trophyId, void *buffer,
                                    u64 *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyCaptureScreenshot(i32 a, void *b, void *c) {
  return OK;  // no screenshot pipeline; accept and drop
}

int PS4ABI sceNpTrophyShowTrophyList(i32 context, i32 handle) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  return OK;  // no trophy-list UI; treat as shown
}

}  // extern "C"
