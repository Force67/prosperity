/*
 * HLE libSceNpTrophy. See libSceNpTrophy.h.
 *
 * Stateful + synchronous: contexts and handles are allocated from small fixed
 * pools (id = slot + 1, matching the real lib); calls validate their ids and
 * return the NpTrophy error codes the SDK uses. We hold no real trophy data, so
 * registration just flips a flag and queries report an empty, all-locked set.
 */

#include "libSceNpTrophy.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utl/options.h>

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

constexpr int32_t kInvalid = -1;
constexpr int kMaxContexts = 8;  // real-lib ceilings
constexpr int kMaxHandles = 4;
constexpr int32_t kInvalidTrophyId = -1;

std::mutex g_mtx;
std::array<bool, kMaxContexts> g_ctxUsed{};
std::array<bool, kMaxContexts> g_ctxReg{};
std::array<bool, kMaxHandles> g_hndUsed{};


bool ctxValid(int32_t c) {
  return c >= 1 && c <= kMaxContexts && g_ctxUsed[c - 1];
}
bool hndValid(int32_t h) {
  return h >= 1 && h <= kMaxHandles && g_hndUsed[h - 1];
}

// The game-info / trophy-info structs begin with a caller-set `size`; the caller
// fills it to sizeof(struct) and the kernel writes up to that many bytes. Zero
// the record (an empty/idle set) without overrunning, keeping the size field.
void zeroSized(void *out) {
  if (!out)
    return;
  uint64_t size = *static_cast<uint64_t *>(out);
  if (size < sizeof(uint64_t) || size > 0x10000)
    return;
  std::memset(out, 0, size);
  *static_cast<uint64_t *>(out) = size;
}

}  // namespace

extern "C" {

int PS4ABI sceNpTrophyCreateContext(int32_t *context, int32_t userId,
                                    uint32_t serviceLabel, uint64_t options) {
  if (!context || options != 0ull)
    return ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(g_mtx);
  for (int i = 0; i < kMaxContexts; i++) {
    if (!g_ctxUsed[i]) {
      g_ctxUsed[i] = true;
      g_ctxReg[i] = false;
      *context = i + 1;
      if (kTrophyTrace)
        std::printf("[trophy] CreateContext user=%d label=%#x -> ctx=%d\n",
                    userId, serviceLabel, i + 1);
      return OK;
    }
  }
  return ERR_EXCEEDS_MAX;
}

int PS4ABI sceNpTrophyCreateHandle(int32_t *handle) {
  if (!handle)
    return ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(g_mtx);
  for (int i = 0; i < kMaxHandles; i++) {
    if (!g_hndUsed[i]) {
      g_hndUsed[i] = true;
      *handle = i + 1;
      if (kTrophyTrace)
        std::printf("[trophy] CreateHandle -> handle=%d\n", i + 1);
      return OK;
    }
  }
  return ERR_EXCEEDS_MAX;
}

int PS4ABI sceNpTrophyDestroyContext(int32_t context) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  g_ctxUsed[context - 1] = false;
  g_ctxReg[context - 1] = false;
  return OK;
}

int PS4ABI sceNpTrophyDestroyHandle(int32_t handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  g_hndUsed[handle - 1] = false;
  return OK;
}

int PS4ABI sceNpTrophyAbortHandle(int32_t handle) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  return OK;
}

int PS4ABI sceNpTrophyRegisterContext(int32_t context, int32_t handle,
                                      uint64_t options) {
  if (options != 0ull)
    return ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  g_ctxReg[context - 1] = true;
  if (kTrophyTrace)
    std::printf("[trophy] RegisterContext ctx=%d handle=%d -> OK\n", context,
                handle);
  return OK;
}

int PS4ABI sceNpTrophyUnlockTrophy(int32_t context, int32_t handle,
                                   int32_t trophyId, int32_t *platinumId) {
  std::lock_guard<std::mutex> lk(g_mtx);
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

int PS4ABI sceNpTrophyGetTrophyUnlockState(int32_t context, int32_t handle,
                                           void *flags, uint32_t *count) {
  if (!flags || !count)
    return ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  if (!g_ctxReg[context - 1])
    return ERR_NOT_REGISTERED;
  std::memset(flags, 0, 16);  // OrbisNpTrophyFlagArray: 128 bits, none unlocked
  *count = 0;                 // empty trophy set
  if (kTrophyTrace)
    std::printf("[trophy] GetTrophyUnlockState ctx=%d -> count=0\n", context);
  return OK;
}

int PS4ABI sceNpTrophyGetGameInfo(int32_t context, int32_t handle, void *details,
                                  void *data) {
  std::lock_guard<std::mutex> lk(g_mtx);
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
    std::printf("[trophy] GetGameInfo ctx=%d -> OK\n", context);
  return OK;
}

int PS4ABI sceNpTrophyGetTrophyInfo(int32_t context, int32_t handle,
                                    int32_t trophyId, void *details,
                                    void *data) {
  std::lock_guard<std::mutex> lk(g_mtx);
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

int PS4ABI sceNpTrophyGetGroupInfo(int32_t context, int32_t handle,
                                   int32_t groupId, void *details, void *data) {
  std::lock_guard<std::mutex> lk(g_mtx);
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
    std::printf("[trophy] GetGroupInfo ctx=%d group=%d -> OK\n", context,
                groupId);
  return OK;
}

// Icon getters: we ship no trophy icons. The two-call protocol is size-query
// (buffer == null -> write the byte size) then fetch (buffer != null -> fill).
// Report a 0-byte icon so callers display nothing instead of erroring/looping.
static int trophyIcon(void *buffer, uint64_t *size) {
  if (!size)
    return ERR_INVALID_ARGUMENT;
  if (!buffer)
    *size = 0;
  return OK;
}

int PS4ABI sceNpTrophyGetGameIcon(int32_t context, int32_t handle, void *buffer,
                                  uint64_t *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetGroupIcon(int32_t context, int32_t handle,
                                   int32_t groupId, void *buffer,
                                   uint64_t *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyGetTrophyIcon(int32_t context, int32_t handle,
                                    int32_t trophyId, void *buffer,
                                    uint64_t *size) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  return trophyIcon(buffer, size);
}

int PS4ABI sceNpTrophyCaptureScreenshot(int32_t a, void *b, void *c) {
  return OK;  // no screenshot pipeline; accept and drop
}

int PS4ABI sceNpTrophyShowTrophyList(int32_t context, int32_t handle) {
  if (!ctxValid(context))
    return ERR_INVALID_CONTEXT;
  if (!hndValid(handle))
    return ERR_INVALID_HANDLE;
  return OK;  // no trophy-list UI; treat as shown
}

}  // extern "C"
