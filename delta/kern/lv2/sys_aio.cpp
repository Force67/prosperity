/*
 * PS4Delta : PS4 emulation and research project
 *
 * Async file IO. Failing these with ENOTSUP was supposed to push a guest onto a
 * synchronous fallback; GTA:SA has none. It submits 23 read commands during
 * engine start-up, takes the error, and never reads another byte -- the title
 * reaches its main menu and stays behind the transition curtain forever.
 *
 * Serving the reads inside submit and reporting the request complete is a legal
 * schedule (a request may finish before the caller ever polls), and it needs no
 * IO thread of its own.
 */

#include "sys_aio.h"

#include <mutex>
#include <unordered_map>

#include <base/logging.h>
#include <utl/mem.h>
#include <utl/options.h>

#include "error_table.h"
#include "sys_vfs_ext.h"

namespace {
DELTA_OPTION(bool, kAioTrace, "DELTA_AIO_TRACE", false);
}  // namespace

namespace krnl {
namespace {

// SceKernelAioRWRequest.
struct AioRequest {
  i64 offset;
  u64 nbyte;
  void *buf;
  void *result;  // SceKernelAioResult *
  i32 fd;
  i32 pad;
};

// SceKernelAioResult.
struct AioResult {
  i64 returnValue;
  u32 state;
  u32 pad;
};

enum : u32 {
  kStateSubmitted = 1,
  kStateProcessing = 2,
  kStateCompleted = 3,
  kStateAborted = 4,
};

enum : u32 {
  kCmdRead = 1,
  kCmdWrite = 2,
  kCmdMask = 0xfff,
  kCmdMulti = 0x1000,
};

// Live submit ids. The value is the SCE error the request finished with, which
// is what wait/poll/delete report back per id.
std::mutex g_mutex;
std::unordered_map<u32, int> g_requests;
u32 g_nextId = 1;

// One request, start to finish. Returns the SCE error for the id.
int runRequest(u32 cmd, AioRequest &req) {
  i64 done = -SysError::eBADF;
  if (req.buf && req.nbyte) {
    done = (cmd & kCmdMask) == kCmdWrite
               ? sys_pwrite(static_cast<u32>(req.fd), req.buf, req.nbyte,
                            req.offset)
               : sys_pread(static_cast<u32>(req.fd), req.buf, req.nbyte,
                           req.offset);
  } else if (!req.nbyte) {
    done = 0;
  }
  if (kAioTrace)
    BASE_LOGI("aio", "cmd={:#x} fd={} off={} nbyte={:#x} -> {}", cmd, req.fd,
              (long long)req.offset, req.nbyte, (long long)done);
  const int err = done < 0 ? static_cast<int>(-done) : 0;
  if (req.result &&
      utl::isMemoryRangeMapped(req.result, sizeof(AioResult))) {
    auto *out = static_cast<AioResult *>(req.result);
    out->returnValue = done;
    out->state = err ? kStateAborted : kStateCompleted;
  }
  return err ? static_cast<int>(0x80020000u | static_cast<u32>(err)) : 0;
}

// Report `num` ids as finished. An id we never handed out still answers 0: the
// caller is about to drop it either way, and refusing one it believes in is
// what turns a stale id into a stalled loader.
int reportIds(const u32 *ids, u32 num, int *errs, bool erase) {
  std::lock_guard<std::mutex> lock(g_mutex);
  for (u32 i = 0; i < num; i++) {
    int err = 0;
    if (ids && utl::isMemoryRangeMapped(ids + i, sizeof(u32))) {
      auto it = g_requests.find(ids[i]);
      if (it != g_requests.end()) {
        err = it->second;
        if (erase)
          g_requests.erase(it);
      }
    }
    if (errs && utl::isMemoryRangeMapped(errs + i, sizeof(int)))
      errs[i] = err;
  }
  return 0;
}

}  // namespace

int PS4ABI sys_aio_submit_cmd(u32 cmd, void *reqs, u32 num, u32 prio,
                              u32 *ids) {
  if (!reqs || !num)
    return -SysError::eINVAL;
  if (!utl::isMemoryRangeMapped(reqs, sizeof(AioRequest) * num))
    return -SysError::eFAULT;
  auto *req = static_cast<AioRequest *>(reqs);

  // Without SCE_KERNEL_AIO_CMD_MULTI the whole batch shares ONE id and the
  // caller passed a pointer to a single SceKernelAioSubmitId. Writing one id
  // per request there overruns its stack.
  const bool multi = (cmd & kCmdMulti) != 0;
  const u32 idCount = multi ? num : 1;
  if (!ids || !utl::isMemoryRangeMapped(ids, sizeof(u32) * idCount))
    return -SysError::eFAULT;

  int worst = 0;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (u32 i = 0; i < idCount; i++)
      ids[i] = g_nextId++;
  }
  for (u32 i = 0; i < num; i++) {
    const int err = runRequest(cmd, req[i]);
    std::lock_guard<std::mutex> lock(g_mutex);
    // A shared id carries the first failure of its batch; a per-request id
    // carries its own.
    if (multi)
      g_requests[ids[i]] = err;
    else if (err && !worst)
      worst = err;
  }
  if (!multi) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_requests[ids[0]] = worst;
  }
  return 0;
}

int PS4ABI sys_aio_submit(u32 cmd, void *reqs, u32 num, u32 prio, u32 *ids) {
  return sys_aio_submit_cmd(cmd, reqs, num, prio, ids);
}

int PS4ABI sys_aio_multi_wait(u32 *ids, u32 num, int *errs, u32 /*mode*/,
                              u32 * /*usec*/) {
  return reportIds(ids, num, errs, false);
}

int PS4ABI sys_aio_multi_poll(u32 *ids, u32 num, int *errs) {
  return reportIds(ids, num, errs, false);
}

int PS4ABI sys_aio_multi_delete(u32 *ids, u32 num, int *errs) {
  return reportIds(ids, num, errs, true);
}

int PS4ABI sys_aio_multi_cancel(u32 *ids, u32 num, int *errs) {
  // Nothing is ever in flight, so a cancel can only report the finished state.
  return reportIds(ids, num, errs, false);
}

}  // namespace krnl
