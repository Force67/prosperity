/*
 * PS4Delta : PS4 emulation and research project
 *
 * Async file IO. Failing these with ENOTSUP was supposed to push a guest onto a
 * synchronous fallback; GTA:SA has none. It submits 23 read commands during
 * engine start-up, takes the error, and never reads another byte, so the title
 * reaches its main menu and stays behind the transition curtain forever.
 *
 * Serving the reads inside submit and reporting the request complete is a legal
 * schedule (a request may finish before the caller ever polls), and it needs no
 * IO thread of its own.
 */

#include "kern/lv2/sys_aio.h"
#include "guest_abi.h"

#include "base/logging.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/stub_log.h"
#include "kern/lv2/sys_vfs.h"

namespace {
DELTA_OPTION(bool, kAioTrace, "DELTA_AIO_TRACE", false);
}  // namespace

namespace kern {
namespace {

// SceKernelAioRWRequest.
struct AioRequest {
  i64 offset;
  u64 nbyte;
  void* buf;
  void* result;  // SceKernelAioResult *
  i32 fd;
  i32 pad;
};

// SceKernelAioResult.
struct AioResult {
  i64 return_value;
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
base::Mutex g_mutex;
base::HashMap<u32, int> g_requests;
u32 g_next_id = 1;

// One request, start to finish. Returns the SCE error for the id.
int RunRequest(u32 cmd, AioRequest& req) {
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
      host_memory::IsMemoryRangeMapped(req.result, sizeof(AioResult))) {
    auto* out = static_cast<AioResult*>(req.result);
    out->return_value = done;
    out->state = err ? kStateAborted : kStateCompleted;
  }
  return err ? static_cast<int>(0x80020000u | static_cast<u32>(err)) : 0;
}

// Report `num` ids as finished. An id we never handed out still answers 0: the
// caller is about to drop it either way, and refusing one it believes in is
// what turns a stale id into a stalled loader.
int ReportIds(const u32* ids, u32 num, int* errs, bool erase) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  for (u32 i = 0; i < num; i++) {
    int err = 0;
    if (ids && host_memory::IsMemoryRangeMapped(ids + i, sizeof(u32))) {
      auto it = g_requests.find(ids[i]);
      if (it != g_requests.end()) {
        err = it->second;
        if (erase)
          g_requests.erase(it);
      }
    }
    if (errs && host_memory::IsMemoryRangeMapped(errs + i, sizeof(int)))
      errs[i] = err;
  }
  return 0;
}

}  // namespace

int PS4ABI
sys_aio_submit_cmd(u32 cmd, void* reqs, u32 num, u32 prio, u32* ids) {
  if (!reqs || !num)
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(reqs, sizeof(AioRequest) * num))
    return -SysError::eFAULT;
  auto* req = static_cast<AioRequest*>(reqs);

  // Without SCE_KERNEL_AIO_CMD_MULTI the whole batch shares ONE id and the
  // caller passed a pointer to a single SceKernelAioSubmitId. Writing one id
  // per request there overruns its stack.
  const bool multi = (cmd & kCmdMulti) != 0;
  const u32 id_count = multi ? num : 1;
  if (!ids || !host_memory::IsMemoryRangeMapped(ids, sizeof(u32) * id_count))
    return -SysError::eFAULT;

  int worst = 0;
  {
    base::LockGuard<base::Mutex> lock(g_mutex);
    for (u32 i = 0; i < id_count; i++)
      ids[i] = g_next_id++;
  }
  for (u32 i = 0; i < num; i++) {
    const int err = RunRequest(cmd, req[i]);
    base::LockGuard<base::Mutex> lock(g_mutex);
    // A shared id carries the first failure of its batch; a per-request id
    // carries its own.
    if (multi)
      g_requests[ids[i]] = err;
    else if (err && !worst)
      worst = err;
  }
  if (!multi) {
    base::LockGuard<base::Mutex> lock(g_mutex);
    g_requests[ids[0]] = worst;
  }
  return 0;
}

int PS4ABI sys_aio_submit(u32 cmd, void* reqs, u32 num, u32 prio, u32* ids) {
  return sys_aio_submit_cmd(cmd, reqs, num, prio, ids);
}

int PS4ABI
sys_aio_multi_wait(u32* ids, u32 num, int* errs, u32 /*mode*/, u32* /*usec*/) {
  return ReportIds(ids, num, errs, false);
}

int PS4ABI sys_aio_multi_poll(u32* ids, u32 num, int* errs) {
  return ReportIds(ids, num, errs, false);
}

int PS4ABI sys_aio_multi_delete(u32* ids, u32 num, int* errs) {
  return ReportIds(ids, num, errs, true);
}

int PS4ABI sys_aio_multi_cancel(u32* ids, u32 num, int* errs) {
  // Nothing is ever in flight, so a cancel can only report the finished state.
  return ReportIds(ids, num, errs, false);
}

// We don't model async IO. Failing with eOPNOTSUPP makes guests fall back to
// synchronous IO. Every AIO entry point funnels here, so the log can't name
// which one; pair it with FEX_SCTRACE to attribute the call.
int PS4ABI sys_aio_unsupported() {
  static base::Atomic<int> n{0};
  int c = ++n;
  if (kAioTrace && c <= 200)
    BASE_LOGI("aio", "unsupported call #{}", c);
  else {
    static base::Atomic<bool> once{false};
    LogOnce(once, "aio unsupported; guest should fall back to sync IO");
  }
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_get_bio_usage_all() {
  return 0;
}
int PS4ABI sys_aio_init() {
  return 0;
}

}  // namespace kern
