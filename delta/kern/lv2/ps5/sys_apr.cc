/*
 * PS4Delta : PS4 emulation and research project
 *
 * APR (0x2bc..0x2bf): the PS5's asynchronous package reader. A title resolves
 * paths to file ids, then submits command buffers (built by libSceAmpr) that
 * read files straight into memory, map memory, wait on and write addresses and
 * counters, and post events. The kernel hands them to the I/O coprocessor; here
 * one worker per priority runs them in order, as the coprocessor does.
 */

#include <cstring>
#include <ctime>
#include "base/arch.h"
#include "base/containers/deque.h"
#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"
#include "io/file.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_thread.h"
#include "kern/process.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/vfs.h"
#include "options/options.h"

DELTA_OPTION(bool, kAprTrace, "DELTA_APR_TRACE", false);

namespace kern {
namespace {
constexpr u32 kResolveMax = 1024;
constexpr u32 kPathMax = 1024;
constexpr u64 kBufferMax = 64ull << 20;
constexpr u32 kPendingMax = 0x1000;  // completions not yet waited for
constexpr u32 kPriorities = 7;       // 1..6
constexpr i16 kEvfiltAmpr = -25;

// SceAprResultBuffer error codes.
constexpr int kErrOverrun = static_cast<int>(0x81912001);
constexpr int kErrOpcode = static_cast<int>(0x81912002);
constexpr int kErrFormat = static_cast<int>(0x81912004);
constexpr int kErrCmdAddress = static_cast<int>(0x81912005);
constexpr int kErrWaitAddress = static_cast<int>(0x81912006);
constexpr int kErrWriteSource = static_cast<int>(0x81912007);
constexpr int kErrVaRange = static_cast<int>(0x81912008);
constexpr int kErrFaultWaitAddress = static_cast<int>(0x81912019);
constexpr int kErrFaultWrite = static_cast<int>(0x81912021);
constexpr int kErrFileId = static_cast<int>(0x81912039);
constexpr int kErrFileOffset = static_cast<int>(0x8191203a);
constexpr int kErrGatherScatter = static_cast<int>(0x8191203b);
constexpr int kErrEventQueue = static_cast<int>(0x8191203d);
constexpr int kErrMapState = static_cast<int>(0x8191203f);

enum Op : u8 {
  kWaitOnAddress = 0x01,
  kWaitOnCounter = 0x02,
  kWriteAddress = 0x05,
  kWriteCounter = 0x06,
  kWriteEventQueue = 0x08,
  kNop = 0x0f,
  kReadFile = 0x28,
  kReadGather = 0x29,
  kReadScatter = 0x2a,
  kReadGatherScatter = 0x2b,
  kMapBegin = 0x2d,
  kMapEnd = 0x2e,
  kResetGatherScatter = 0x2f,
  kImmediate = 0x70,  // the "immediately" form of a write
};

struct AprFile {
  base::String path;
  i64 size = 0;
  base::Mutex m;
  io::File file;  // opened on first read
};

struct Request {
  const u32* cmd;
  u32 ndw;
  u32 id;  // 0: nobody waits for it
};

struct Completion {
  bool done = false;
  int result = 0;
  u32 error_offset = 0;
  u32 waiter = 0;
  u32* result_ptr = nullptr;
};

base::Mutex g_files_m;
base::Vector<base::UniquePointer<AprFile>> g_files;
base::HashMap<base::String, u32> g_file_ids;

base::Mutex g_m;
base::ConditionVariable g_cv;
base::SimpleDeque<Request> g_queues[kPriorities];
bool g_workers[kPriorities];
base::HashMap<u32, Completion> g_completions;
u32 g_next_id = 1;
u32 g_counters[256];

const guest::SessionReset g_reset([] {
  {
    base::LockGuard<base::Mutex> lk(g_files_m);
    guest::ResetResource(g_files);
    guest::ResetResource(g_file_ids);
  }
  base::LockGuard<base::Mutex> lk(g_m);
  for (auto& q : g_queues)
    guest::ResetResource(q);
  for (auto& w : g_workers)
    w = false;
  guest::ResetResource(g_completions);
  g_next_id = 1;
  std::memset(g_counters, 0, sizeof(g_counters));
});

// A file id is the index into g_files: one byte of mount, 24 bits of file.
AprFile* FileOf(u32 id) {
  base::LockGuard<base::Mutex> lk(g_files_m);
  return id < g_files.size() ? &*g_files[id] : nullptr;
}

int Resolve(const char* path, u32* id, i64* size) {
  i64 sz = 0;
  bool is_dir = false;
  if (!vfs::Stat(path, sz, is_dir))
    return SysError::eNOENT;
  if (is_dir)
    return SysError::eINVAL;  // no file system serves a directory to APR
  base::LockGuard<base::Mutex> lk(g_files_m);
  const base::String key(path);
  auto it = g_file_ids.find(key);
  if (it == g_file_ids.end()) {
    if (g_files.size() > 0xffffff)
      return SysError::eNOSPC;
    auto f = base::MakeUnique<AprFile>();
    f->path = key;
    f->size = sz;
    g_files.push_back(base::move(f));
    it = g_file_ids.emplace(key, static_cast<u32>(g_files.size() - 1)).first;
  }
  *id = it->second;
  *size = sz;
  return 0;
}

// Reads `len` bytes at `off` of file `id` into guest memory at `dst`.
int ReadFile(u32 id, u64 dst, u64 len, u64 off) {
  AprFile* f = FileOf(id & 0x7fffffff);
  if (!f)
    return kErrFileId;
  if (off > static_cast<u64>(f->size))
    return kErrFileOffset;
  auto* p = reinterpret_cast<u8*>(dst);
  if (!host_memory::IsMemoryRangeMapped(p, len))
    return kErrFaultWrite;
  const u64 n = base::Min<u64>(len, static_cast<u64>(f->size) - off);
  if (!n)
    return 0;
  base::LockGuard<base::Mutex> lk(f->m);
  if (!f->file.Exists())
    f->file.Reset(vfs::OpenRead(f->path.c_str()).GetBase());
  if (!f->file.Exists())
    return kErrFileId;
  host_memory::BeforeHostWrite(p, n);
  f->file.Seek(off, io::SeekMode::kSeekSet);
  f->file.Read(p, n);
  return 0;
}

u64 TimeCounter() {
  timespec ts{};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64(ts.tv_sec) * 1'000'000'000 + u64(ts.tv_nsec)) / 10;  // 100 MHz
}

// Counter access: 0 a counter pair, 1 one counter, 2..3 a half, 4..7 a byte.
u32 AccessBits(u32 access) {
  return access == 0 ? 64 : access == 1 ? 32 : access < 4 ? 16 : 8;
}

u32 AccessShift(u32 access) {
  return access < 2 ? 0 : access < 4 ? (access - 2) * 16 : (access - 4) * 8;
}

// Under g_m.
u64 ReadCounter(u32 idx, u32 access) {
  if (access == 0)
    return g_counters[idx & 0xfe] | u64(g_counters[(idx & 0xfe) + 1]) << 32;
  const u32 bits = AccessBits(access);
  const u64 mask = bits == 32 ? 0xffffffffull : (1ull << bits) - 1;
  return (g_counters[idx] >> AccessShift(access)) & mask;
}

// Under g_m.
void WriteCounter(u32 idx, u32 access, u64 value, u32 op) {
  u64 v = ReadCounter(idx, access);
  switch (op) {
    case 1:
      v |= value;
      break;
    case 2:
      v &= ~value;
      break;
    case 3:
      v ^= value;
      break;
    case 4:
      v += value;
      break;
    default:
      v = value;
      break;
  }
  if (access == 0) {
    g_counters[idx & 0xfe] = static_cast<u32>(v);
    g_counters[(idx & 0xfe) + 1] = static_cast<u32>(v >> 32);
    return;
  }
  const u32 bits = AccessBits(access);
  const u32 shift = AccessShift(access);
  const u32 mask = (bits == 32 ? 0xffffffffu : (1u << bits) - 1) << shift;
  g_counters[idx] =
      (g_counters[idx] & ~mask) | ((static_cast<u32>(v) << shift) & mask);
}

i64 SignExtend(u64 v, u32 bits) {
  return bits == 64 ? static_cast<i64>(v)
                    : static_cast<i64>(v << (64 - bits)) >> (64 - bits);
}

bool Compare(u64 v, u64 ref, u32 cmp, u32 bits) {
  switch (cmp) {
    case 0:
      return v == ref;
    case 1:
      return v > ref;
    case 2:
      return v < ref;
    case 3:
      return v != ref;
    case 4:
      return SignExtend(v - ref, bits) >= 0;
    case 5:
      return SignExtend(v, bits) > SignExtend(ref, bits);
    case 6:
      return SignExtend(v, bits) < SignExtend(ref, bits);
    default:
      return false;
  }
}

// Blocks the worker until `ready` holds or the session stops.
template <class Pred>
void WaitUntil(Pred ready, const char* what, u64 at) {
  base::UniqueLock<base::Mutex> lk(g_m);
  const auto start = base::TimeTicks::Now();
  bool reported = false;
  while (!ready() && !guest::Stopping()) {
    g_cv.WaitFor(lk, base::Microseconds(100));
    if (!reported && base::TimeTicks::Now() - start > base::Seconds(2)) {
      reported = true;
      BASE_LOGW("apr", "{} {:#x} still unmet after 2 s", what, at);
    }
  }
}

struct GatherScatter {
  bool valid = false;
  u32 file = 0;
  u64 dst = 0;  // where the next gather writes
  u64 off = 0;  // where the next scatter reads
};

int Read(GatherScatter& gs, u32 file, u64 dst, u64 len, u64 off) {
  if (const int err = ReadFile(file, dst, len, off))
    return err;
  gs = {true, file, dst + len, off + len};
  return 0;
}

// Runs one command buffer; on failure `*err_dw` is the failing command.
int Execute(const u32* cmd, u32 ndw, u32* err_dw) {
  GatherScatter gs;
  bool mapping = false;
  for (u32 pos = 0; pos < ndw;) {
    *err_dw = pos;
    const u32* c = cmd + pos;
    const u32 hdr = c[0];
    const u32 n = ((hdr >> 8) & 0xf) + 1;
    if (pos + n > ndw)
      return kErrOverrun;
    pos += n;
    const u8 op = static_cast<u8>(hdr);
    switch (op) {
      case kWaitOnAddress: {
        const u64 addr = c[1] | u64(hdr >> 16) << 32;
        const u64 ref = (n >= 3 ? c[2] : 0) | (n >= 4 ? u64(c[3]) << 32 : 0);
        const u32 cmp = (hdr >> 13) & 7;
        if (addr & 7)
          return kErrWaitAddress;
        const auto* p = reinterpret_cast<const volatile u64*>(addr);
        if (!host_memory::IsMemoryRangeMapped(const_cast<u64*>(p), 8))
          return kErrFaultWaitAddress;
        WaitUntil([&] { return Compare(*p, ref, cmp, 64); }, "waitOnAddress",
                  addr);
        break;
      }
      case kWaitOnCounter: {
        const u32 idx = hdr >> 24;
        const u32 cmp = (hdr >> 13) & 7;
        u64 ref = (hdr >> 16) & 0xff;
        u32 access = 1, mask_op = 0;
        u64 mask = ~0ull;
        if (n >= 2) {
          access = ((c[1] >> 24) & 7) ^ 1;
          mask_op = (c[1] >> 28) & 3;
          if (n == 2 && mask_op) {
            ref |= u64(c[1] & 0xff) << 8;
            mask = (c[1] >> 8) & 0xffff;
          } else {
            ref |= u64(c[1] & 0xffffff) << 8;
          }
          if (n == 3) {
            if (mask_op)
              mask = c[2];
            else
              ref |= u64(c[2]) << 32;
          }
          if (n >= 5) {
            ref |= u64(c[2]) << 32;
            mask = c[3] | u64(c[4]) << 32;
          }
        }
        if (access == 0 && (idx & 1))
          return kErrFormat;
        const u32 bits = AccessBits(access);
        WaitUntil(
            [&] {
              u64 v = ReadCounter(idx, access);
              if (mask_op)
                v &= mask;
              return Compare(v, ref, cmp, bits);
            },
            "waitOnCounter", idx);
        break;
      }
      case kWriteAddress:
      case kWriteAddress | kImmediate: {
        const u64 addr = (c[1] & ~7u) | u64(hdr >> 16) << 32;
        u64 value = (hdr >> 12) & 3;
        if (n >= 3)
          value |= u64(c[2]) << 2;
        if (n >= 4)
          value |= u64(c[3]) << 34;
        auto* p = reinterpret_cast<u64*>(addr);
        if (!host_memory::IsMemoryRangeMapped(p, 8))
          return kErrFaultWrite;
        switch (c[1] & 7) {
          case 0:
            break;
          case 1:
            value = TimeCounter();
            break;
          case 2:
          case 3: {
            base::LockGuard<base::Mutex> lk(g_m);
            value =
                ReadCounter(static_cast<u32>(value), (c[1] & 7) == 3 ? 0 : 1);
            break;
          }
          default:
            return kErrWriteSource;
        }
        host_memory::BeforeHostWrite(p, 8);
        __atomic_store_n(p, value, __ATOMIC_RELEASE);
        break;
      }
      case kWriteCounter:
      case kWriteCounter | kImmediate: {
        const u32 idx = hdr >> 24;
        u64 value = (hdr >> 12) & 0xfff;
        u32 access = 1, wop = 0;
        if (n >= 2) {
          value |= u64(c[1] & 0xfffff) << 12;
          access = ((c[1] >> 20) & 7) ^ 1;
          wop = (c[1] >> 24) & 7;
        }
        if (n >= 3)
          value |= u64(c[2]) << 32;
        if (access == 0 && (idx & 1))
          return kErrFormat;
        base::LockGuard<base::Mutex> lk(g_m);
        WriteCounter(idx, access, value, wop);
        g_cv.NotifyAll();
        break;
      }
      case kWriteEventQueue:
      case kWriteEventQueue | kImmediate: {
        if (n != 5)
          return kErrFormat;
        const u64 eq = c[1] | u64(hdr >> 16) << 32;
        auto* proc = Process::GetActive();
        auto obj = proc ? proc->GetObjTable().Get(static_cast<u32>(eq))
                        : ObjectRef<Object>();
        if (!obj || obj->type() != Object::Type::kEqueue)
          return kErrEventQueue;
        static_cast<Equeue*>(obj.get())->Trigger(
            c[2], kEvfiltAmpr, static_cast<i64>(c[3] | u64(c[4]) << 32));
        break;
      }
      case kNop:
        break;
      case kReadFile: {
        if (n < 5)
          return kErrFormat;
        const u64 dst = c[3] | u64(c[4] & 0xffff) << 32;
        const u64 off = ((hdr >> 12) & 0x3ffff) | (c[4] & 0xfffc0000) |
                        (n >= 6 ? u64(c[5] & 0xff) << 32 : 0);
        if (const int err = Read(gs, c[2], dst, u64(c[1]) + 1, off))
          return err;
        break;
      }
      case kReadGather: {
        if (!gs.valid)
          return kErrGatherScatter;
        const u64 off =
            ((hdr >> 12) & 0x3ffff) | (n >= 3 ? u64(c[2] & 0x3fffff) << 18 : 0);
        if (const int err = Read(gs, gs.file, gs.dst, u64(c[1]) + 1, off))
          return err;
        break;
      }
      case kReadScatter: {
        if (n != 3)
          return kErrFormat;
        if (!gs.valid)
          return kErrGatherScatter;
        const u64 dst = c[2] | u64((hdr >> 12) & 0xffff) << 32;
        if (const int err = Read(gs, gs.file, dst, u64(c[1]) + 1, gs.off))
          return err;
        break;
      }
      case kReadGatherScatter: {
        if (n < 4)
          return kErrFormat;
        if (!gs.valid)
          return kErrGatherScatter;
        const u64 dst = c[2] | u64(c[3] & 0xffff) << 32;
        const u64 off = ((hdr >> 12) & 0x3ffff) | (c[3] & 0xfffc0000) |
                        (n >= 5 ? u64(c[4] & 0xff) << 32 : 0);
        if (const int err = Read(gs, gs.file, dst, u64(c[1]) + 1, off))
          return err;
        break;
      }
      case kResetGatherScatter:
        gs.valid = false;
        break;
      case kMapBegin: {
        if (n != 3 && n != 4)
          return kErrFormat;
        if (mapping)
          return kErrMapState;
        const u64 va = u64(c[1]) << 14 | u64(c[2] & 3) << 46;
        const u64 size = u64(c[2] & 0x1ffffffc) << 12;
        const u32 prot = (hdr >> 12) & 0x7fff;
        void* p = reinterpret_cast<void*>(va);
        if (n == 4) {  // mapDirectBegin
          const i64 dmem = static_cast<i64>(u64(c[3] & 0x3ffffff) << 14);
          if (sys_mmap_dmem(p, size, static_cast<int>(prot), kFixed, 0, dmem) <
              0)
            return kErrVaRange;
        } else if (IsErrnoPtr(sys_mmap(p, size, prot, kAnon | kFixed,
                                       static_cast<u32>(-1), 0))) {
          return kErrVaRange;
        }
        mapping = true;
        break;
      }
      case kMapEnd:
        if (!mapping)
          return kErrMapState;
        mapping = false;
        break;
      default:
        return kErrOpcode;
    }
  }
  return mapping ? kErrMapState : 0;
}

void Worker(u32 prio) {
  base::UniqueLock<base::Mutex> lk(g_m);
  while (!guest::Stopping()) {
    if (g_queues[prio].empty()) {
      g_cv.WaitFor(lk, base::Milliseconds(50));
      continue;
    }
    const Request req = g_queues[prio].front();
    g_queues[prio].pop_front();
    lk.unlock();
    u32 err_dw = 0;
    const int result = host_memory::IsMemoryRangeMapped(req.cmd, req.ndw * 4u)
                           ? Execute(req.cmd, req.ndw, &err_dw)
                           : kErrCmdAddress;
    if (result)
      BASE_LOGW("apr", "command buffer {:p} failed {:#x} at +{:#x}",
                static_cast<const void*>(req.cmd), static_cast<u32>(result),
                err_dw * 4);
    else if (kAprTrace)
      BASE_LOGI("apr", "prio {} id {} ran {} dwords", prio, req.id, req.ndw);
    lk.lock();
    if (req.id) {
      auto it = g_completions.find(req.id);
      if (it != g_completions.end()) {
        it->second.done = true;
        it->second.result = result;
        it->second.error_offset = result ? err_dw * 4 : 0;
      }
    }
    g_cv.NotifyAll();
  }
  g_workers[prio] = false;
}

// Every APR call: only games (category below 0x1000) and category 0x10300 may
// use APR.
int CategoryGate() {
  const Process* proc = Process::GetActive();
  if (!proc)
    return -SysError::eOPNOTSUPP;
  const u32 category = proc->GetAppCategory();
  return category >= 0x1000 && category != 0x10300 ? -SysError::eOPNOTSUPP : 0;
}

bool ValidGuest(const void* p, size_t n) {
  return p && host_memory::IsMemoryRangeMapped(p, n);
}

// Writes one resolve outcome. `err` is an errno, 0 on success.
int Report(u32 i,
           int err,
           u32 id,
           i64 size,
           u32* ids,
           u64* sizes,
           int* results) {
  if (results) {
    if (!ValidGuest(results + i, 4))
      return SysError::eFAULT;
    results[i] =
        err ? static_cast<int>(0x80020000u | static_cast<u32>(err)) : 0;
  }
  if (!ValidGuest(ids + i, 4) || (sizes && !ValidGuest(sizes + i, 8)))
    return SysError::eFAULT;
  ids[i] = err ? 0xffffffffu : id;
  if (sizes)
    sizes[i] = err ? 0 : static_cast<u64>(size);
  return 0;
}
}  // namespace

int PS4ABI Ps5AprResolve(const char* prefix,
                         const char* const* paths,
                         u64 num,
                         u32* ids,
                         u64* sizes,
                         u32* error_index,
                         int* results,
                         u32* count) {
  if (const int err = CategoryGate())
    return err;
  if (num - 1 >= kResolveMax)
    return -SysError::eINVAL;
  base::String pre;
  if (prefix) {
    if (!ValidGuest(prefix, 1))
      return -SysError::eFAULT;
    const size_t len = ::strnlen(prefix, kPathMax);
    if (len >= kPathMax)
      return -SysError::eNAMETOOLONG;
    pre = base::String(prefix, len);
  }
  if (!ValidGuest(paths, num * sizeof(char*)))
    return -SysError::eFAULT;
  u32 ok = 0;
  for (u32 i = 0; i < num; i++) {
    const char* path = paths[i];
    int err = 0;
    base::String full;
    if (!ValidGuest(path, 1)) {
      err = SysError::eFAULT;
    } else {
      const size_t len = ::strnlen(path, kPathMax);
      if (len >= kPathMax || pre.size() + len >= kPathMax)
        err = SysError::eNAMETOOLONG;
      else
        full = pre + base::String(path, len);
    }
    u32 id = 0;
    i64 size = 0;
    if (!err)
      err = Resolve(full.c_str(), &id, &size);
    if (kAprTrace)
      BASE_LOGI("apr", "resolve {} -> id {} size {:#x} errno {}", full.c_str(),
                id, size, err);
    if (err && !results) {
      if (error_index && ValidGuest(error_index, 4))
        *error_index = i;
      return -err;
    }
    if (const int fault = Report(i, err, id, size, ids, sizes, results))
      return -fault;
    if (!err)
      ok++;
  }
  if (count) {
    if (!ValidGuest(count, 4))
      return -SysError::eFAULT;
    *count = ok;
  }
  return 0;
}

// apr_resolve takes eight arguments; the last two come off the guest stack.
void PS4ABI Ps5AprResolveContext(u32 /*sid*/, GuestFrame* f, u8* /*xsave*/) {
  f->rsp += 8;  // past the lifted call's return address
  const auto* stack = reinterpret_cast<const u64*>(f->rsp);
  u64 results = 0, count = 0;
  if (ValidGuest(stack + 1, 16)) {
    results = stack[1];
    count = stack[2];
  }
  const int r = Ps5AprResolve(
      reinterpret_cast<const char*>(f->rdi),
      reinterpret_cast<const char* const*>(f->rsi), f->rdx,
      reinterpret_cast<u32*>(f->rcx), reinterpret_cast<u64*>(f->r8),
      reinterpret_cast<u32*>(f->r9), reinterpret_cast<int*>(results),
      reinterpret_cast<u32*>(count));
  if (r < 0) {
    f->rax = static_cast<u64>(-r);
    f->rflags |= 1;
  } else {
    f->rax = 0;
    f->rflags &= ~1ull;
  }
}

// flag 0: fire and forget; 1: also hand the result to the wait; 2: id only.
int PS4ABI Ps5AprSubmit(const u32* cmd,
                        u64 ndw,
                        u32 prio,
                        u32 flag,
                        u32* result,
                        u32* id) {
  if (const int err = CategoryGate())
    return err;
  if (prio < 1 || prio > 6 || flag > 2 ||
      (reinterpret_cast<uintptr_t>(cmd) & 3))
    return -SysError::eINVAL;
  if (ndw * 4 > kBufferMax)
    return -SysError::eINVAL;
  base::UniqueLock<base::Mutex> lk(g_m);
  u32 req_id = 0;
  if (flag) {
    while (g_completions.size() >= kPendingMax) {
      if (SignalInterrupts())
        return -SysError::eINTR;
      g_cv.WaitFor(lk, base::Milliseconds(1));
    }
    req_id = g_next_id++;
    if (!g_next_id)
      g_next_id = 1;
    Completion c;
    c.result_ptr = flag == 1 ? result : nullptr;
    g_completions[req_id] = c;
  }
  g_queues[prio].push_back({cmd, static_cast<u32>(ndw), req_id});
  if (kAprTrace && host_memory::IsMemoryRangeMapped(cmd, ndw * 4)) {
    base::String ops;
    for (u32 pos = 0; pos < ndw && pos < 256;) {
      base::FormatTo(ops, " {:02x}", cmd[pos] & 0xff);
      pos += ((cmd[pos] >> 8) & 0xf) + 1;
    }
    BASE_LOGI("apr", "submit prio {} flag {} id {} ndw {}:{}", prio, flag,
              req_id, ndw, ops.c_str());
  }
  if (!g_workers[prio]) {
    g_workers[prio] = true;
    guest::SpawnThread("apr", [prio] { Worker(prio); });
  }
  g_cv.NotifyAll();
  lk.unlock();
  if (flag) {
    if (!ValidGuest(id, 4))
      return -SysError::eFAULT;
    *id = req_id;
  }
  return 0;
}

int PS4ABI Ps5AprStat(const u32* ids, u64 num, void* st) {
  if (const int err = CategoryGate())
    return err;
  if (num != 1)
    return -SysError::eINVAL;
  if (!ValidGuest(ids, 4))
    return -SysError::eFAULT;
  AprFile* f = *ids <= 0x50ffffff ? FileOf(*ids) : nullptr;
  if (!f)
    return -SysError::eNOENT;
  if (!ValidGuest(st, sizeof(SceKernelStat)))
    return -SysError::eFAULT;
  std::memset(st, 0, sizeof(SceKernelStat));
  FillStat(*static_cast<SceKernelStat*>(st), kSceFileModeReg, f->size);
  return 0;
}

int PS4ABI Ps5AprWait(const u32* ids, u64 num) {
  if (const int err = CategoryGate())
    return err;
  if (num != 1)
    return -SysError::eINVAL;
  if (!ValidGuest(ids, 4))
    return -SysError::eFAULT;
  const u32 id = *ids;
  const u32 self = *CurrentGuestTidPtr();
  base::UniqueLock<base::Mutex> lk(g_m);
  auto it = g_completions.find(id);
  if (it == g_completions.end())
    return -SysError::eSRCH;
  if (it->second.waiter && it->second.waiter != self)
    return -SysError::eBUSY;
  it->second.waiter = self;
  while (!it->second.done) {
    if (SignalInterrupts() || guest::Stopping()) {
      it->second.waiter = 0;
      return -SysError::eINTR;
    }
    g_cv.WaitFor(lk, base::Milliseconds(1));
    it = g_completions.find(id);
    if (it == g_completions.end())
      return -SysError::eSRCH;
  }
  const Completion c = it->second;
  g_completions.erase(it);
  if (kAprTrace)
    BASE_LOGI("apr", "wait id {} -> {:#x}", id, static_cast<u32>(c.result));
  g_cv.NotifyAll();
  lk.unlock();
  if (c.result_ptr) {
    if (!ValidGuest(c.result_ptr, c.result ? 8 : 4))
      return -SysError::eFAULT;
    c.result_ptr[0] = static_cast<u32>(c.result);
    if (c.result)
      c.result_ptr[1] = c.error_offset;
  }
  return 0;
}

}  // namespace kern
