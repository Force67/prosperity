/*
 * PS4Delta : PS4 emulation and research project
 *
 * The IPMI manager: client table + op dispatch. See ipmi.h for the ABI.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"

#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "host_memory/host_memory.h"

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/memory/move.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "kern/crash.h"
#include "kern/ipmi/ipmi.h"
#include "kern/ipmi/services.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kIpmiDump, "DELTA_IPMI_DUMP", 0);
DELTA_OPTION(u32, kIpmiOpDump, "DELTA_IPMI_OPDUMP", 0);
DELTA_OPTION(u32, kIpmiFailOp, "DELTA_IPMI_FAILOP", 0);
DELTA_OPTION(bool, kIpmiHist, "DELTA_IPMI_HIST", false);
DELTA_OPTION(bool, kIpmiTrace, "DELTA_IPMI_TRACE", false);
}  // namespace

namespace krnl::ipmi {
namespace {

// Manager command numbers (libSceIpmi op -> syscall 622).
enum {
  kCreateClient = 2,       // sceIpmiMgrCreateClient  -> result = client kid
  kDestroyClient = 3,      // sceIpmiMgrDestroyClient -> result = 0 (asserted)
  kStopSession = 784,      // StopSession/Disconnect
  kInvokeSync = 800,       // sceIpmiClientInvokeSyncMethod
  kPollAsyncReply = 1168,  // async-reply poll, paired with the invoke before it
  kConnect = 1024,         // sceIpmiClientConnect (carries the service name)
  kWaitEventFlag = 594,    // client event-flag wait (see the case below)
};

// A guest descriptor that says "n bytes here" is not to be trusted: clamp what
// we are willing to read or write through one.
constexpr u64 kMaxBuffer = 0x10000;

// The request block op=800 passes, plus its descriptor arrays.
struct InvokeRequest {
  u32 method_id;
  u32 num_in;
  u32 num_out;
  u32 pad;
  const u64* in_desc;   // {const void *data; uint64 size}  (16 bytes)
  const u64* out_desc;  // {void *data; uint64 cap; ...}    (24 bytes)
  i32* result;
};

// Strides confirmed by dumping multi-descriptor invokes on both platforms: a
// 5-output SceNpService call on PS5 fw 01.14.00 and a 2-output one on PS4 both
// decode cleanly at 24 bytes per output descriptor and nowhere else.
constexpr u32 kInDescWords = 2;
constexpr u32 kOutDescWords = 3;

bool Readable(const void* p, u64 n) {
  return p && n && n <= kMaxBuffer && host_memory::IsMemoryRangeMapped(p, n);
}

// ---------------------------------------------------------------- clients

struct Client {
  base::String service;
  Service* impl = nullptr;
};

base::Mutex g_clients_mtx;
base::HashMap<u32, Client> g_clients;
base::Atomic<u32> g_next_kid{1};

// Previous manager op per client. A repeated consecutive async-reply poll is a
// client waiting on something (libSceIpmi runs event-flag waits as a 1168 poll
// loop), not collecting an invoke's reply; see kPollAsyncReply.
base::HashMap<u32, u32> g_last_op;

bool IsRepeatPoll(u32 op, u32 kid) {
  base::LockGuard<base::Mutex> lk(g_clients_mtx);
  u32& last = g_last_op[kid];
  const bool repeat = op == kPollAsyncReply && last == kPollAsyncReply;
  last = op;
  return repeat;
}

Service* FindService(const char* name) {
  if (!name)
    return nullptr;
  Service* all[] = {&PlayGoService(), &NpManagerService(),
                    &NpWebService(),  &UserService(),
                    &LncService(),    &ArbitratorIpcService()};
  for (Service* s : all)
    if (std::strcmp(s->Name(), name) == 0)
      return s;
  return nullptr;
}

Client LookupClient(u32 kid) {
  base::LockGuard<base::Mutex> lk(g_clients_mtx);
  auto it = g_clients.find(kid);
  return it == g_clients.end() ? Client{} : it->second;
}

// The create/connect payload carries a pointer to the service name somewhere in
// its fields; find it rather than assuming a fixed offset (the block's shape
// differs between the two ops and across SDK versions).
const char* PayloadServiceName(const void* in, u64 insize) {
  auto* q = static_cast<const u64*>(in);
  for (u64 i = 0; i < insize / 8; i++) {
    u64 v = q[i];
    if (v <= 0x10000)  // nulls and small inline ints are never pointers
      continue;
    auto* s = reinterpret_cast<const char*>(v);
    if (!host_memory::IsMemoryRangeMapped(s, 4) ||
        std::strncmp(s, "Sce", 3) != 0)
      continue;
    int n = 3;
    bool printable = true;
    for (; n < 64 && s[n]; n++)
      if (s[n] < 0x20 || s[n] > 0x7e) {
        printable = false;
        break;
      }
    if (printable && n < 64)
      return s;
  }
  return nullptr;
}

// ---------------------------------------------------------------- tracing

base::Atomic<u64> g_op_hist[2048];
base::Atomic<u64> g_method_hist[64];
base::Atomic<u32> g_method_id[64];

bool TraceOn() {
  return kIpmiTrace;
}

// DELTA_IPMI_HIST: per-op and per-method call counts, dumped every 15s.
// Deliberately lock-free: a mutex-guarded version throttled the spin it was
// meant to measure, so the numbers lied.
void Histogram(u32 op, const InvokeRequest* req) {
  if (!kIpmiHist)
    return;
  if (op < 2048)
    g_op_hist[op].fetch_add(1, base::memory_order_relaxed);
  if (req) {
    const u32 m = req->method_id;
    for (u32 i = 0; i < 64; i++) {
      u32 want = g_method_id[i].load(base::memory_order_relaxed);
      if (want == m) {
        g_method_hist[i].fetch_add(1, base::memory_order_relaxed);
        break;
      }
      if (!want) {
        u32 expect = 0;
        if (g_method_id[i].compare_exchange_strong(expect, m))
          g_method_hist[i].fetch_add(1, base::memory_order_relaxed);
        break;
      }
    }
  }
  static const bool kStarted = [] {
    base::SpawnDetachedThread("ipmi", [] {
      for (;;) {
        base::SleepForMilliseconds((15) * 1000);
        for (u32 i = 0; i < 2048; i++)
          if (u64 c = g_op_hist[i].load(base::memory_order_relaxed))
            BASE_LOGI("ipmihist", "op={} {}", i, (unsigned long long)c);
        for (u32 i = 0; i < 64; i++)
          if (u64 c = g_method_hist[i].load(base::memory_order_relaxed))
            BASE_LOGI("ipmihist", "method={:#x} {}",
                      g_method_id[i].load(base::memory_order_relaxed),
                      (unsigned long long)c);
      }
    });
    return true;
  }();
  (void)kStarted;
}

void TraceInvoke(u32 kid,
                 const char* svc,
                 const InvokeRequest* req,
                 bool handled) {
  if (!TraceOn())
    return;
  BASE_LOGI("ipmi", "{} kid={} method={:#x} in={} out={}{}",
            svc && *svc ? svc : "?", kid, req->method_id, req->num_in,
            req->num_out, handled ? "" : " (default)");
}

// DELTA_IPMI_DUMP=<method>: dump one unknown method's descriptors and the guest
// call chain that made it, so its shape can be read instead of guessed. The
// request block sits on the guest stack, so scanning up from it finds the
// callers. Bounded to a handful of hits; this is a research knob.
void DumpInvoke(u32 kid, const char* svc, const InvokeRequest* req) {
  if (!kIpmiDump || req->method_id != kIpmiDump)
    return;
  static base::Atomic<int> seen{0};
  if (seen.fetch_add(1) >= 2)
    return;

  BASE_LOGI("ipmidump", "{} kid={} method={:#x} in={} out={}",
            svc && *svc ? svc : "?", kid, req->method_id, req->num_in,
            req->num_out);
  auto descriptors = [](const char* tag, const u64* d, u32 n, u32 stride) {
    if (!host_memory::IsMemoryRangeMapped(d, n * stride * 8))
      return;
    for (u32 i = 0; i < n; i++) {
      base::String line;
      base::FormatTo(line, "  {}[{}] data={:#x} size={:#x}", tag, i,
                     (unsigned long long)d[i * stride],
                     (unsigned long long)d[i * stride + 1]);
      if (stride > 2)  // the unidentified third word; see OutSlot()
        base::FormatTo(line, " w2={:#x}",
                       (unsigned long long)d[i * stride + 2]);
      auto* p = reinterpret_cast<const u8*>(d[i * stride]);
      const u64 sz = d[i * stride + 1];
      if (Readable(p, sz) && sz <= 64) {
        base::FormatTo(line, " :");
        for (u64 k = 0; k < sz; k++)
          base::FormatTo(line, " {:02x}", p[k]);
      }
      BASE_LOGI("ipmidump", "{}", line.c_str());
    }
  };
  descriptors("in", req->in_desc, req->num_in, kInDescWords);
  descriptors("out", req->out_desc, req->num_out, kOutDescWords);

  auto* sp = reinterpret_cast<const uintptr_t*>(req);
  int shown = 0;
  for (int i = 0; i < 2048 && shown < 24; i++) {
    char sym[256];
    Symbolize(sp[i], sym, sizeof(sym));
    if (std::strstr(sym, "(.text)")) {
      BASE_LOGI("ipmidump", "  caller {}", sym);
      shown++;
    }
  }
}

// DELTA_IPMI_OPDUMP=<op>: same idea as dumpInvoke, but for a MANAGER op the
// decoder does not know. An unhandled op that a module spins on is answered
// "empty success", which is a guess; dumping its request block and the guest
// call chain shows what the caller actually reads back, so the op can be
// implemented instead of guessed at.
void DumpManagerOp(u32 op, u32 kid, void* out, void* in, u64 insize) {
  if (!kIpmiOpDump || op != kIpmiOpDump)
    return;
  static base::Atomic<int> seen{0};
  if (seen.fetch_add(1) >= 3)
    return;

  BASE_LOGI("ipmiop", "op={} kid={} out={:p} in={:p} insize={:#x}", op, kid,
            out, in, (unsigned long long)insize);
  auto hexdump = [](const char* tag, const void* p, u64 n) {
    if (!p || !Readable(p, n))
      return;
    const auto* b = static_cast<const u8*>(p);
    base::String bytes;
    base::FormatTo(bytes, "  {}:", tag);
    for (u64 i = 0; i < n && i < 96; i++)
      base::FormatTo(bytes, " {:02x}", b[i]);
    BASE_LOGI("ipmiop", "{}", bytes.c_str());
    // Any 8-byte field that looks like a guest pointer is worth following:
    // these blocks are mostly pointers to status words the caller polls.
    for (u64 i = 0; i + 8 <= n && i < 96; i += 8) {
      u64 v = 0;
      std::memcpy(&v, b + i, sizeof(v));
      const auto* t = reinterpret_cast<const u8*>(v);
      if (v >= 0x1000 && Readable(t, 8)) {
        base::String line;
        base::FormatTo(line, "    +{:#x} -> {:#x} :", (unsigned long long)i,
                       (unsigned long long)v);
        for (int k = 0; k < 8; k++)
          base::FormatTo(line, " {:02x}", t[k]);
        BASE_LOGI("ipmiop", "{}", line.c_str());
      }
    }
  };
  hexdump("in", in, insize ? insize : 64);
  hexdump("out", out, 16);

  auto* sp = reinterpret_cast<const uintptr_t*>(in ? in : out);
  int shown = 0;
  for (int i = 0; sp && i < 768 && shown < 6; i++) {
    char sym[256];
    Symbolize(sp[i], sym, sizeof(sym));
    if (std::strstr(sym, "(.text)")) {
      BASE_LOGI("ipmiop", "  caller {}", sym);
      shown++;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------- Invocation

Invocation::Invocation(void* request) : req_(request) {}

u32 Invocation::Method() const {
  return static_cast<InvokeRequest*>(req_)->method_id;
}
u32 Invocation::NumIn() const {
  return static_cast<InvokeRequest*>(req_)->num_in;
}
u32 Invocation::NumOut() const {
  return static_cast<InvokeRequest*>(req_)->num_out;
}

const void* Invocation::Input(u32 i, u64& size) const {
  size = 0;
  auto* r = static_cast<InvokeRequest*>(req_);
  if (i >= r->num_in || !r->in_desc)
    return nullptr;
  const u64* d = r->in_desc + i * kInDescWords;
  auto* p = reinterpret_cast<const void*>(d[0]);
  if (!Readable(p, d[1]))
    return nullptr;
  size = d[1];
  return p;
}

// Only the data pointer and capacity are acted on. The third word is not a
// "bytes transferred" slot: dumps caught it holding a literal 1 and, on the
// next descriptor of the same call, a stale return address. Writing through it
// made libSceSystemService fault on a clobbered object pointer, so leave it
// alone.
bool Invocation::OutSlot(u32 i, u8*& data, u64& cap) const {
  data = nullptr;
  cap = 0;
  auto* r = static_cast<InvokeRequest*>(req_);
  if (i >= r->num_out || !r->out_desc)
    return false;
  const u64* d = r->out_desc + i * kOutDescWords;
  auto* p = reinterpret_cast<u8*>(d[0]);
  if (!Readable(p, d[1]))
    return false;
  data = p;
  cap = d[1];
  return true;
}

bool Invocation::Reply(u32 i, const void* data, u64 size) {
  u8* p;
  u64 cap;
  if (!OutSlot(i, p, cap))
    return false;
  const u64 n = size < cap ? size : cap;
  std::memcpy(p, data, n);
  if (n < cap)
    std::memset(p + n, 0, cap - n);
  return true;
}

bool Invocation::ReplyU32(u32 i, u32 v) {
  return Reply(i, &v, sizeof(v));
}

bool Invocation::ReplyFill(u32 i, u8 byte) {
  u8* p;
  u64 cap;
  if (!OutSlot(i, p, cap))
    return false;
  std::memset(p, byte, cap);
  return true;
}

void Invocation::ReplyEmpty() {
  auto* r = static_cast<InvokeRequest*>(req_);
  for (u32 i = 0; i < r->num_out; i++)
    ReplyFill(i, 0);
}

void Invocation::SetResult(i32 v) {
  auto* r = static_cast<InvokeRequest*>(req_);
  if (host_memory::IsMemoryRangeMapped(r->result, sizeof(*r->result)))
    *r->result = v;
}

// ---------------------------------------------------------------- manager

int ManagerCall(u32 op, u32 kid, void* out, void* in, u64 insize) {
  auto* req = op == kInvokeSync && in && insize >= sizeof(InvokeRequest)
                  ? static_cast<InvokeRequest*>(in)
                  : nullptr;
  Histogram(op, req);
  const bool repeat_poll = IsRepeatPoll(op, kid);

  auto set_result = [&](u32 v) {
    if (out)
      *static_cast<u32*>(out) = v;
  };

  switch (op) {
    case kCreateClient: {
      const u32 new_kid = g_next_kid.fetch_add(1);
      const char* svc = in ? PayloadServiceName(in, insize) : nullptr;
      Client c;
      c.service = svc ? svc : "";
      c.impl = FindService(svc);
      if (TraceOn())
        BASE_LOGI("ipmi", "create kid={} service=\"{}\"{}", new_kid,
                  c.service.c_str(), c.impl ? "" : " (no handler)");
      {
        base::LockGuard<base::Mutex> lk(g_clients_mtx);
        g_clients[new_kid] = base::move(c);
      }
      set_result(new_kid);
      return 0;
    }

    case kDestroyClient: {
      base::LockGuard<base::Mutex> lk(g_clients_mtx);
      g_clients.erase(kid);
      set_result(0);
      return 0;
    }

    case kConnect: {
      // The connect payload names the service too; a client created without a
      // recognisable name still gets bound here.
      const char* svc = in ? PayloadServiceName(in, insize) : nullptr;
      if (svc) {
        base::LockGuard<base::Mutex> lk(g_clients_mtx);
        Client& c = g_clients[kid];
        if (c.service.empty()) {
          c.service = svc;
          c.impl = FindService(svc);
        }
      }
      set_result(0);
      return 0;
    }

    case kInvokeSync: {
      if (req) {
        Client c = LookupClient(kid);
        Invocation inv(req);
        inv.SetResult(0);  // SCE_OK unless the service says otherwise
        TraceInvoke(kid, c.service.c_str(), req, c.impl != nullptr);
        DumpInvoke(kid, c.service.c_str(), req);
        if (c.impl)
          c.impl->Invoke(inv);
        else
          inv.ReplyEmpty();
      }
      set_result(0);
      return 0;
    }

    case kPollAsyncReply: {
      // Paired 1:1 with the invoke before it. With no daemon the request block
      // keeps its pre-call sentinel (0xFFFFFFFF at +8) and the client spins on
      // it, measured at ~30M calls/s. Report the same empty, successful reply
      // the synchronous path gives. A REPEAT poll (no invoke in between) is a
      // wait loop that nothing we do will satisfy; park it to a poll cadence
      // (the resource-arbitrator worker measured ~150k polls/s otherwise).
      DumpManagerOp(op, kid, out, in, insize);
      if (repeat_poll)
        base::SleepForMilliseconds(2);
      if (in && insize >= 40) {
        auto* b = static_cast<u8*>(in);
        u32 status = 0;
        std::memcpy(&status, b + 8, sizeof(status));
        if (status == 0xFFFFFFFFu) {
          const u32 done = 0;
          std::memcpy(b + 8, &done, sizeof(done));
        }
        u32* words[2] = {};
        std::memcpy(&words[0], b + 24, sizeof(words[0]));
        std::memcpy(&words[1], b + 32, sizeof(words[1]));
        for (u32* w : words)
          if (host_memory::IsMemoryRangeMapped(w, sizeof(*w)))
            *w = 0;
      }
      set_result(0);
      return 0;
    }

    case kStopSession:
      // The payload's first field points at a guest status word that libSceIpmi
      // asserts is zero once the manager syscall returns.
      if (in && insize >= sizeof(u64)) {
        u32* status = nullptr;
        std::memcpy(&status, in, sizeof(status));
        if (host_memory::IsMemoryRangeMapped(status, sizeof(*status)))
          *status = 0;
      }
      set_result(0);
      return 0;

    // Client event-flag wait. libSceUserService creates
    // "SceUserServiceClientEventFlag" and waits on it from
    // sceUserServiceGetEvent (callers libSceUserService+0x1c45 -> +0x1ded) for
    // the daemon to publish a login/logout event. Request block insize 0x20:
    // +0x00 flag index, +0x04 pattern, +0x08 and +0x10 out pointers, +0x18
    // size. One local user, permanently signed in, so no event is ever pending
    // and the wait must report UNSATISFIED. Answering generic "empty success"
    // instead makes sceUserServiceGetEvent allocate an event record out of
    // libkernel's 16 MiB SceKernelInternalMemory arena, find nothing in it, and
    // retry, leaking per iteration until libkernel throws std::bad_alloc and
    // std::terminate lands on a UD2 in libSceLibcInternal. Failing the wait is
    // what makes GetEvent return "no event" and let the caller proceed.
    case kWaitEventFlag:
      DumpManagerOp(op, kid, out, in, insize);
      // Leaves the wrapper's pre-set -1 result in place, which is what
      // libSceIpmi reads back as the call's value.
      return -1;

    // DELTA_IPMI_FAILOP=<op>: answer one op as a hard failure instead of "empty
    // success". Kept as a research knob: it is what separated a caller that
    // retries on failure from one that spins because we claimed success with no
    // data, and the next unknown op will need the same distinction drawn.
    default:
      if (TraceOn())
        BASE_LOGI("ipmi", "op={} kid={} (unhandled, empty success)", op, kid);
      DumpManagerOp(op, kid, out, in, insize);
      {
        if (kIpmiFailOp && op == kIpmiFailOp)
          return -1;  // leaves the wrapper's pre-set -1 result in place
      }
      set_result(0);
      return 0;
  }
}

}  // namespace krnl::ipmi
