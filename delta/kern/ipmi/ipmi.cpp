/*
 * PS4Delta : PS4 emulation and research project
 *
 * The IPMI manager: client table + op dispatch. See ipmi.h for the ABI.
 */

#include <atomic>
#include "base/arch.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/mem.h>

#include "ipmi.h"
#include "kern/crash.h"
#include "services.h"
#include <utl/options.h>

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
  kCreateClient = 2,     // sceIpmiMgrCreateClient  -> result = client kid
  kDestroyClient = 3,    // sceIpmiMgrDestroyClient -> result = 0 (asserted)
  kStopSession = 784,    // StopSession/Disconnect
  kInvokeSync = 800,     // sceIpmiClientInvokeSyncMethod
  kPollAsyncReply = 1168,// async-reply poll, paired with the invoke before it
  kConnect = 1024,       // sceIpmiClientConnect (carries the service name)
  kWaitEventFlag = 594,  // client event-flag wait (see the case below)
};

// A guest descriptor that says "n bytes here" is not to be trusted: clamp what
// we are willing to read or write through one.
constexpr u64 kMaxBuffer = 0x10000;

// The request block op=800 passes, plus its descriptor arrays.
struct InvokeRequest {
  u32 methodId;
  u32 numIn;
  u32 numOut;
  u32 pad;
  const u64 *inDesc;  // {const void *data; uint64 size}  (16 bytes)
  const u64 *outDesc; // {void *data; uint64 cap; ...}    (24 bytes)
  i32 *result;
};

// Strides confirmed by dumping multi-descriptor invokes on both platforms: a
// 5-output SceNpService call on PS5 fw 01.14.00 and a 2-output one on PS4 both
// decode cleanly at 24 bytes per output descriptor and nowhere else.
constexpr u32 kInDescWords = 2;
constexpr u32 kOutDescWords = 3;

bool readable(const void *p, u64 n) {
  return p && n && n <= kMaxBuffer && utl::isMemoryRangeMapped(p, n);
}

// ---------------------------------------------------------------- clients

struct Client {
  std::string service;
  Service *impl = nullptr;
};

std::mutex g_clientsMtx;
std::unordered_map<u32, Client> g_clients;
std::atomic<u32> g_nextKid{1};

// Previous manager op per client. A repeated consecutive async-reply poll is a
// client waiting on something (libSceIpmi runs event-flag waits as a 1168 poll
// loop), not collecting an invoke's reply; see kPollAsyncReply.
std::unordered_map<u32, u32> g_lastOp;

bool isRepeatPoll(u32 op, u32 kid) {
  std::lock_guard<std::mutex> lk(g_clientsMtx);
  u32 &last = g_lastOp[kid];
  const bool repeat = op == kPollAsyncReply && last == kPollAsyncReply;
  last = op;
  return repeat;
}

Service *findService(const char *name) {
  if (!name)
    return nullptr;
  Service *all[] = {&playGoService(), &npManagerService(), &npWebService(),
                    &userService(), &lncService(), &arbitratorIpcService()};
  for (Service *s : all)
    if (std::strcmp(s->name(), name) == 0)
      return s;
  return nullptr;
}

Client lookupClient(u32 kid) {
  std::lock_guard<std::mutex> lk(g_clientsMtx);
  auto it = g_clients.find(kid);
  return it == g_clients.end() ? Client{} : it->second;
}

// The create/connect payload carries a pointer to the service name somewhere in
// its fields; find it rather than assuming a fixed offset (the block's shape
// differs between the two ops and across SDK versions).
const char *payloadServiceName(const void *in, u64 insize) {
  auto *q = static_cast<const u64 *>(in);
  for (u64 i = 0; i < insize / 8; i++) {
    u64 v = q[i];
    if (v <= 0x10000) // nulls and small inline ints are never pointers
      continue;
    auto *s = reinterpret_cast<const char *>(v);
    if (!utl::isMemoryRangeMapped(s, 4) || std::strncmp(s, "Sce", 3) != 0)
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

std::atomic<u64> g_opHist[2048];
std::atomic<u64> g_methodHist[64];
std::atomic<u32> g_methodId[64];

bool traceOn() {
  return kIpmiTrace;
}

// DELTA_IPMI_HIST: per-op and per-method call counts, dumped every 15s.
// Deliberately lock-free: a mutex-guarded version throttled the very spin it was
// meant to measure, so the numbers lied.
void histogram(u32 op, const InvokeRequest *req) {
  if (!kIpmiHist)
    return;
  if (op < 2048)
    g_opHist[op].fetch_add(1, std::memory_order_relaxed);
  if (req) {
    const u32 m = req->methodId;
    for (u32 i = 0; i < 64; i++) {
      u32 want = g_methodId[i].load(std::memory_order_relaxed);
      if (want == m) {
        g_methodHist[i].fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (!want) {
        u32 expect = 0;
        if (g_methodId[i].compare_exchange_strong(expect, m))
          g_methodHist[i].fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
  }
  static const bool started = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        for (u32 i = 0; i < 2048; i++)
          if (u64 c = g_opHist[i].load(std::memory_order_relaxed))
            BASE_LOGI("ipmihist", "op={} {}", i, (unsigned long long)c);
        for (u32 i = 0; i < 64; i++)
          if (u64 c = g_methodHist[i].load(std::memory_order_relaxed))
            BASE_LOGI("ipmihist", "method={:#x} {}",
                      g_methodId[i].load(std::memory_order_relaxed),
                      (unsigned long long)c);
      }
    }).detach();
    return true;
  }();
  (void)started;
}

void traceInvoke(u32 kid, const char *svc, const InvokeRequest *req,
                 bool handled) {
  if (!traceOn())
    return;
  BASE_LOGI("ipmi", "{} kid={} method={:#x} in={} out={}{}",
            svc && *svc ? svc : "?", kid, req->methodId, req->numIn,
            req->numOut, handled ? "" : " (default)");
}

// DELTA_IPMI_DUMP=<method>: dump one unknown method's descriptors and the guest
// call chain that made it, so its shape can be read instead of guessed. The
// request block sits on the guest stack, so scanning up from it finds the
// callers. Bounded to a handful of hits; this is a research knob.
void dumpInvoke(u32 kid, const char *svc, const InvokeRequest *req) {
  if (!kIpmiDump || req->methodId != kIpmiDump)
    return;
  static std::atomic<int> seen{0};
  if (seen.fetch_add(1) >= 2)
    return;

  BASE_LOGI("ipmidump", "{} kid={} method={:#x} in={} out={}",
            svc && *svc ? svc : "?", kid, req->methodId, req->numIn,
            req->numOut);
  auto descriptors = [](const char *tag, const u64 *d, u32 n,
                        u32 stride) {
    if (!utl::isMemoryRangeMapped(d, n * stride * 8))
      return;
    for (u32 i = 0; i < n; i++) {
      base::String line;
      base::FormatTo(line, "  {}[{}] data={:#x} size={:#x}", tag, i,
                     (unsigned long long)d[i * stride],
                     (unsigned long long)d[i * stride + 1]);
      if (stride > 2) // the unidentified third word; see outSlot()
        base::FormatTo(line, " w2={:#x}",
                       (unsigned long long)d[i * stride + 2]);
      auto *p = reinterpret_cast<const u8 *>(d[i * stride]);
      const u64 sz = d[i * stride + 1];
      if (readable(p, sz) && sz <= 64) {
        base::FormatTo(line, " :");
        for (u64 k = 0; k < sz; k++)
          base::FormatTo(line, " {:02x}", p[k]);
      }
      BASE_LOGI("ipmidump", "{}", line.c_str());
    }
  };
  descriptors("in", req->inDesc, req->numIn, kInDescWords);
  descriptors("out", req->outDesc, req->numOut, kOutDescWords);

  auto *sp = reinterpret_cast<const uintptr_t *>(req);
  int shown = 0;
  for (int i = 0; i < 2048 && shown < 24; i++) {
    char sym[256];
    symbolize(sp[i], sym, sizeof(sym));
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
void dumpManagerOp(u32 op, u32 kid, void *out, void *in,
                   u64 insize) {
  if (!kIpmiOpDump || op != kIpmiOpDump)
    return;
  static std::atomic<int> seen{0};
  if (seen.fetch_add(1) >= 3)
    return;

  BASE_LOGI("ipmiop", "op={} kid={} out={:p} in={:p} insize={:#x}", op, kid,
            out, in, (unsigned long long)insize);
  auto hexdump = [](const char *tag, const void *p, u64 n) {
    if (!p || !readable(p, n))
      return;
    const auto *b = static_cast<const u8 *>(p);
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
      const auto *t = reinterpret_cast<const u8 *>(v);
      if (v >= 0x1000 && readable(t, 8)) {
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

  auto *sp = reinterpret_cast<const uintptr_t *>(in ? in : out);
  int shown = 0;
  for (int i = 0; sp && i < 768 && shown < 6; i++) {
    char sym[256];
    symbolize(sp[i], sym, sizeof(sym));
    if (std::strstr(sym, "(.text)")) {
      BASE_LOGI("ipmiop", "  caller {}", sym);
      shown++;
    }
  }
}

} // namespace

// ---------------------------------------------------------------- Invocation

Invocation::Invocation(void *request) : req_(request) {}

u32 Invocation::method() const {
  return static_cast<InvokeRequest *>(req_)->methodId;
}
u32 Invocation::numIn() const {
  return static_cast<InvokeRequest *>(req_)->numIn;
}
u32 Invocation::numOut() const {
  return static_cast<InvokeRequest *>(req_)->numOut;
}

const void *Invocation::input(u32 i, u64 &size) const {
  size = 0;
  auto *r = static_cast<InvokeRequest *>(req_);
  if (i >= r->numIn || !r->inDesc)
    return nullptr;
  const u64 *d = r->inDesc + i * kInDescWords;
  auto *p = reinterpret_cast<const void *>(d[0]);
  if (!readable(p, d[1]))
    return nullptr;
  size = d[1];
  return p;
}

// Only the data pointer and capacity are acted on. The third word is not a
// "bytes transferred" slot: dumps caught it holding a literal 1 and, on the next
// descriptor of the same call, a stale return address. Writing through it made
// libSceSystemService fault on a clobbered object pointer, so leave it alone.
bool Invocation::outSlot(u32 i, u8 *&data, u64 &cap) const {
  data = nullptr;
  cap = 0;
  auto *r = static_cast<InvokeRequest *>(req_);
  if (i >= r->numOut || !r->outDesc)
    return false;
  const u64 *d = r->outDesc + i * kOutDescWords;
  auto *p = reinterpret_cast<u8 *>(d[0]);
  if (!readable(p, d[1]))
    return false;
  data = p;
  cap = d[1];
  return true;
}

bool Invocation::reply(u32 i, const void *data, u64 size) {
  u8 *p;
  u64 cap;
  if (!outSlot(i, p, cap))
    return false;
  const u64 n = size < cap ? size : cap;
  std::memcpy(p, data, n);
  if (n < cap)
    std::memset(p + n, 0, cap - n);
  return true;
}

bool Invocation::replyU32(u32 i, u32 v) {
  return reply(i, &v, sizeof(v));
}

bool Invocation::replyFill(u32 i, u8 byte) {
  u8 *p;
  u64 cap;
  if (!outSlot(i, p, cap))
    return false;
  std::memset(p, byte, cap);
  return true;
}

void Invocation::replyEmpty() {
  auto *r = static_cast<InvokeRequest *>(req_);
  for (u32 i = 0; i < r->numOut; i++)
    replyFill(i, 0);
}

void Invocation::setResult(i32 v) {
  auto *r = static_cast<InvokeRequest *>(req_);
  if (utl::isMemoryRangeMapped(r->result, sizeof(*r->result)))
    *r->result = v;
}

// ---------------------------------------------------------------- manager

int managerCall(u32 op, u32 kid, void *out, void *in,
                u64 insize) {
  auto *req = op == kInvokeSync && in && insize >= sizeof(InvokeRequest)
                  ? static_cast<InvokeRequest *>(in)
                  : nullptr;
  histogram(op, req);
  const bool repeatPoll = isRepeatPoll(op, kid);

  auto setResult = [&](u32 v) {
    if (out)
      *static_cast<u32 *>(out) = v;
  };

  switch (op) {
  case kCreateClient: {
    const u32 newKid = g_nextKid.fetch_add(1);
    const char *svc = in ? payloadServiceName(in, insize) : nullptr;
    Client c;
    c.service = svc ? svc : "";
    c.impl = findService(svc);
    if (traceOn())
      BASE_LOGI("ipmi", "create kid={} service=\"{}\"{}", newKid,
                c.service.c_str(), c.impl ? "" : " (no handler)");
    {
      std::lock_guard<std::mutex> lk(g_clientsMtx);
      g_clients[newKid] = std::move(c);
    }
    setResult(newKid);
    return 0;
  }

  case kDestroyClient: {
    std::lock_guard<std::mutex> lk(g_clientsMtx);
    g_clients.erase(kid);
    setResult(0);
    return 0;
  }

  case kConnect: {
    // The connect payload names the service too; a client created without a
    // recognisable name still gets bound here.
    const char *svc = in ? payloadServiceName(in, insize) : nullptr;
    if (svc) {
      std::lock_guard<std::mutex> lk(g_clientsMtx);
      Client &c = g_clients[kid];
      if (c.service.empty()) {
        c.service = svc;
        c.impl = findService(svc);
      }
    }
    setResult(0);
    return 0;
  }

  case kInvokeSync: {
    if (req) {
      Client c = lookupClient(kid);
      Invocation inv(req);
      inv.setResult(0); // SCE_OK unless the service says otherwise
      traceInvoke(kid, c.service.c_str(), req, c.impl != nullptr);
      dumpInvoke(kid, c.service.c_str(), req);
      if (c.impl)
        c.impl->invoke(inv);
      else
        inv.replyEmpty();
    }
    setResult(0);
    return 0;
  }

  case kPollAsyncReply: {
    // Paired 1:1 with the invoke before it. With no daemon the request block
    // keeps its pre-call sentinel (0xFFFFFFFF at +8) and the client spins on it,
    // measured at ~30M calls/s. Report the same empty, successful reply the
    // synchronous path gives. A REPEAT poll (no invoke in between) is a
    // wait loop that nothing we do will satisfy; park it to a poll cadence
    // (the resource-arbitrator worker measured ~150k polls/s otherwise).
    dumpManagerOp(op, kid, out, in, insize);
    if (repeatPoll)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (in && insize >= 40) {
      auto *b = static_cast<u8 *>(in);
      u32 status = 0;
      std::memcpy(&status, b + 8, sizeof(status));
      if (status == 0xFFFFFFFFu) {
        const u32 done = 0;
        std::memcpy(b + 8, &done, sizeof(done));
      }
      u32 *words[2] = {};
      std::memcpy(&words[0], b + 24, sizeof(words[0]));
      std::memcpy(&words[1], b + 32, sizeof(words[1]));
      for (u32 *w : words)
        if (utl::isMemoryRangeMapped(w, sizeof(*w)))
          *w = 0;
    }
    setResult(0);
    return 0;
  }

  case kStopSession:
    // The payload's first field points at a guest status word that libSceIpmi
    // asserts is zero once the manager syscall returns.
    if (in && insize >= sizeof(u64)) {
      u32 *status = nullptr;
      std::memcpy(&status, in, sizeof(status));
      if (utl::isMemoryRangeMapped(status, sizeof(*status)))
        *status = 0;
    }
    setResult(0);
    return 0;

  // Client event-flag wait. libSceUserService creates
  // "SceUserServiceClientEventFlag" and waits on it from sceUserServiceGetEvent
  // (callers libSceUserService+0x1c45 -> +0x1ded) for the daemon to publish a
  // login/logout event. Request block insize 0x20: +0x00 flag index,
  // +0x04 pattern, +0x08 and +0x10 out pointers, +0x18 size.
  //
  // There is one local user and it is permanently signed in, so no event is ever
  // pending and the wait must report UNSATISFIED. That is not the same as the
  // generic "empty success": answering success says an event arrived, so
  // sceUserServiceGetEvent allocates an event record out of libkernel's 16 MiB
  // SceKernelInternalMemory arena, finds nothing in it, and retries -- leaking
  // per iteration until libkernel prints "Internal Memory is running out",
  // throws std::bad_alloc, and std::terminate lands on a UD2 in
  // libSceLibcInternal. Failing the wait is what makes GetEvent return
  // "no event" and let the caller proceed.
  case kWaitEventFlag:
    dumpManagerOp(op, kid, out, in, insize);
    // Leaves the wrapper's pre-set -1 result in place, which is what libSceIpmi
    // reads back as the call's value.
    return -1;

  // DELTA_IPMI_FAILOP=<op>: answer one op as a hard failure instead of "empty
  // success". Kept as a research knob -- it is what separated a caller that
  // retries on failure from one that spins because we claimed success with no
  // data, and the next unknown op will need the same distinction drawn.
  default:
    if (traceOn())
      BASE_LOGI("ipmi", "op={} kid={} (unhandled, empty success)", op, kid);
    dumpManagerOp(op, kid, out, in, insize);
    {
      if (kIpmiFailOp && op == kIpmiFailOp)
        return -1;  // leaves the wrapper's pre-set -1 result in place
    }
    setResult(0);
    return 0;
  }
}

} // namespace krnl::ipmi
