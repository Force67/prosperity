/*
 * PS4Delta : PS4 emulation and research project
 *
 * The IPMI manager: client table + op dispatch. See ipmi.h for the ABI.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <base.h>
#include <utl/mem.h>

#include "ipmi.h"
#include "kern/crash.h"
#include "services.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(uint32_t, kIpmiDump, "DELTA_IPMI_DUMP", 0);
DELTA_OPTION(uint32_t, kIpmiOpDump, "DELTA_IPMI_OPDUMP", 0);
DELTA_OPTION(uint32_t, kIpmiFailOp, "DELTA_IPMI_FAILOP", 0);
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
constexpr uint64_t kMaxBuffer = 0x10000;

// The request block op=800 passes, plus its descriptor arrays.
struct InvokeRequest {
  uint32_t methodId;
  uint32_t numIn;
  uint32_t numOut;
  uint32_t pad;
  const uint64_t *inDesc;  // {const void *data; uint64 size}  (16 bytes)
  const uint64_t *outDesc; // {void *data; uint64 cap; ...}    (24 bytes)
  int32_t *result;
};

// Strides confirmed by dumping multi-descriptor invokes on both platforms: a
// 5-output SceNpService call on PS5 fw 01.14.00 and a 2-output one on PS4 both
// decode cleanly at 24 bytes per output descriptor and nowhere else.
constexpr uint32_t kInDescWords = 2;
constexpr uint32_t kOutDescWords = 3;

bool readable(const void *p, uint64_t n) {
  return p && n && n <= kMaxBuffer && utl::isMemoryRangeMapped(p, n);
}

// ---------------------------------------------------------------- clients

struct Client {
  std::string service;
  Service *impl = nullptr;
};

std::mutex g_clientsMtx;
std::unordered_map<uint32_t, Client> g_clients;
std::atomic<uint32_t> g_nextKid{1};

Service *findService(const char *name) {
  if (!name)
    return nullptr;
  Service *all[] = {&playGoService(), &npManagerService(), &npWebService(),
                    &userService()};
  for (Service *s : all)
    if (std::strcmp(s->name(), name) == 0)
      return s;
  return nullptr;
}

Client lookupClient(uint32_t kid) {
  std::lock_guard<std::mutex> lk(g_clientsMtx);
  auto it = g_clients.find(kid);
  return it == g_clients.end() ? Client{} : it->second;
}

// The create/connect payload carries a pointer to the service name somewhere in
// its fields; find it rather than assuming a fixed offset (the block's shape
// differs between the two ops and across SDK versions).
const char *payloadServiceName(const void *in, uint64_t insize) {
  auto *q = static_cast<const uint64_t *>(in);
  for (uint64_t i = 0; i < insize / 8; i++) {
    uint64_t v = q[i];
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

std::atomic<uint64_t> g_opHist[2048];
std::atomic<uint64_t> g_methodHist[64];
std::atomic<uint32_t> g_methodId[64];

bool traceOn() {
  return kIpmiTrace;
}

// DELTA_IPMI_HIST: per-op and per-method call counts, dumped every 15s.
// Deliberately lock-free: a mutex-guarded version throttled the very spin it was
// meant to measure, so the numbers lied.
void histogram(uint32_t op, const InvokeRequest *req) {
  if (!kIpmiHist)
    return;
  if (op < 2048)
    g_opHist[op].fetch_add(1, std::memory_order_relaxed);
  if (req) {
    const uint32_t m = req->methodId;
    for (uint32_t i = 0; i < 64; i++) {
      uint32_t want = g_methodId[i].load(std::memory_order_relaxed);
      if (want == m) {
        g_methodHist[i].fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (!want) {
        uint32_t expect = 0;
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
        for (uint32_t i = 0; i < 2048; i++)
          if (uint64_t c = g_opHist[i].load(std::memory_order_relaxed))
            std::fprintf(stderr, "[ipmihist] op=%u %llu\n", i,
                         (unsigned long long)c);
        for (uint32_t i = 0; i < 64; i++)
          if (uint64_t c = g_methodHist[i].load(std::memory_order_relaxed))
            std::fprintf(stderr, "[ipmihist] method=%#x %llu\n",
                         g_methodId[i].load(std::memory_order_relaxed),
                         (unsigned long long)c);
      }
    }).detach();
    return true;
  }();
  (void)started;
}

void traceInvoke(uint32_t kid, const char *svc, const InvokeRequest *req,
                 bool handled) {
  if (!traceOn())
    return;
  std::fprintf(stderr, "[ipmi] %s kid=%u method=%#x in=%u out=%u%s\n",
               svc && *svc ? svc : "?", kid, req->methodId, req->numIn,
               req->numOut, handled ? "" : " (default)");
}

// DELTA_IPMI_DUMP=<method>: dump one unknown method's descriptors and the guest
// call chain that made it, so its shape can be read instead of guessed. The
// request block sits on the guest stack, so scanning up from it finds the
// callers. Bounded to a handful of hits; this is a research knob.
void dumpInvoke(uint32_t kid, const char *svc, const InvokeRequest *req) {
  if (!kIpmiDump || req->methodId != kIpmiDump)
    return;
  static std::atomic<int> seen{0};
  if (seen.fetch_add(1) >= 2)
    return;

  std::fprintf(stderr, "[ipmidump] %s kid=%u method=%#x in=%u out=%u\n",
               svc && *svc ? svc : "?", kid, req->methodId, req->numIn,
               req->numOut);
  auto descriptors = [](const char *tag, const uint64_t *d, uint32_t n,
                        uint32_t stride) {
    if (!utl::isMemoryRangeMapped(d, n * stride * 8))
      return;
    for (uint32_t i = 0; i < n; i++) {
      std::fprintf(stderr, "[ipmidump]   %s[%u] data=%#llx size=%#llx", tag, i,
                   (unsigned long long)d[i * stride],
                   (unsigned long long)d[i * stride + 1]);
      if (stride > 2) // the unidentified third word; see outSlot()
        std::fprintf(stderr, " w2=%#llx",
                     (unsigned long long)d[i * stride + 2]);
      auto *p = reinterpret_cast<const uint8_t *>(d[i * stride]);
      const uint64_t sz = d[i * stride + 1];
      if (readable(p, sz) && sz <= 64) {
        std::fprintf(stderr, " :");
        for (uint64_t k = 0; k < sz; k++)
          std::fprintf(stderr, " %02x", p[k]);
      }
      std::fprintf(stderr, "\n");
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
      std::fprintf(stderr, "[ipmidump]   caller %s\n", sym);
      shown++;
    }
  }
}

// DELTA_IPMI_OPDUMP=<op>: same idea as dumpInvoke, but for a MANAGER op the
// decoder does not know. An unhandled op that a module spins on is answered
// "empty success", which is a guess; dumping its request block and the guest
// call chain shows what the caller actually reads back, so the op can be
// implemented instead of guessed at.
void dumpManagerOp(uint32_t op, uint32_t kid, void *out, void *in,
                   uint64_t insize) {
  if (!kIpmiOpDump || op != kIpmiOpDump)
    return;
  static std::atomic<int> seen{0};
  if (seen.fetch_add(1) >= 3)
    return;

  std::fprintf(stderr, "[ipmiop] op=%u kid=%u out=%p in=%p insize=%#llx\n", op,
               kid, out, in, (unsigned long long)insize);
  auto hexdump = [](const char *tag, const void *p, uint64_t n) {
    if (!p || !readable(p, n))
      return;
    const auto *b = static_cast<const uint8_t *>(p);
    std::fprintf(stderr, "[ipmiop]   %s:", tag);
    for (uint64_t i = 0; i < n && i < 96; i++)
      std::fprintf(stderr, " %02x", b[i]);
    std::fprintf(stderr, "\n");
    // Any 8-byte field that looks like a guest pointer is worth following:
    // these blocks are mostly pointers to status words the caller polls.
    for (uint64_t i = 0; i + 8 <= n && i < 96; i += 8) {
      uint64_t v = 0;
      std::memcpy(&v, b + i, sizeof(v));
      const auto *t = reinterpret_cast<const uint8_t *>(v);
      if (v >= 0x1000 && readable(t, 8)) {
        std::fprintf(stderr, "[ipmiop]     +%#llx -> %#llx :",
                     (unsigned long long)i, (unsigned long long)v);
        for (int k = 0; k < 8; k++)
          std::fprintf(stderr, " %02x", t[k]);
        std::fprintf(stderr, "\n");
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
      std::fprintf(stderr, "[ipmiop]   caller %s\n", sym);
      shown++;
    }
  }
}

} // namespace

// ---------------------------------------------------------------- Invocation

Invocation::Invocation(void *request) : req_(request) {}

uint32_t Invocation::method() const {
  return static_cast<InvokeRequest *>(req_)->methodId;
}
uint32_t Invocation::numIn() const {
  return static_cast<InvokeRequest *>(req_)->numIn;
}
uint32_t Invocation::numOut() const {
  return static_cast<InvokeRequest *>(req_)->numOut;
}

const void *Invocation::input(uint32_t i, uint64_t &size) const {
  size = 0;
  auto *r = static_cast<InvokeRequest *>(req_);
  if (i >= r->numIn || !r->inDesc)
    return nullptr;
  const uint64_t *d = r->inDesc + i * kInDescWords;
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
bool Invocation::outSlot(uint32_t i, uint8_t *&data, uint64_t &cap) const {
  data = nullptr;
  cap = 0;
  auto *r = static_cast<InvokeRequest *>(req_);
  if (i >= r->numOut || !r->outDesc)
    return false;
  const uint64_t *d = r->outDesc + i * kOutDescWords;
  auto *p = reinterpret_cast<uint8_t *>(d[0]);
  if (!readable(p, d[1]))
    return false;
  data = p;
  cap = d[1];
  return true;
}

bool Invocation::reply(uint32_t i, const void *data, uint64_t size) {
  uint8_t *p;
  uint64_t cap;
  if (!outSlot(i, p, cap))
    return false;
  const uint64_t n = size < cap ? size : cap;
  std::memcpy(p, data, n);
  if (n < cap)
    std::memset(p + n, 0, cap - n);
  return true;
}

bool Invocation::replyU32(uint32_t i, uint32_t v) {
  return reply(i, &v, sizeof(v));
}

bool Invocation::replyFill(uint32_t i, uint8_t byte) {
  uint8_t *p;
  uint64_t cap;
  if (!outSlot(i, p, cap))
    return false;
  std::memset(p, byte, cap);
  return true;
}

void Invocation::replyEmpty() {
  auto *r = static_cast<InvokeRequest *>(req_);
  for (uint32_t i = 0; i < r->numOut; i++)
    replyFill(i, 0);
}

void Invocation::setResult(int32_t v) {
  auto *r = static_cast<InvokeRequest *>(req_);
  if (utl::isMemoryRangeMapped(r->result, sizeof(*r->result)))
    *r->result = v;
}

// ---------------------------------------------------------------- manager

int managerCall(uint32_t op, uint32_t kid, void *out, void *in,
                uint64_t insize) {
  auto *req = op == kInvokeSync && in && insize >= sizeof(InvokeRequest)
                  ? static_cast<InvokeRequest *>(in)
                  : nullptr;
  histogram(op, req);

  auto setResult = [&](uint32_t v) {
    if (out)
      *static_cast<uint32_t *>(out) = v;
  };

  switch (op) {
  case kCreateClient: {
    const uint32_t newKid = g_nextKid.fetch_add(1);
    const char *svc = in ? payloadServiceName(in, insize) : nullptr;
    Client c;
    c.service = svc ? svc : "";
    c.impl = findService(svc);
    if (traceOn())
      std::fprintf(stderr, "[ipmi] create kid=%u service=\"%s\"%s\n", newKid,
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
    // synchronous path gives.
    if (in && insize >= 40) {
      auto *b = static_cast<uint8_t *>(in);
      uint32_t status = 0;
      std::memcpy(&status, b + 8, sizeof(status));
      if (status == 0xFFFFFFFFu) {
        const uint32_t done = 0;
        std::memcpy(b + 8, &done, sizeof(done));
      }
      uint32_t *words[2] = {};
      std::memcpy(&words[0], b + 24, sizeof(words[0]));
      std::memcpy(&words[1], b + 32, sizeof(words[1]));
      for (uint32_t *w : words)
        if (utl::isMemoryRangeMapped(w, sizeof(*w)))
          *w = 0;
    }
    setResult(0);
    return 0;
  }

  case kStopSession:
    // The payload's first field points at a guest status word that libSceIpmi
    // asserts is zero once the manager syscall returns.
    if (in && insize >= sizeof(uint64_t)) {
      uint32_t *status = nullptr;
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
      std::fprintf(stderr, "[ipmi] op=%u kid=%u (unhandled, empty success)\n",
                   op, kid);
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
