/*
 * PS4Delta : PS4 emulation and research project
 *
 * sys_ipmimgr_call (622): the kernel side of PS4 IPMI (Inter-Process Method
 * Invocation), the RPC the real Sony libraries use to reach the system-service
 * processes (ShellCore, LNC, AppMessaging, PlayGo, ...). We host no service
 * processes, so the job here is to answer the calls the game's libSceIpmi makes
 * well enough that those clients initialise instead of failing and leaving null
 * singletons.
 *
 * ABI (recovered from libSceIpmi's wrapper at .text+0x3fd0 -> libkernel stub ->
 * syscall 622): the kernel receives
 *     rdi=op  rsi=kid  rdx=out  r10=in  r8=insize  r9=arg6(0xdeadbadecafebeaf)
 * so the result buffer (out) comes BEFORE the request buffer (in). The wrapper
 * pre-sets the 32-bit result to -1 and, on a 0 (success) return, reads the
 * result word back out as the call's value (CreateClient's becomes m_clientKid).
 * A handler therefore returns 0 and writes its result word into `out`.
 *
 * PlayGo: libScePlayGo is a thin libSceIpmi client to the PlayGo system daemon
 * (service name "ScePlayGo"). scePlayGoOpen does invokeSyncMethod(0x30000 open,
 * 0x3000f get-chunk-count, 0x30008 get-loci, ...) and asserts FATAL when the
 * count is 0. With no daemon, the default success-with-empty-payload reply left
 * the count 0. We answer those invokes directly as a fully-installed, single
 * chunk title (all loci LOCAL_FAST, progress 100%) so the real library succeeds.
 */
#include <base.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "kern/vfs.h"

namespace krnl {

// IPMI manager command numbers (libSceIpmi op -> sys_ipmimgr_call).
enum {
  IPMI_CREATE_CLIENT = 2,  // sceIpmiMgrCreateClient  -> result = client kid
  IPMI_DESTROY_CLIENT = 3, // sceIpmiMgrDestroyClient -> result = 0 (asserted)
  IPMI_INVOKE_SYNC = 800,  // sceIpmiClientInvokeSyncMethod
  IPMI_CONNECT = 1024,     // sceIpmiClientConnect (carries the service name)
};

// ScePlayGo locus enum (per-chunk availability).
enum {
  PLAYGO_LOCUS_NOT_DOWNLOADED = 0,
  PLAYGO_LOCUS_LOCAL_SLOW = 2,
  PLAYGO_LOCUS_LOCAL_FAST = 3,
};

static std::atomic<uint32_t> g_nextClientKid{1};

// Kids whose IPMI client connected to the "ScePlayGo" service. Guarded because
// IPMI calls come from several guest threads.
static std::mutex g_playgoMtx;
static std::unordered_set<uint32_t> g_playgoKids;

static bool isPlayGoKid(uint32_t kid) {
  std::lock_guard<std::mutex> lk(g_playgoMtx);
  return g_playgoKids.count(kid) != 0;
}

// op=800 sceIpmiClientInvokeSyncMethod request, repacked for the syscall.
// in/out descriptors are arrays of {void* data; uint64_t size} (16 bytes each).
struct IpmiInvokeReq {
  uint32_t methodId;
  uint32_t numIn;
  uint32_t numOut;
  uint32_t pad;
  uint64_t *pInData;
  uint64_t *pOutData;
  int32_t *pResult;
};

static void dumpStr(const char *tag, void *p) {
  if (!p)
    return;
  auto *s = static_cast<const char *>(p);
  bool printable = true;
  for (int i = 0; i < 32; i++) {
    char c = s[i];
    if (c == 0)
      break;
    if (c < 0x20 || c > 0x7e) {
      printable = false;
      break;
    }
  }
  if (printable && s[0])
    std::fprintf(stderr, "    %s str=\"%.32s\"\n", tag, s);
}

static std::string kidService(uint32_t kid);

static void ipmiTrace(uint32_t op, uint32_t kid, void *out, void *in,
                      uint64_t insize) {
  static const bool on = std::getenv("DELTA_IPMI_TRACE") != nullptr;
  if (!on)
    return;
  std::fprintf(stderr, "[ipmi] op=%u kid=%u out=%p in=%p insize=%llu\n", op, kid,
               out, in, (unsigned long long)insize);
  if (op == IPMI_INVOKE_SYNC && in && insize >= sizeof(IpmiInvokeReq)) {
    auto *req = static_cast<IpmiInvokeReq *>(in);
    std::fprintf(stderr, "  invoke svc=\"%s\" method=%#x numIn=%u numOut=%u",
                 kidService(kid).c_str(), req->methodId, req->numIn,
                 req->numOut);
    // out descriptor sizes help identify the method's expected reply layout
    if (req->pOutData && req->numOut >= 1 && req->numOut <= 4) {
      std::fprintf(stderr, " outsz=[");
      for (uint32_t i = 0; i < req->numOut; i++)
        std::fprintf(stderr, "%s%llu", i ? "," : "",
                     (unsigned long long)req->pOutData[i * 2 + 1]);
      std::fprintf(stderr, "]");
    }
    std::fprintf(stderr, "\n");
  }
}

// Scan the request payload's pointer fields for the service-name string the
// create/connect calls carry (e.g. "ScePlayGo", "SceNpMgrIpc", ...).
static const char *payloadServiceName(void *in, uint64_t insize) {
  auto *q = static_cast<uint64_t *>(in);
  for (uint64_t i = 0; i < insize / 8; i++) {
    uint64_t v = q[i];
    if (v <= 0x10000) // skip nulls / small inline ints (never valid pointers)
      continue;
    auto *s = reinterpret_cast<const char *>(v);
    if (std::strncmp(s, "Sce", 3) == 0) {
      int n = 3;
      bool printable = true;
      for (; n < 64 && s[n]; n++)
        if (s[n] < 0x20 || s[n] > 0x7e) { printable = false; break; }
      if (printable && n < 64)
        return s;
    }
  }
  return nullptr;
}

static bool payloadNamesPlayGo(void *in, uint64_t insize) {
  const char *s = payloadServiceName(in, insize);
  return s && std::strncmp(s, "ScePlayGo", 9) == 0;
}

// kid -> service name (from the create payload), so invoke traffic is
// attributable per service in traces and per-service handlers can match.
static std::mutex g_kidNameMtx;
static std::unordered_map<uint32_t, std::string> g_kidNames;

static void rememberKid(uint32_t kid, const char *name) {
  std::lock_guard<std::mutex> lk(g_kidNameMtx);
  g_kidNames[kid] = name ? name : "";
}

static std::string kidService(uint32_t kid) {
  std::lock_guard<std::mutex> lk(g_kidNameMtx);
  auto it = g_kidNames.find(kid);
  return it != g_kidNames.end() ? it->second : std::string();
}

// PlayGo chunk count. The real per-title value lives in the pkg header file
// /app0/sce_sys/playgo-chunk.dat (magic "pgd\0"; chunk_count is a u16 at offset
// 0x0A). A wrong count breaks multi-chunk titles: Shadow of the Tomb Raider
// enumerates chunk loci 0..N during boot and, when the count it reads back is
// too small, an unsigned `count - 0x50` underflows into a ~4-billion-iteration
// loop that smashes the stack. Read the real value once; fall back to one chunk
// when the file is missing/unreadable (the single-chunk titles we already boot).
// DELTA_PLAYGO_CHUNKS overrides for experimentation.
static uint32_t playGoChunkCount() {
  static uint32_t cached = 0;
  if (cached)
    return cached;
  if (const char *ov = std::getenv("DELTA_PLAYGO_CHUNKS")) {
    int v = std::atoi(ov);
    cached = v > 0 ? static_cast<uint32_t>(v) : 1;
    return cached;
  }
  // Default: 0x50. Multi-chunk titles enumerate chunk loci 0..N at boot and
  // build per-chunk tables sized from this count; a too-small value underflows
  // those loops (Shadow of the Tomb Raider) and a count of 1 (the old default)
  // crashes. 0x50 is the value such titles treat as "the standard set, fully
  // installed", which is exactly what an emulated whole-pkg mount is. Reading a
  // real /app0/sce_sys/playgo-chunk.dat would refine this, but the pkgs we run
  // don't ship one (PlayGo data is absent), so a sane installed-count is best.
  cached = 0x50;
  utl::File f = vfs::openRead("/app0/sce_sys/playgo-chunk.dat");
  if (f.Exists()) {
    uint8_t hdr[0x10] = {};
    if (f.Read(hdr, sizeof(hdr)) == sizeof(hdr) && hdr[0] == 'p' &&
        hdr[1] == 'g' && hdr[2] == 'd') {
      uint32_t cc = static_cast<uint32_t>(hdr[0x0a] | (hdr[0x0b] << 8));
      if (cc > 0)
        cached = cc;
    }
  }
  return cached;
}

// Answer a PlayGo invokeSyncMethod as a fully-installed single-chunk title.
// Only the queried-data methods need a reply; the work area was zeroed at
// scePlayGoInitialize, so "nothing pending" getters and setters just need
// SCE_OK. We touch only the first out descriptor (data=pOutData[0],
// size=pOutData[1]) of single-output methods, since multi-output descriptor
// strides vary; writing the wrong one corrupts host memory.
static void playGoInvoke(IpmiInvokeReq *req) {
  uint8_t *out0 = req->pOutData && req->numOut >= 1
                      ? reinterpret_cast<uint8_t *>(req->pOutData[0])
                      : nullptr;
  uint64_t out0sz = req->pOutData && req->numOut >= 1 ? req->pOutData[1] : 0;
  if (req->pResult)
    *req->pResult = 0; // SCE_OK for the method itself

  switch (req->methodId) {
  case 0x30000: // open -> server-side handle (must be != 0 / != -1)
    if (out0 && out0sz >= 4)
      *reinterpret_cast<uint32_t *>(out0) = 1;
    break;
  case 0x3000f: // get chunk count (0 -> scePlayGoOpen FATAL)
    if (out0 && out0sz >= 4)
      *reinterpret_cast<uint32_t *>(out0) = playGoChunkCount();
    break;
  case 0x30008: // get per-chunk loci -> byte array indexed by chunk id
    if (out0 && out0sz && out0sz <= 0x10000)
      std::memset(out0, PLAYGO_LOCUS_LOCAL_FAST, out0sz);
    break;
  case 0x3000d: // get progress -> { uint64 progressSize; uint64 totalSize }
    if (out0 && out0sz >= 16) {
      reinterpret_cast<uint64_t *>(out0)[0] = 1; // progressSize
      reinterpret_cast<uint64_t *>(out0)[1] = 1; // totalSize (== 100%)
    }
    break;
  default:
    // Remaining getters (todo list, eta, install speed, language) and all
    // setters: the zeroed work area already means "installed, nothing pending".
    break;
  }
}

int PS4ABI sys_ipmimgr_call(uint32_t op, uint32_t kid, void *out, void *in,
                            uint64_t insize, uint64_t arg6) {
  (void)arg6;
  ipmiTrace(op, kid, out, in, insize);
  auto setResult = [&](uint32_t v) {
    if (out)
      *static_cast<uint32_t *>(out) = v;
  };

  switch (op) {
  case IPMI_CREATE_CLIENT: {
    // Hand back a fresh client kid; the wrapper stores it as m_clientKid. The
    // create payload carries a pointer to the service name, so remember the kid
    // if this is the PlayGo daemon (so we can answer its method invokes).
    uint32_t newKid = g_nextClientKid.fetch_add(1);
    const char *svc = in ? payloadServiceName(in, insize) : nullptr;
    if (svc)
      rememberKid(newKid, svc);
    if (std::getenv("DELTA_IPMI_TRACE"))
      std::fprintf(stderr, "  create kid=%u service=\"%s\"\n", newKid,
                   svc ? svc : "?");
    if (svc && std::strncmp(svc, "ScePlayGo", 9) == 0) {
      std::lock_guard<std::mutex> lk(g_playgoMtx);
      g_playgoKids.insert(newKid);
    }
    setResult(newKid);
    return 0;
  }
  case IPMI_DESTROY_CLIENT: {
    std::lock_guard<std::mutex> lk(g_playgoMtx);
    g_playgoKids.erase(kid);
    setResult(0);
    return 0;
  }
  case IPMI_INVOKE_SYNC:
    if (in && insize >= sizeof(IpmiInvokeReq) && isPlayGoKid(kid))
      playGoInvoke(static_cast<IpmiInvokeReq *>(in));
    setResult(0);
    return 0;
  case 784:
    // StopSession/Disconnect-style calls pass a pointer to a guest status word
    // in the request payload. libSceIpmi asserts that status is zero after the
    // manager syscall returns; leave it clear when no service process exists.
    if (in && insize >= sizeof(uint64_t)) {
      uint32_t *status = nullptr;
      std::memcpy(&status, in, sizeof(status));
      if (status)
        *status = 0;
    }
    setResult(0);
    return 0;
  default:
    // Other ops: report success with a zero result word and no payload so the
    // wrapper takes its success path.
    setResult(0);
    return 0;
  }
}

} // namespace krnl
