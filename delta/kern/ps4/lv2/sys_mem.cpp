
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <base.h>
#include <logger/logger.h>
#include <utl/mem.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "../../proc.h"
#include "../../crash.h"
#include "../audio_daemon.h"
#include "../../guest_vaspace.h"
#include "../../thread_names.h"
#include "error_table.h"
#include "sys_mem.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kShmFilter, "DELTA_SHM_AUDIO_FILTER", nullptr);
DELTA_OPTION(const char *, kShmPoison, "DELTA_SHM_AUDIO_POISON", nullptr);
DELTA_OPTION(const char *, kShmPoisonFilter, "DELTA_SHM_AUDIO_POISON_FILTER", nullptr);
DELTA_OPTION(bool, kShmRepoison, "DELTA_SHM_AUDIO_REPOISON", false);
DELTA_OPTION(const char *, kShmProbe, "DELTA_SHM_AUDIO_PROBE", nullptr);
DELTA_OPTION(const char *, kShmDump, "DELTA_SHM_AUDIO_DUMP", nullptr);
DELTA_OPTION(unsigned, kShmDumpMs, "DELTA_SHM_AUDIO_DUMP_MS", 10);
DELTA_OPTION(unsigned, kShmDumpN, "DELTA_SHM_AUDIO_DUMP_N", 400);
DELTA_OPTION(size_t, kShmDumpMax, "DELTA_SHM_AUDIO_DUMP_MAX", 1u << 20);
DELTA_OPTION(const char *, kMmapCaller, "DELTA_MMAP_CALLER", nullptr);
DELTA_OPTION(bool, kShmNoAuto, "DELTA_SHM_NOAUTO", false);
DELTA_OPTION(bool, kAllocTrace, "DELTA_ALLOC_TRACE", false);
DELTA_OPTION(bool, kGnmapTrace, "DELTA_GNMAP_TRACE", false);
DELTA_OPTION(bool, kMmapLog, "DELTA_MMAP_LOG", false);
DELTA_OPTION(bool, kMmapfdTrace, "DELTA_MMAPFD_TRACE", false);
DELTA_OPTION(bool, kShmAudioDumpDelta, "DELTA_SHM_AUDIO_DUMP_DELTA", false);
DELTA_OPTION(bool, kShmAudioTrace, "DELTA_SHM_AUDIO_TRACE", false);
}  // namespace

namespace krnl {

using ppt = utl::pageProtection;
using alt = utl::allocationType;

// Floor of the arena we hand out addresses from (see allocLowGuest). Below it is
// the guest's own space, including the round 64 GiB slots titles MAP_FIXED their
// direct/flexible pools into.
#ifdef __ANDROID__
constexpr uintptr_t kGuestArenaFloor = 0x4000000000ull;  // 256 GiB
#else
constexpr uintptr_t kGuestArenaFloor = 0x8000000000ull;  // 512 GiB
#endif

// Ranges the guest has unmapped. sys_munmap deliberately keeps the host pages
// mapped, so the MAP_FIXED_NOREPLACE probe a hint goes through always fails
// there and the mapping gets relocated. That is fatal to an allocator that
// reserves a padded region, frees it, then re-reserves an exact aligned
// sub-range of it: V8's pointer-compression cage does exactly that, and a
// relocated cage leaves every compressed pointer resolving into memory nothing
// lives in.
namespace {
struct ReleasedRange {
  uintptr_t base, end;
};
std::mutex g_releasedLock;
std::vector<ReleasedRange> g_released;
}  // namespace

void noteGuestReleased(uint8_t *ptr, size_t size) {
  if (!ptr || !size)
    return;
  const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lk(g_releasedLock);
  for (auto &r : g_released) {
    if (base <= r.end && base + size >= r.base) {  // merge touching ranges
      r.base = std::min(r.base, base);
      r.end = std::max(r.end, base + size);
      return;
    }
  }
  if (g_released.size() < 4096)
    g_released.push_back({base, base + size});
}

bool wasGuestReleased(uint8_t *ptr, size_t size) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lk(g_releasedLock);
  for (const auto &r : g_released)
    if (base >= r.base && base + size <= r.end)
      return true;
  return false;
}

// PS4 user-space pointers must live below 2^40: libc's sceLibcMspaceCreate (and
// other allocators) reject a base whose bits >= 40 are set. The host kernel
// hands mmap(NULL) addresses far above that ceiling, so when we have to pick an
// address ourselves, carve it from a dedicated low arena instead. Bump-only and
// MAP_FIXED_NOREPLACE so we never clobber an existing mapping.
uint8_t *allocLowGuest(size_t size, size_t align) {
#ifdef __ANDROID__
  // 39-bit user VA: keep the guest arena clear of the module region (32..~224
  // GiB) and the FEX heap / bionic up top, and still under the PS4 2^40 ceiling.
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 256 GiB
  constexpr uintptr_t kCeil = 0x6000000000ull;    // 384 GiB
#else
  // Start the arena at 512 GiB. Titles map their own fixed-address direct/flexible
  // memory pools at round 64 GiB slots (N * 0x10_0000_0000): Uncharted 2 uses
  // 0x10..0x12_0000_0000 (Onion/Garlic/Flexible); GTA:SA's Gameface engine
  // MAP_FIXEDs pools at 0x10/0x20/0x30/0x40_0000_0000, the last being a 128 MB
  // direct-memory pool exactly on our old 256 GiB floor -- it clobbered the
  // primary TCB (fs:0x10 -> 0), which crashed the first scePthreadMutexLock. Our
  // bookkeeping must sit above every slot a title fixed-maps; 512 GiB clears all
  // observed pools while staying under the PS4 2^40 user ceiling.
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 512 GiB
  constexpr uintptr_t kCeil = 0x10000000000ull;   // 2^40, the PS4 user ceiling
#endif
  // Align bases to 64 KiB, not just the 16 KiB page: GNM tiled textures/render
  // targets carry alignment requirements above a page, and titles that allocate
  // a GPU pool here and sub-allocate surfaces from its base assert when the base
  // isn't aligned enough (DOOM's rhiTextureGnm buffer-block alignment check).
  constexpr uintptr_t kAlign = 0x10000;
  static std::atomic<uintptr_t> next{kFloor};
  size = (size + 0x3FFF) & ~uintptr_t(0x3FFF);
  // Caller-requested alignment (MAP_ALIGNED(n) in the mmap flags): the kernel
  // CONTRACTUALLY returns a base aligned to 2^n. Engines size their arena
  // bookkeeping around it -- SotC reserves its streaming arenas with
  // MAP_ALIGNED(20) and indexes them by VA>>20; a 64 KiB-aligned base breaks
  // every lookup (AllocationTracker null-record crash in LoadInitialWorld).
  const uintptr_t al = align > kAlign ? align : kAlign;
  for (int tries = 0; tries < 8192; tries++) {
    uintptr_t raw = next.load(std::memory_order_relaxed);
    uintptr_t base = (raw + (al - 1)) & ~(al - 1);  // align the base up
    if (base + size + 0x4000 > kCeil)
      return nullptr;  // doesn't fit; do NOT poison `next` (CAS, not fetch_add)
    if (!next.compare_exchange_weak(raw, base + size + 0x4000,
                                    std::memory_order_relaxed))
      continue;  // another thread advanced it; reload and retry
    void *p = ::mmap(reinterpret_cast<void *>(base), size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == reinterpret_cast<void *>(base)) {
      if (kAllocTrace)
        std::printf("[lowalloc] %#lx +%#lx\n", (unsigned long)base,
                    (unsigned long)size);
      return static_cast<uint8_t *>(p);
    }
    if (p != MAP_FAILED)
      ::munmap(p, size);  // hint occupied; the CAS already skipped past it
  }
  return nullptr;
}

// POSIX shared memory (shm_open/shm_unlink/ftruncate + fd-backed mmap).
//
// A named shm object is sized with ftruncate then mmap'd to share a region
// between components. In a single guest process "shared" means the same name
// resolves to the same backing block, so every mapper sees one region. The block
// is a low (<2^40) guest allocation; ftruncate or the first mmap allocates it.
namespace {
struct shmBacking {
  uint8_t *base = nullptr;
  size_t size = 0;
};
std::mutex g_shmMutex;
std::unordered_map<std::string, shmBacking> g_shmByName;

// ---------------------------------------------------------------------------
// RESEARCH INSTRUMENTATION (env-gated, default OFF, no effect on the normal
// path). Used to reverse-engineer the shared-memory mixer protocol the REAL
// libSceAudioOut speaks when it runs LLE: it creates per-port POSIX shm named
// "/shm_<pid>_<n>[_A]" and mixes into it, expecting the system audio daemon to
// consume. See the note at the head of runtime/vprx/ps4/libSceAudioOut.
//
//   DELTA_SHM_AUDIO_TRACE=1        log shm_open/ftruncate/mmap for matching names
//   DELTA_SHM_AUDIO_FILTER=<sub>   name substring to match (default "shm_")
//   DELTA_SHM_AUDIO_DUMP=<dir>     periodically snapshot every matching region
//   DELTA_SHM_AUDIO_DUMP_MS=<n>    snapshot period, default 10 ms
//   DELTA_SHM_AUDIO_DUMP_N=<n>     max snapshots per region, default 400
//   DELTA_SHM_AUDIO_DUMP_MAX=<n>   cap bytes copied per region, default 1 MiB
// Snapshots are appended to <dir>/<name>.bin and indexed in <dir>/index.txt as
// "<seq> <t_us> <name> <off> <len>", so a grower/reallocation is visible.
// ---------------------------------------------------------------------------
const char *shmAudioFilter() {
  const char *v = kShmFilter;
  return (v && *v) ? v : "shm_";
}

bool shmAudioMatch(const std::string &n) {
  return n.find(shmAudioFilter()) != std::string::npos;
}

bool shmAudioTraceOn() {
  return kShmAudioTrace;
}

uint64_t shmAudioNowUs() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void shmAudioTrace(const char *op, const std::string &name, const void *base,
                   size_t size, size_t off) {
  if (!shmAudioTraceOn() || !shmAudioMatch(name))
    return;
  std::fprintf(stderr,
               "[shmaudio] t=%llu tid=%ld %-9s '%s' base=%p size=%#zx off=%#zx\n",
               (unsigned long long)shmAudioNowUs(), (long)gettid(), op,
               name.c_str(), base, size, off);
}

std::string shmAudioSanitize(const std::string &n) {
  std::string s;
  for (char c : n)
    s += (c == '/' || c == '\\') ? '_' : c;
  return s;
}

// DELTA_SHM_AUDIO_POISON=<byte>: fill a matching region with a poison byte when
// it is first mapped. Without this a region the guest fills with SILENCE is
// indistinguishable from one the guest never touches -- both read back as
// zeros. Only regions matching DELTA_SHM_AUDIO_POISON_FILTER (default "_A", the
// per-port sample regions) are poisoned, so the descriptor block stays clean.
bool shmAudioPoisonQuiet(const std::string &name, uint8_t *base, size_t size) {
  const char *pv = kShmPoison;
  if (!pv || !base || !size)
    return false;
  const char *pf = (kShmPoisonFilter && *kShmPoisonFilter.get())
                       ? kShmPoisonFilter.get()
                       : "_A";
  if (name.find(pf) == std::string::npos)
    return false;
  std::memset(base, (int)std::strtol(pv, nullptr, 0), size);
  return true;
}

void shmAudioPoison(const std::string &name, uint8_t *base, size_t size) {
  if (shmAudioPoisonQuiet(name, base, size))
    std::fprintf(stderr, "[shmaudio] poisoned '%s' %p +%#zx\n", name.c_str(),
                 base, size);
}

void shmAudioRepoison(const std::string &name, uint8_t *base, size_t size) {
  if (!kShmRepoison)
    return;
  shmAudioPoisonQuiet(name, base, size);
}

std::atomic<bool> g_shmAudioDumper{false};

void shmAudioDumperMain(std::string dir, unsigned periodMs, unsigned maxSnaps,
                        size_t maxBytes, bool deltaOnly) {
  std::unordered_map<std::string, FILE *> files;
  std::unordered_map<std::string, std::vector<uint8_t>> prev;
  FILE *idx = std::fopen((dir + "/index.txt").c_str(), "w");
  struct reg { std::string n; uint8_t *b; size_t sz; };
  std::vector<uint8_t> cur;
  for (unsigned seq = 0; seq < maxSnaps; seq++) {
    std::vector<reg> regs;
    {
      std::lock_guard<std::mutex> lk(g_shmMutex);
      for (auto &kv : g_shmByName)
        if (kv.second.base && kv.second.size && shmAudioMatch(kv.first))
          regs.push_back({kv.first, kv.second.base, kv.second.size});
    }
    const uint64_t t = shmAudioNowUs();
    for (auto &r : regs) {
      FILE *&f = files[r.n];
      if (!f)
        f = std::fopen((dir + "/" + shmAudioSanitize(r.n) + ".bin").c_str(), "wb");
      if (!f)
        continue;
      const size_t len = r.sz < maxBytes ? r.sz : maxBytes;
      // Racy by construction: the guest mixes while we copy. Tearing shows up as
      // a torn sample block, never as a wrong cursor VALUE, so cursor tracking
      // stays sound; treat the sample area as "sampled", not "coherent".
      cur.assign(r.b, r.b + len);
      auto &p = prev[r.n];
      const bool same = (p.size() == len) &&
                        std::memcmp(p.data(), cur.data(), len) == 0;
      // Cheap per-tick fingerprint so a 75 s run can be judged without keeping
      // 75 s of bytes: how much of the region is non-zero, and an FNV-1a hash.
      size_t nz = 0;
      uint64_t h = 1469598103934665603ull;
      for (size_t i = 0; i < len; i++) {
        nz += cur[i] != 0;
        h = (h ^ cur[i]) * 1099511628211ull;
      }
      long off = -1;
      if (!deltaOnly || !same) {
        off = std::ftell(f);
        std::fwrite(cur.data(), 1, len, f);
        p = cur;
      }
      // DELTA_SHM_AUDIO_REPOISON: refill with the poison byte after sampling, so
      // the NEXT snapshot shows exactly the bytes written during this tick. A
      // producer that rewrites the same block with silence every tick is
      // otherwise invisible -- silence over silence is no change. Destroys the
      // region's contents, so it is only valid while nothing consumes them.
      shmAudioRepoison(r.n, r.b, len);
      if (idx)
        std::fprintf(idx, "%u %llu %s %ld %zu %zu %016llx\n", seq,
                     (unsigned long long)t, r.n.c_str(), off, len, nz,
                     (unsigned long long)h);
    }
    if (idx)
      std::fflush(idx);
    ::usleep(periodMs * 1000);
  }
  for (auto &kv : files)
    if (kv.second) std::fclose(kv.second);
  if (idx) std::fclose(idx);
  std::fprintf(stderr, "[shmaudio] dumper finished\n");
}

// ---------------------------------------------------------------------------
// DELTA_SHM_AUDIO_PROBE=<us>: SPEC VALIDATION HARNESS, default OFF. Performs
// exactly the consumer half of the LLE libSceAudioOut handshake and reports what
// it finds, WITHOUT playing anything -- the point is to prove the decode, not to
// be the daemon (that is the next stage's job).
//
// Every <us> it walks the 26 port slots of "/shm_<pid>_C" and, for any slot
// whose +0x00 token is non-zero, reads grain*bytesPerFrame bytes from the head
// of "/shm_<pid>_<idx>_A", computes a peak level under the slot's declared
// format, and then zeroes +0x00 to release the port. Pair it with
// DELTA_AUDIOMIX_ACK so the guest's mixer wait also completes.
//
// If the layout below is right, Isaac's peak here must track the HLE reference
// (climbing from ~0.03 to ~0.4); if it is wrong, the peak is garbage or zero.
void shmAudioProbeMain(long periodUs) {
  constexpr size_t kHdr = 0x20, kStride = 0x250, kSlots = 26;
  uint64_t ticks = 0, blocks = 0;
  float peakMax = 0.f;
  auto last = shmAudioNowUs();
  for (;;) {
    ::usleep((useconds_t)periodUs);
    ticks++;
    uint8_t *ctlRaw = nullptr;
    size_t ctlSize = 0;
    struct areg { int idx; uint8_t *b; size_t sz; };
    std::vector<areg> as;
    {
      std::lock_guard<std::mutex> lk(g_shmMutex);
      for (auto &kv : g_shmByName) {
        if (!kv.second.base) continue;
        const std::string &n = kv.first;
        if (n.size() > 2 && n.compare(n.size() - 2, 2, "_C") == 0 &&
            n.compare(0, 5, "/shm_") == 0) {
          ctlRaw = kv.second.base;
          ctlSize = kv.second.size;
        } else if (n.size() > 2 && n.compare(n.size() - 2, 2, "_A") == 0 &&
                   n.compare(0, 5, "/shm_") == 0) {
          // "/shm_<pid>_<idx>_A" -> idx
          size_t e = n.size() - 2;            // at the '_' of "_A"
          size_t s = n.rfind('_', e - 1);
          if (s != std::string::npos)
            as.push_back({std::atoi(n.c_str() + s + 1), kv.second.base,
                          kv.second.size});
        }
      }
    }
    if (!ctlRaw || ctlSize < kHdr + kSlots * kStride)
      continue;
    for (size_t k = 0; k < kSlots; k++) {
      uint8_t *slot = ctlRaw + kHdr + k * kStride;
      auto u32 = [&](size_t o) { return *reinterpret_cast<volatile uint32_t *>(slot + o); };
      const uint32_t bpf = u32(0x08), rate = u32(0x34), grain = u32(0x60);
      const uint32_t stype = u32(0x2c), state = u32(0x90);
      if (!bpf || !grain)
        continue;                       // slot never opened
      if (!u32(0x00))
        continue;                       // no block pending
      uint8_t *src = nullptr;
      size_t srcSz = 0;
      for (auto &a : as)
        if (a.idx == (int)k) { src = a.b; srcSz = a.sz; }
      const size_t need = (size_t)grain * bpf;
      float peak = 0.f;
      if (src && srcSz >= need) {
        const uint32_t bps = (stype == 0) ? 2u : 4u;
        const uint32_t n = need / bps;
        if (stype == 1) {
          const float *p = reinterpret_cast<const float *>(src);
          for (uint32_t i = 0; i < n; i++) {
            float a = p[i] < 0 ? -p[i] : p[i];
            if (a > peak) peak = a;
          }
        } else if (stype == 0) {
          const int16_t *p = reinterpret_cast<const int16_t *>(src);
          for (uint32_t i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -p[i] : p[i]) / 32768.f;
            if (a > peak) peak = a;
          }
        } else {
          const int32_t *p = reinterpret_cast<const int32_t *>(src);
          for (uint32_t i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -(float)p[i] : (float)p[i]) / 2147483648.f;
            if (a > peak) peak = a;
          }
        }
      }
      blocks++;
      if (peak > peakMax) peakMax = peak;
      const uint64_t now = shmAudioNowUs();
      if (now - last > 2000000) {
        last = now;
        std::fprintf(stderr,
                     "[shmprobe] port=%zu bpf=%u type=%u ch=%u rate=%u grain=%u "
                     "state=%u blocks=%llu peak=%.4f peakMax=%.4f\n",
                     k, bpf, stype, bpf / (stype == 0 ? 2u : 4u), rate, grain,
                     state, (unsigned long long)blocks, peak, peakMax);
      }
      // Release the port: the guest's next submit is gated on this being 0.
      *reinterpret_cast<volatile uint64_t *>(slot + 0x00) = 0;
    }
    (void)ticks;
  }
}

void shmAudioProbeMaybeStart() {
  const char *v = kShmProbe;
  if (!v || !*v)
    return;
  static std::atomic<bool> started{false};
  bool e = false;
  if (!started.compare_exchange_strong(e, true))
    return;
  const long us = std::strtol(v, nullptr, 0);
  std::fprintf(stderr, "[shmprobe] consumer probe every %ldus\n", us);
  std::thread(shmAudioProbeMain, us > 0 ? us : 10667).detach();
}

// Start the periodic dumper once, on the first matching shm we see.
void shmAudioDumpMaybeStart() {
  const char *dir = kShmDump;
  if (!dir || !*dir)
    return;
  bool expected = false;
  if (!g_shmAudioDumper.compare_exchange_strong(expected, true))
    return;
  const unsigned ms = kShmDumpMs;
  const unsigned n = kShmDumpN;
  const size_t mx = kShmDumpMax;
  std::fprintf(stderr,
               "[shmaudio] dumper -> %s every %ums x%u (<=%#zx B)%s\n", dir, ms,
               n, mx, kShmAudioDumpDelta ? " delta-only" : "");
  std::thread(shmAudioDumperMain, std::string(dir), ms, n, mx,
              kShmAudioDumpDelta.get()).detach();
}

class shmObject : public kObject {
public:
  shmObject(proc *p, std::string nm)
      : kObject(p, kObject::oType::shm), shmName(std::move(nm)) {}
  std::string shmName;  // key into g_shmByName
};

// Return the backing block for a shm, allocating/growing it to cover the
// requested range. Caller must NOT hold g_shmMutex. -1 on failure.
uint8_t *shmMap(shmObject *shm, size_t size, size_t offset) {
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = g_shmByName[shm->shmName];
  size_t need = (offset + size + 0x3FFF) & ~size_t(0x3FFF);
  if (!b.base && need) {
    b.base = allocLowGuest(need);
    if (!b.base)
      return reinterpret_cast<uint8_t *>(-1);
    b.size = need;
    proc::getActive()->getVma().add(b.base, need, ppt::w);
  }
  if (!b.base || offset > b.size)
    return reinterpret_cast<uint8_t *>(-1);
  shmAudioTrace("mmap", shm->shmName, b.base + offset, size, offset);
  // Also announce here: a region mmap'd without a prior ftruncate gets its
  // backing allocated above, and the audio daemon must not miss that case.
  audioDaemonNoticeShm(shm->shmName.c_str(), b.base, b.size);
  return b.base + offset;
}
}  // namespace

uint8_t *PS4ABI sys_mmap(void *addr, size_t size, uint32_t prot, uint32_t flags,
                         uint32_t fd, size_t offset) {
  auto *proc = proc::getActive();
  if (!proc)
    return reinterpret_cast<uint8_t *>(-1);

  if (flags & mFlags::stack || flags & mFlags::noextend)
    flags |= mFlags::anon;

  /*align the page*/
  size = (size + 0x3FFF) & 0xFFFFFFFFFFFFC000LL;

  // A zero-length mapping is invalid (BSD returns EINVAL). Guests hit this on an
  // error-recovery path, e.g. mmap()ing an fd from a failed physhm_open/fstat.
  // Without this it fell into allocLowGuest(0) -- 8192 failing mmap(len=0) host
  // calls -- and returned (uint8_t*)-1, which the errno convention reports as
  // EPERM (1) rather than EINVAL (22), misleading the guest's fallback.
  if (size == 0)
    return reinterpret_cast<uint8_t *>(-SysError::eINVAL);

  // SCOUT (DELTA_MMAP_CALLER=<minMB>): scan the guest stack for return addresses
  // in a loaded module's .text to pin which guest code requested a big map (e.g.
  // the libc heap). Handler runs on the guest stack on native.
  if (const char *mc = kMmapCaller) {
    size_t minB = static_cast<size_t>(std::strtoull(mc, nullptr, 0)) * 1024 * 1024;
    if (minB == 0) minB = 64ull * 1024 * 1024;
    if (size >= minB) {
      std::fprintf(stderr, "[mmap-caller] size=%#zx prot=%#x flags=%#x fd=%u:\n",
                   size, prot, flags, fd);
      auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
      int shown = 0;
      for (int i = 0; i < 8192 && shown < 12; i++) {
        uintptr_t v = sp[i];
        for (auto &m : proc->getModuleList()) {
          auto &mi = m->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && v >= (uintptr_t)t && v < (uintptr_t)t + mi.textSeg.size) {
            std::fprintf(stderr, "  sp+%-4x %s+%#lx\n", i * 8, mi.name.c_str(),
                         v - (uintptr_t)t);
            shown++;
            break;
          }
        }
      }
    }
  }

  // addr is a hint unless MAP_FIXED: relocate it rather than alias an existing map
  if (!(flags & mFlags::fixed)) {
    if (!addr || proc->getVma().overlaps(static_cast<uint8_t *>(addr), size))
      addr = nullptr;
  }

  // A hint inside the reserved user-stack region is guest-owned address space:
  // kern.usrstack hands the guest that region's top and libkernel places the
  // initial thread's system TLS just below it. Our own PROT_NONE reservation
  // makes the MAP_FIXED_NOREPLACE probe below fail, so the block would get
  // relocated and libkernel then reports its internal memory pool as exhausted.
  // Commit it where the guest asked instead.
  bool inUserStack = false;
  if (addr) {
    auto &env = proc->getEnv();
    auto a = reinterpret_cast<uintptr_t>(addr);
    auto lo = reinterpret_cast<uintptr_t>(env.userStack);
    inUserStack = env.userStack && a >= lo && a + size <= lo + env.userStackSize;
    // Same reasoning for the ranges reserveGuestVaSpace() claimed up front
    // (libkernel's arena, the GNM areas, the title pool slots): the only thing
    // occupying them is our own PROT_NONE placeholder, so a hint landing there
    // is guest-owned address space and must be COMMITTED where the guest asked.
    // Relocating instead is what makes libkernel fall back to its internal
    // arena and exhaust it -- see guest_vaspace.cpp.
    inUserStack = inUserStack || isGuestReservedVa(addr, size);
  }

  if (fd != -1) {
    auto *obj = proc->getObjTable().get(fd);
    if (kMmapfdTrace)
      std::fprintf(stderr, "[mmapfd] fd=%u addr=%p size=%#zx off=%#zx objType=%d\n",
                   fd, addr, size, offset, obj ? (int)obj->type() : -1);
    if (obj && obj->type() == kObject::oType::shm) {
      // POSIX shared memory: hand back the shared backing so every mapper of
      // this shm sees the same region (sized by ftruncate).
      return shmMap(static_cast<shmObject *>(obj), size, offset);
    }
    if (obj) {
      // Device-backed mmap (e.g. /dev/dce's scanout pool): use the region the
      // device hands back instead of an anonymous fallback, so the guest maps
      // the device's real memory. -1 means "not device-backed"; fall through.
      auto *m = static_cast<device *>(obj)->map(addr, size, prot, flags, offset);
      if (m != reinterpret_cast<uint8_t *>(-1)) {
        proc->getVma().add(
            m, size, static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx)),
            prot);
        return m;
      }
    }
  }

  // MAP_ALIGNED(n): bits 31..24 of the flags carry log2 of a base alignment the
  // kernel must honor (FreeBSD 9 semantics; Sony titles rely on it -- SotC
  // reserves streaming arenas with MAP_ALIGNED(20) and keys its allocator
  // bookkeeping on the 1 MiB-aligned base).
  const uint32_t alignLog = (flags >> 24) & 0x1F;
  const size_t mapAlign = (alignLog >= 14 && alignLog < 40)
                              ? (size_t(1) << alignLog)
                              : 0;
  if (mapAlign && addr && !(flags & mFlags::fixed) &&
      (reinterpret_cast<uintptr_t>(addr) & (mapAlign - 1)))
    addr = nullptr;  // misaligned hint: pick our own aligned base instead

  // A pure address-space reservation (prot 0) must not be planted in the band
  // titles MAP_FIXED their own direct/flexible pools into -- the round 64 GiB
  // slots below our arena floor. The hint there is only advisory, but a later
  // MAP_FIXED over it is not: Minecraft maps direct memory at 0x10_0000_0000,
  // which is also the hint V8 uses for its pointer-compression cage, and the
  // remap silently replaced the cage's first page (and with it V8's whole
  // read-only heap) long after the cage was handed out.
  if (addr && !(flags & mFlags::fixed) && prot == 0 &&
      reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ull &&
      reinterpret_cast<uintptr_t>(addr) < kGuestArenaFloor)
    addr = nullptr;

  void *ptr = nullptr;
  if (addr) {
    if (flags & mFlags::fixed) {
      // MAP_FIXED: the guest demands this exact address; overlay whatever's there.
      ptr = utl::allocMem(addr, size, ppt::w, alt::reservecommit);
      if (!ptr)
        ptr = utl::allocMem(addr, size, ppt::w, alt::commit);  // maybe pre-reserved
    } else if (inUserStack) {
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);
    } else if (utl::allocMem(addr, size, ppt::w, alt::reserve)) {
      // A hint must never alias an existing mapping. reservecommit uses MAP_FIXED
      // and would clobber it, so probe with a NOREPLACE reserve first and only
      // commit if the address was free; otherwise fall through to the low arena.
      // Without this a guest TLS/TCB hint lands on and destroys a loaded module
      // (seen on Android, where the guest hints into the low module region).
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);
    } else if (wasGuestReleased(static_cast<uint8_t *>(addr), size) &&
               !proc->getVma().overlaps(static_cast<uint8_t *>(addr), size)) {
      // The probe can only fail here because we kept the host pages of a range
      // the guest ITSELF unmapped, and nothing has been mapped there since. The
      // address is free as far as the guest is concerned, so honour the hint.
      ptr = utl::allocMem(addr, size, ppt::w, alt::reservecommit);
    }
  }
  // No usable hint (or it was taken): pick a low (<2^40) address the guest's
  // own allocators will accept, not whatever high address the host hands out.
  if (!ptr)
    ptr = allocLowGuest(size, mapAlign);
  if (!ptr) {
    return reinterpret_cast<uint8_t *>(-1);
  }

  // Track the prot the guest actually asked for (BSD r=1/w=2/x=4 maps 1:1 onto
  // pageProtection) so sceKernelVirtualQuery / QueryMemoryProtection report the
  // truth instead of a blanket rwx. The host pages stay rwx: FEX reads guest
  // memory directly and we don't deliver protection faults, so restricting them
  // would only risk spurious crashes, not faithful behaviour.
  auto gprot = static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx));

  // No zero-fill for anonymous maps: every ptr above comes from an anonymous
  // ::mmap (utl::allocMem or allocLowGuest), which the kernel already hands
  // back zeroed. Writing it ourselves faulted in the whole mapping -- a title
  // that maps multi-GiB pools (Minecraft maps 4 and 8 GiB ones) went to 43 GiB
  // RSS at ~3 GB/s and took the host down. The shm and device-backed paths
  // return before this point, so they are unaffected.

  // File-backed mmap: copy the file's content into the freshly-mapped pages so the
  // guest reads the file it mapped (Doom64 mmaps its asset/WAD files and samples
  // textures straight out of the mapping; an anonymous zero-fill left them black).
  // readAt is a no-op (-1) for non-file devices; the read stops at EOF so a sparse
  // over-sized mapping keeps zeros past the file's end.
  if (fd != static_cast<uint32_t>(-1)) {
    if (auto *o = proc->getObjTable().get(fd))
      if (o->type() == kObject::oType::device) {
        int64_t got = static_cast<device *>(o)->readAt(ptr, size, offset);
        if (got > 0 && kMmapfdTrace)
          std::fprintf(stderr, "[mmapfd]   filled %p from fd=%u off=%#zx -> %lld bytes\n",
                       ptr, fd, offset, (long long)got);
      }
  }

  // 0x100 = Sony MAP_VOID: an address-space reservation (titles later commit
  // pieces inside with MAP_FIXED, which punches the reservation apart in the
  // VMA). Virtual query must see it as reserved, not committed memory.
  const bool voidReserve = (flags & 0x100) && prot == 0;
  proc->getVma().add(static_cast<uint8_t *>(ptr), size, gprot, prot,
                     voidReserve);

  utl::protectMem(static_cast<void *>(ptr), size, ppt::rwx);

  // DELTA_GNMAP_TRACE: log any mapping that lands in the GNM/GPU aperture
  // (0x8000_.. below the big dmem pools) to pin how the AGC ring buffers (ACQRB
  // etc.) are mapped and by whom (return address). The AGC command ring is mapped
  // here as plain anon (fd=-1); its coherency with the guest's PM4 writes is an
  // open item (Onion/Garlic dual mapping -- see ps5-boot-progress memory).
  if (kGnmapTrace) {
    uint64_t r = reinterpret_cast<uint64_t>(ptr);
    if (r >= 0x8000000000ull && r < 0x8300000000ull) {
      // Walk the guest frame chain (libkernel keeps frame pointers) so we see the
      // real caller above libkernel's mmap wrapper, not just the wrapper itself.
      std::fprintf(stderr, "[gnmap] %p size=%#x fd=%d flags=%#x  callers:", ptr,
                   size, static_cast<int>(fd), flags);
      void *fp = __builtin_frame_address(0);
      for (int depth = 0; depth < 20 && fp; depth++) {
        auto *frame = static_cast<void **>(fp);
        void *ret = frame[1];
        if (!ret) break;
        char sym[160];
        krnl::symbolize(reinterpret_cast<uintptr_t>(ret), sym, sizeof(sym));
        std::fprintf(stderr, " [%s]", sym);
        fp = frame[0];
        if (reinterpret_cast<uintptr_t>(fp) < 0x10000) break;
      }
      std::fprintf(stderr, "\n");
    }
  }

  if (kMmapLog)
    std::printf("mmap %p, %zx, prot=%x flags=%x, %p -> %p\n", addr, size, prot,
                flags, _ReturnAddress(), ptr);

  if (flags & mFlags::stack)
    return &static_cast<uint8_t *>(ptr)[size];

  return static_cast<uint8_t *>(ptr);
}

int PS4ABI sys_mprotect(uint8_t *addr, size_t len, int prot) {
  auto *proc = proc::getActive();
  if (!proc)
    return -SysError::eINVAL;

  // BSD prot bits (r=1/w=2/x=4) map 1:1 onto pageProtection. Reflect the change
  // in the region we track so sceKernelVirtualQuery reports the new protection.
  // We don't restrict the host pages (see sys_mmap) and we don't fail on an
  // untracked range: the dynamic linker mprotects its own RELRO segments, which
  // the module loader maps outside this table, and erroring there would break
  // relocation finalisation. So update what we know and report success.
  auto np = static_cast<ppt>(prot & static_cast<uint32_t>(ppt::rwx));
  if (auto *region = proc->getVma().get(addr))
    region->prot = np;
  (void)len;
  return 0;
}

int PS4ABI sys_shm_open(const char *path, uint32_t flags, uint16_t mode) {
  auto *proc = proc::getActive();
  if (!proc || !path)
    return -SysError::eINVAL;

  constexpr uint32_t kO_CREAT = 0x0200, kO_EXCL = 0x0800;
  std::string name(path);
  {
    std::lock_guard<std::mutex> lk(g_shmMutex);
    auto it = g_shmByName.find(name);
    if (it == g_shmByName.end()) {
      if (!(flags & kO_CREAT) && kShmNoAuto) {
        // DIAGNOSTIC: restore the pre-LLE behaviour (fail an open of a system shm
        // the guest didn't create) to test whether auto-providing it makes a title
        // block waiting for a ShellCore handshake that never arrives (Doom64).
        std::fprintf(stderr, "[shm_open] NOAUTO: '%s' -> ENOENT\n", name.c_str());
        return -SysError::eNOENT;
      }
      if (!(flags & kO_CREAT)) {
        // A read-only open of a shm that wasn't created by the guest: this is a
        // SYSTEM shared region the kernel would have published at boot (e.g.
        // libSceAvSetting's audio/video settings block). We don't model its
        // contents, so auto-provide a zeroed, pre-sized backing -- the title
        // then fstat()s a real size and mmaps it (reading defaults) instead of
        // failing init with a -ENOENT shm fd it tries to map anyway.
        shmBacking b;
        b.size = 0x10000;  // 64 KiB, ample for a settings block
        b.base = allocLowGuest(b.size);
        if (b.base) {
          std::memset(b.base, 0, b.size);
          proc->getVma().add(b.base, b.size, ppt::w);
        }
        std::fprintf(stderr, "[shm_open] auto-provide system shm '%s' size=%#zx\n",
                     name.c_str(), b.size);
        g_shmByName.emplace(name, b);
      } else {
        g_shmByName.emplace(name, shmBacking{});  // empty; sized later by ftruncate
      }
    } else if ((flags & kO_CREAT) && (flags & kO_EXCL)) {
      return -SysError::eEXIST;
    }
  }

  // A fresh fd per open, all sharing the named backing (POSIX-ish for a single
  // guest process). The ctor registers it in the object table.
  auto *obj = new shmObject(proc, std::move(name));
  std::fprintf(stderr, "[shm_open] '%s' flags=%#x -> fd=%u\n", path, flags,
               obj->handle());
  shmAudioTrace("shm_open", obj->shmName, nullptr, 0, flags);
  shmAudioDumpMaybeStart();
  shmAudioProbeMaybeStart();
  return obj->handle();
}

int PS4ABI sys_shm_unlink(const char *path) {
  if (!path)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto it = g_shmByName.find(path);
  if (it == g_shmByName.end())
    return -SysError::eNOENT;
  // Drop the name only. Any region already handed to mmap is a raw pointer that
  // stays valid; we keep the host allocation (reclaimed at process exit).
  g_shmByName.erase(it);
  return 0;
}

// Report a shm fd's backing size for fstat (shm objects aren't device-backed,
// so sys_fstat's fdToDevice path can't size them). Returns SIZE_MAX if `fd`
// isn't a shm, so the caller falls through to the normal path.
size_t shmFstatSize(uint32_t fd) {
  auto *proc = proc::getActive();
  if (!proc)
    return SIZE_MAX;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::shm)
    return SIZE_MAX;
  auto *shm = static_cast<shmObject *>(obj);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  return g_shmByName[shm->shmName].size;
}

int PS4ABI sys_ftruncate(uint32_t fd, int64_t length) {
  auto *proc = proc::getActive();
  if (!proc || length < 0)
    return -SysError::eINVAL;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::shm)
    return -SysError::eBADF;

  auto *shm = static_cast<shmObject *>(obj);
  size_t want = (static_cast<size_t>(length) + 0x3FFF) & ~size_t(0x3FFF);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = g_shmByName[shm->shmName];
  // The RAW length matters for the protocol spec (the rounded `want` hides it).
  shmAudioTrace("ftruncate", shm->shmName, b.base,
                static_cast<size_t>(length), want);
  if (want == 0 || want <= b.size)
    return 0;
  uint8_t *nb = allocLowGuest(want);
  if (!nb)
    return -SysError::eNOMEM;
  if (b.base)
    std::memcpy(nb, b.base, b.size);  // grow before first mmap: preserve contents
  b.base = nb;
  b.size = want;
  proc->getVma().add(b.base, want, ppt::w);
  shmAudioTrace("sized", shm->shmName, b.base, want, 0);
  shmAudioPoison(shm->shmName, b.base, want);
  // The LLE libSceAudioOut's regions become consumable here; audioDaemonNotice
  // ignores every name that is not part of that protocol. Note the base can
  // MOVE on a later grow, which is why the daemon is told the current one rather
  // than caching a pointer.
  audioDaemonNoticeShm(shm->shmName.c_str(), b.base, b.size);
  return 0;
}

int PS4ABI sys_mname(uint8_t *ptr, size_t len, const char *name, void *) {
  auto *proc = proc::getActive();
  if (!proc)
    return -1;

  auto *info = proc->getVma().get(ptr);
  if (!info) {
    LOG_WARNING("attempted to tag unknown memory ({}, {})", fmt::ptr(ptr),
                name);
    return -1;
  }

  LOG_WARNING("tagged {}+{:#x} with name {}", fmt::ptr(ptr), len, name);
  info->name = name;
  // Titles name their thread STACKS this way; carry the tag onto the host
  // thread running on that stack (wait probe / gdb / perf attribution).
  nameThreadsForRange(ptr, len, name);
  return 0;
}

struct mdbg_property {
  int32_t unk;
  int32_t unk2;
  void *addr;
  size_t areaSize;
  int64_t unk3;
  int64_t unk4;
  char name[32];
};

static_assert(sizeof(mdbg_property) == 72);

int PS4ABI sys_mdbg_service(uint32_t op, void *arg1, void *arg2, void *a3) {
  switch (op) {
  case 1: {
    auto *info = static_cast<mdbg_property *>(arg1);
    LOG_WARNING("set property {} for addr {} with size {}", info->name,
                info->addr, info->areaSize);
    /*TODO: create named object*/

    break;
  }
  }

  return 0;
}

// sceKernelGet/SetDirectMemoryContainer: -1 queries the current container id,
// any other value selects it and returns the previous one. We don't enforce
// separate dmem pools, so just track the selected id (default 0) and never trap.
int PS4ABI sys_dmem_container(uint32_t op) {
  static std::atomic<uint32_t> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  return static_cast<int>(current.exchange(op));
}
} // namespace krnl