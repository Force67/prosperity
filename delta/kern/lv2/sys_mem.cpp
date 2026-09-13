
/*
 * PS4Delta: PS4 emulation and research project
 * Copyright 2019-2020 Force67. See LICENSE in the root of the source tree.
 */

#include <cstdio>
#include "base/arch.h"
#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
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

#include "kern/proc.h"
#include "kern/crash.h"
#include "kern/ps4/audio_daemon.h"
#include "kern/guest_vaspace.h"
#include "kern/thread_names.h"
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

// Floor of the guest address arena; below sit the round 64 GiB slots titles
// MAP_FIXED their direct/flexible pools into.
#ifdef __ANDROID__
constexpr uintptr_t kGuestArenaFloor = 0x4000000000ull;  // 256 GiB
#else
constexpr uintptr_t kGuestArenaFloor = 0x8000000000ull;  // 512 GiB
#endif

// Ranges the guest has unmapped. sys_munmap keeps the host pages mapped, so a
// hint there relocates, which breaks allocators that reserve a padded region,
// free it, then re-reserve an exact aligned sub-range (V8's pointer cage).
namespace {
struct ReleasedRange {
  uintptr_t base, end;
};
std::mutex g_releasedLock;
std::vector<ReleasedRange> g_released;
}  // namespace

void noteGuestReleased(u8 *ptr, size_t size) {
  if (!ptr || !size)
    return;
  utl::forgetMemoryMapping(ptr, size);
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

void noteGuestTaken(u8 *ptr, size_t size) {
  if (!ptr || !size)
    return;
  utl::forgetMemoryMapping(ptr, size);
  const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr), hi = lo + size;
  std::lock_guard<std::mutex> lk(g_releasedLock);
  for (auto it = g_released.begin(); it != g_released.end();) {
    if (lo < it->end && it->base < hi)
      it = g_released.erase(it);  // partially reused: no longer safe to reuse
    else
      ++it;
  }
}

bool wasGuestReleased(u8 *ptr, size_t size) {
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
u8 *allocLowGuest(size_t size, size_t align) {
#ifdef __ANDROID__
  // 39-bit user VA: keep the guest arena clear of the module region (32..~224
  // GiB) and the FEX heap / bionic up top, and still under the PS4 2^40 ceiling.
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 256 GiB
  constexpr uintptr_t kCeil = 0x6000000000ull;    // 384 GiB
#else
  // Start at 512 GiB: titles fixed-map pools at round 64 GiB slots (GTA:SA put a
  // 128 MB pool exactly on the old 256 GiB floor, clobbering the primary TCB).
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 512 GiB
  constexpr uintptr_t kCeil = 0x10000000000ull;   // 2^40, the PS4 user ceiling
#endif
  // 64 KiB, not just one page: GNM surfaces sub-allocated from a pool base
  // assert on weaker alignment (DOOM rhiTextureGnm).
  constexpr uintptr_t kAlign = 0x10000;
  static std::atomic<uintptr_t> next{kFloor};
  size = (size + 0x3FFF) & ~uintptr_t(0x3FFF);
  // MAP_ALIGNED(n) is contractual: SotC indexes its streaming arenas by VA>>20,
  // so a weaker base breaks every lookup.
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
      utl::trackMemoryMapping(p, size);
      if (kAllocTrace)
        BASE_LOGI("lowalloc", "{:#x} +{:#x}", (unsigned long)base,
                  (unsigned long)size);
      return static_cast<u8 *>(p);
    }
    if (p != MAP_FAILED)
      ::munmap(p, size);  // hint occupied; the CAS already skipped past it
  }
  return nullptr;
}

// POSIX shared memory (shm_open/shm_unlink/ftruncate + fd-backed mmap). In a
// single guest process the same name resolves to one backing block; ftruncate
// or the first mmap allocates it.
namespace {
struct shmBacking {
  u8 *base = nullptr;
  size_t size = 0;
};
using shmRef = std::shared_ptr<shmBacking>;
std::mutex g_shmMutex;
// name -> backing. The backing outlives the name (shmObject holds a shared
// ref), like the kernel refcounting the shm object by fd.
std::unordered_map<std::string, shmRef> g_shmByName;

// ---------------------------------------------------------------------------
// RESEARCH INSTRUMENTATION (env-gated, default OFF), for reverse-engineering
// the LLE libSceAudioOut shared-memory mixer protocol (see the note at the
// head of runtime/vprx/ps4/libSceAudioOut):
//   DELTA_SHM_AUDIO_TRACE=1        log shm_open/ftruncate/mmap for matching names
//   DELTA_SHM_AUDIO_FILTER=<sub>   name substring to match (default "shm_")
//   DELTA_SHM_AUDIO_DUMP=<dir>     periodically snapshot every matching region
//   DELTA_SHM_AUDIO_DUMP_MS/N/MAX  period (10 ms), count (400), byte cap (1 MiB)
// Snapshots append to <dir>/<name>.bin, indexed in <dir>/index.txt.
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

u64 shmAudioNowUs() {
  return (u64)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void shmAudioTrace(const char *op, const std::string &name, const void *base,
                   size_t size, size_t off) {
  if (!shmAudioTraceOn() || !shmAudioMatch(name))
    return;
  BASE_LOGI("shmaudio",
            "t={} tid={} {:<9} '{}' base={:p} size={:#x} off={:#x}",
            (unsigned long long)shmAudioNowUs(), (long)gettid(), op,
            name.c_str(), base, size, off);
}

std::string shmAudioSanitize(const std::string &n) {
  std::string s;
  for (char c : n)
    s += (c == '/' || c == '\\') ? '_' : c;
  return s;
}

// DELTA_SHM_AUDIO_POISON=<byte>: fill matching regions (filter default "_A") on
// first map; silence is otherwise indistinguishable from untouched.
bool shmAudioPoisonQuiet(const std::string &name, u8 *base, size_t size) {
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

void shmAudioPoison(const std::string &name, u8 *base, size_t size) {
  if (shmAudioPoisonQuiet(name, base, size))
    BASE_LOGI("shmaudio", "poisoned '{}' {:p} +{:#x}", name.c_str(), base,
              size);
}

void shmAudioRepoison(const std::string &name, u8 *base, size_t size) {
  if (!kShmRepoison)
    return;
  shmAudioPoisonQuiet(name, base, size);
}

std::atomic<bool> g_shmAudioDumper{false};

void shmAudioDumperMain(std::string dir, unsigned periodMs, unsigned maxSnaps,
                        size_t maxBytes, bool deltaOnly) {
  std::unordered_map<std::string, FILE *> files;
  std::unordered_map<std::string, std::vector<u8>> prev;
  FILE *idx = std::fopen((dir + "/index.txt").c_str(), "w");
  struct reg { std::string n; u8 *b; size_t sz; };
  std::vector<u8> cur;
  for (unsigned seq = 0; seq < maxSnaps; seq++) {
    std::vector<reg> regs;
    {
      std::lock_guard<std::mutex> lk(g_shmMutex);
      for (auto &kv : g_shmByName)
        if (kv.second && kv.second->base && kv.second->size &&
            shmAudioMatch(kv.first))
          regs.push_back({kv.first, kv.second->base, kv.second->size});
    }
    const u64 t = shmAudioNowUs();
    for (auto &r : regs) {
      FILE *&f = files[r.n];
      if (!f)
        f = std::fopen((dir + "/" + shmAudioSanitize(r.n) + ".bin").c_str(), "wb");
      if (!f)
        continue;
      const size_t len = r.sz < maxBytes ? r.sz : maxBytes;
      // Racy by construction: tearing shows up as torn samples, never a wrong
      // cursor value.
      cur.assign(r.b, r.b + len);
      auto &p = prev[r.n];
      const bool same = (p.size() == len) &&
                        std::memcmp(p.data(), cur.data(), len) == 0;
      // Per-tick fingerprint (non-zero count + FNV-1a) so a long run can be
      // judged without keeping its bytes.
      size_t nz = 0;
      u64 h = 1469598103934665603ull;
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
      // Repoison after sampling so the next snapshot shows exactly this tick's
      // writes; destroys the contents.
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
  BASE_LOGI("shmaudio", "dumper finished");
}

// ---------------------------------------------------------------------------
// DELTA_SHM_AUDIO_PROBE=<us>: SPEC VALIDATION HARNESS, default OFF. Plays
// nothing; performs the consumer half of the LLE libSceAudioOut handshake:
// walk the 26 port slots of "/shm_<pid>_C", read a block from the matching
// "_A" region, compute a peak, zero +0x00 to release the port. Pair with
// DELTA_AUDIOMIX_ACK. Correct layout => peaks track the HLE reference.
// ---------------------------------------------------------------------------
void shmAudioProbeMain(long periodUs) {
  constexpr size_t kHdr = 0x20, kStride = 0x250, kSlots = 26;
  u64 ticks = 0, blocks = 0;
  float peakMax = 0.f;
  auto last = shmAudioNowUs();
  for (;;) {
    ::usleep((useconds_t)periodUs);
    ticks++;
    u8 *ctlRaw = nullptr;
    size_t ctlSize = 0;
    struct areg { int idx; u8 *b; size_t sz; };
    std::vector<areg> as;
    {
      std::lock_guard<std::mutex> lk(g_shmMutex);
      for (auto &kv : g_shmByName) {
        if (!kv.second || !kv.second->base) continue;
        const std::string &n = kv.first;
        if (n.size() > 2 && n.compare(n.size() - 2, 2, "_C") == 0 &&
            n.compare(0, 5, "/shm_") == 0) {
          ctlRaw = kv.second->base;
          ctlSize = kv.second->size;
        } else if (n.size() > 2 && n.compare(n.size() - 2, 2, "_A") == 0 &&
                   n.compare(0, 5, "/shm_") == 0) {
          // "/shm_<pid>_<idx>_A" -> idx
          size_t e = n.size() - 2;            // at the '_' of "_A"
          size_t s = n.rfind('_', e - 1);
          if (s != std::string::npos)
            as.push_back({std::atoi(n.c_str() + s + 1), kv.second->base,
                          kv.second->size});
        }
      }
    }
    if (!ctlRaw || ctlSize < kHdr + kSlots * kStride)
      continue;
    for (size_t k = 0; k < kSlots; k++) {
      u8 *slot = ctlRaw + kHdr + k * kStride;
      auto word = [&](size_t o) { return *reinterpret_cast<volatile u32 *>(slot + o); };
      const u32 bpf = word(0x08), rate = word(0x34), grain = word(0x60);
      const u32 stype = word(0x2c), state = word(0x90);
      if (!bpf || !grain)
        continue;                       // slot never opened
      if (!word(0x00))
        continue;                       // no block pending
      u8 *src = nullptr;
      size_t srcSz = 0;
      for (auto &a : as)
        if (a.idx == (int)k) { src = a.b; srcSz = a.sz; }
      const size_t need = (size_t)grain * bpf;
      float peak = 0.f;
      if (src && srcSz >= need) {
        const u32 bps = (stype == 0) ? 2u : 4u;
        const u32 n = need / bps;
        if (stype == 1) {
          const float *p = reinterpret_cast<const float *>(src);
          for (u32 i = 0; i < n; i++) {
            float a = p[i] < 0 ? -p[i] : p[i];
            if (a > peak) peak = a;
          }
        } else if (stype == 0) {
          const i16 *p = reinterpret_cast<const i16 *>(src);
          for (u32 i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -p[i] : p[i]) / 32768.f;
            if (a > peak) peak = a;
          }
        } else {
          const i32 *p = reinterpret_cast<const i32 *>(src);
          for (u32 i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -(float)p[i] : (float)p[i]) / 2147483648.f;
            if (a > peak) peak = a;
          }
        }
      }
      blocks++;
      if (peak > peakMax) peakMax = peak;
      const u64 now = shmAudioNowUs();
      if (now - last > 2000000) {
        last = now;
        BASE_LOGI("shmprobe",
                  "port={} bpf={} type={} ch={} rate={} grain={} state={} "
                  "blocks={} peak={:.4f} peakMax={:.4f}",
                  k, bpf, stype, bpf / (stype == 0 ? 2u : 4u), rate, grain,
                  state, (unsigned long long)blocks, peak, peakMax);
      }
      // Release the port: the guest's next submit is gated on this being 0.
      *reinterpret_cast<volatile u64 *>(slot + 0x00) = 0;
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
  BASE_LOGI("shmprobe", "consumer probe every {}us", us);
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
  BASE_LOGI("shmaudio", "dumper -> {} every {}ms x{} (<={:#x} B){}", dir, ms,
            n, mx, kShmAudioDumpDelta ? " delta-only" : "");
  std::thread(shmAudioDumperMain, std::string(dir), ms, n, mx,
              kShmAudioDumpDelta.get()).detach();
}

class shmObject : public kObject {
public:
  shmObject(objectTable &objects, std::string nm, shmRef b)
      : kObject(objects, kObject::oType::shm), shmName(std::move(nm)),
        backing(std::move(b)) {}
  std::string shmName;  // diagnostics / audio protocol key
  shmRef backing;       // keeps the backing alive while this fd is open
};

// Backing block for a shm, grown to cover the requested range. Caller must not
// hold g_shmMutex; -1 on failure.
u8 *shmMap(shmObject *shm, size_t size, size_t offset) {
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = *shm->backing;
  size_t need = (offset + size + 0x3FFF) & ~size_t(0x3FFF);
  if (!b.base && need) {
    b.base = allocLowGuest(need);
    if (!b.base)
      return reinterpret_cast<u8 *>(-1);
    b.size = need;
    proc::getActive()->getVma().add(b.base, need, ppt::w);
  }
  if (!b.base || offset > b.size)
    return reinterpret_cast<u8 *>(-1);
  shmAudioTrace("mmap", shm->shmName, b.base + offset, size, offset);
  // Announce here too: a region mmap'd without ftruncate allocates above.
  audioDaemonNoticeShm(shm->shmName.c_str(), b.base, b.size);
  return b.base + offset;
}
}  // namespace

u8 *PS4ABI sys_mmap(void *addr, size_t size, u32 prot, u32 flags,
                         u32 fd, size_t offset) {
  auto *proc = proc::getActive();
  if (!proc)
    return reinterpret_cast<u8 *>(-1);

  if (flags & mFlags::stack || flags & mFlags::noextend)
    flags |= mFlags::anon;

  /*align the page*/
  size = (size + 0x3FFF) & 0xFFFFFFFFFFFFC000LL;

  // A zero-length mapping is invalid (BSD EINVAL); guests hit it on
  // error-recovery paths (mmap of an fd from a failed physhm_open/fstat).
  if (size == 0)
    return reinterpret_cast<u8 *>(-SysError::eINVAL);

  // DELTA_MMAP_CALLER=<minMB>: scan the guest stack for module .text return
  // addresses to pin which guest code requested a big map.
  if (const char *mc = kMmapCaller) {
    size_t minB = static_cast<size_t>(std::strtoull(mc, nullptr, 0)) * 1024 * 1024;
    if (minB == 0) minB = 64ull * 1024 * 1024;
    if (size >= minB) {
      BASE_LOGI("mmap-caller", "size={:#x} prot={:#x} flags={:#x} fd={}:", size,
                prot, flags, fd);
      auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
      int shown = 0;
      for (int i = 0; i < 8192 && shown < 12; i++) {
        uintptr_t v = sp[i];
        for (auto &m : proc->getModuleList()) {
          auto &mi = m->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && v >= (uintptr_t)t && v < (uintptr_t)t + mi.textSeg.size) {
            BASE_LOGI("mmap-caller", "  sp+{:<4x} {}+{:#x}", i * 8,
                      mi.name.c_str(), v - (uintptr_t)t);
            shown++;
            break;
          }
        }
      }
    }
  }

  // Faithful validation, mirroring the kernel: MAP_STACK needs an anon fd + r|w
  // and drops the offset; MAP_VOID forces prot 0; MAP_FIXED must be page-aligned
  // and not wrap. (Unbounded by VM_MAXUSER_ADDRESS: libkernel fixed-maps a host
  // stack guard far above the 2^40 guest ceiling.)
  if (flags & mFlags::stack) {
    if (fd != static_cast<u32>(-1) || (prot & 3) != 3)
      return reinterpret_cast<u8 *>(-SysError::eINVAL);
    offset = 0;
  }
  const bool voidReserve = (flags & 0x100) != 0;
  if (voidReserve)
    prot = 0;
  if (flags & mFlags::fixed) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
    if ((a & 0x3FFF) != 0)
      return reinterpret_cast<u8 *>(-SysError::eINVAL);
    uintptr_t aEnd;
    if (__builtin_add_overflow(a, size, &aEnd))
      return reinterpret_cast<u8 *>(-SysError::eINVAL);
  }

  // addr is a hint unless MAP_FIXED: relocate it rather than alias an existing map
  if (!(flags & mFlags::fixed)) {
    if (!addr || proc->getVma().overlaps(static_cast<u8 *>(addr), size))
      addr = nullptr;
  }

  // A hint inside the guest's user-stack region (or a VA range
  // reserveGuestVaSpace() claimed) is guest-owned: our PROT_NONE placeholder
  // would relocate it and exhaust libkernel's internal arena.
  bool inUserStack = false;
  if (addr) {
    auto &env = proc->getEnv();
    auto a = reinterpret_cast<uintptr_t>(addr);
    auto lo = reinterpret_cast<uintptr_t>(env.userStack);
    inUserStack = env.userStack && a >= lo && a + size <= lo + env.userStackSize;
    inUserStack = inUserStack || isGuestReservedVa(addr, size);
  }

  // A /dev/dmem mmap carries the direct-memory physical offset in `offset`.
  bool dmemMap = false;
  if (fd != -1) {
    auto *obj = proc->getObjTable().get(fd);
    if (obj && obj->type() == kObject::oType::device &&
        static_cast<device *>(obj)->isDirectMemory())
      dmemMap = true;
    if (kMmapfdTrace)
      BASE_LOGI("mmapfd", "fd={} addr={:p} size={:#x} off={:#x} objType={}",
                fd, addr, size, offset, obj ? (int)obj->type() : -1);
    if (obj && obj->type() == kObject::oType::shm) {
      // Every mapper of this shm shares the backing (sized by ftruncate).
      return shmMap(static_cast<shmObject *>(obj), size, offset);
    }
    if (obj) {
      // Device-backed mmap: use the region the device hands back; -1 = fall through.
      auto *m = static_cast<device *>(obj)->map(addr, size, prot, flags, offset);
      if (m != reinterpret_cast<u8 *>(-1)) {
        proc->getVma().add(
            m, size, static_cast<ppt>(prot & static_cast<u32>(ppt::rwx)),
            prot);
        return m;
      }
    }
  }

  // MAP_ALIGNED(n): flags bits 31..24 carry log2 base alignment the kernel must
  // honor (SotC keys streaming-arena bookkeeping on it).
  const u32 alignLog = (flags >> 24) & 0x1F;
  const size_t mapAlign = (alignLog >= 14 && alignLog < 40)
                              ? (size_t(1) << alignLog)
                              : 0;
  if (mapAlign && addr && !(flags & mFlags::fixed) &&
      (reinterpret_cast<uintptr_t>(addr) & (mapAlign - 1)))
    addr = nullptr;  // misaligned hint: pick our own aligned base instead

  // Keep prot-0 reservations out of the band titles MAP_FIXED their pools into:
  // Minecraft's direct-memory MAP_FIXED would silently replace the first page
  // of V8's pointer cage (and its read-only heap with it).
  if (addr && !(flags & mFlags::fixed) && prot == 0 &&
      reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ull &&
      reinterpret_cast<uintptr_t>(addr) < kGuestArenaFloor)
    addr = nullptr;

  void *ptr = nullptr;
  if (addr) {
    if (flags & mFlags::fixed) {
      // MAP_FIXED stack maps [start-size, start): plant below the hint.
      void *want = addr;
      if (flags & mFlags::stack)
        want = static_cast<u8 *>(addr) - size;
      ptr = utl::allocMem(want, size, ppt::w, alt::reservecommit);
      if (!ptr)
        ptr = utl::allocMem(want, size, ppt::w, alt::commit);  // maybe pre-reserved
    } else if (inUserStack) {
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);
    } else if (utl::allocMem(addr, size, ppt::w, alt::reserve)) {
      // A hint must never alias an existing mapping; reservecommit would clobber
      // it (a guest TLS hint destroyed a loaded module on Android).
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);
    } else if (wasGuestReleased(static_cast<u8 *>(addr), size) &&
               !proc->getVma().overlaps(static_cast<u8 *>(addr), size)) {
      // The probe only fails here because we kept the pages of a guest-unmapped
      // range; the address is free as far as the guest is concerned.
      ptr = utl::allocMem(addr, size, ppt::w, alt::reservecommit);
    }
  }
  // No usable hint (or it was taken): pick a low (<2^40) address the guest's
  // own allocators will accept, not whatever high address the host hands out.
  if (!ptr)
    ptr = allocLowGuest(size, mapAlign);
  if (!ptr)
    return reinterpret_cast<u8 *>(-SysError::eNOMEM);

  // Track the guest-requested prot so VirtualQuery reports the truth; host pages
  // stay rwx (FEX reads guest memory directly, no protection faults delivered).
  auto gprot = static_cast<ppt>(prot & static_cast<u32>(ppt::rwx));

  // No zero-fill: every path above already returns kernel-zeroed anonymous
  // pages; self-fill faulted in multi-GiB pools (Minecraft, 43 GiB RSS).

  // File-backed mmap: copy the file's content in (Doom64 mmaps its WADs and
  // samples textures from the mapping); the read stops at EOF so an oversized
  // mapping keeps zeros past the file's end.
  if (fd != static_cast<u32>(-1)) {
    if (auto *o = proc->getObjTable().get(fd))
      if (o->type() == kObject::oType::device) {
        // The kernel hands back base + (offset & 0x3FFF), so the fill starts at
        // the page-aligned offset for that contract to hold.
        const i64 fileOff = static_cast<i64>(offset & ~size_t(0x3FFF));
        i64 got = static_cast<device *>(o)->readAt(ptr, size, fileOff);
        if (got > 0 && kMmapfdTrace)
          BASE_LOGI("mmapfd", "  filled {:p} from fd={} off={:#x} -> {} bytes",
                    ptr, fd, (unsigned long long)fileOff, (long long)got);
      }
  }

  // MAP_VOID is a reservation (later MAP_FIXED commits punch it apart in the
  // VMA); virtual query must see it as reserved, not committed.
  if (dmemMap)
    proc->getVma().addDirect(static_cast<u8 *>(ptr), size, gprot, prot,
                             offset);
  else
    proc->getVma().add(static_cast<u8 *>(ptr), size, gprot, prot,
                       voidReserve);

  utl::protectMem(static_cast<void *>(ptr), size, ppt::rwx);

  // DELTA_GNMAP_TRACE: log mappings in the GNM/GPU aperture to pin how the AGC
  // ring buffers are mapped and by whom (coherency with guest PM4 writes is
  // an open item).
  if (kGnmapTrace) {
    u64 r = reinterpret_cast<u64>(ptr);
    if (r >= 0x8000000000ull && r < 0x8300000000ull) {
      // Walk the guest frame chain (libkernel keeps frame pointers) for the
      // real caller above libkernel's mmap wrapper.
      base::String callers;
      base::FormatTo(callers, "{:p} size={:#x} fd={} flags={:#x}  callers:",
                     ptr, size, static_cast<int>(fd), flags);
      void *fp = __builtin_frame_address(0);
      for (int depth = 0; depth < 20 && fp; depth++) {
        auto *frame = static_cast<void **>(fp);
        void *ret = frame[1];
        if (!ret) break;
        char sym[160];
        krnl::symbolize(reinterpret_cast<uintptr_t>(ret), sym, sizeof(sym));
        base::FormatTo(callers, " [{}]", sym);
        fp = frame[0];
        if (reinterpret_cast<uintptr_t>(fp) < 0x10000) break;
      }
      BASE_LOGI("gnmap", "{}", callers.c_str());
    }
  }

  if (kMmapLog)
    BASE_LOGI("mmap",
              "mmap {:p}, {:x}, prot={:x} flags={:x} fd={} off={:#x}, {:p} -> "
              "{:p}",
              addr, size, prot, flags, static_cast<int>(fd), offset,
              _ReturnAddress(), ptr);

  // Returned address = base + (offset & 0x3FFF), FreeBSD mmap semantics;
  // MAP_STACK returns the region top, like vm_map_stack.
  if (flags & mFlags::stack)
    return &static_cast<u8 *>(ptr)[size];

  return static_cast<u8 *>(ptr) + (offset & 0x3FFF);
}

int PS4ABI sys_mprotect(u8 *addr, size_t len, int prot) {
  auto *proc = proc::getActive();
  if (!proc)
    return -SysError::eINVAL;

  // Same range math as the kernel's sys_mprotect: round the base down and the
  // span up, adding the page offset of `addr`; a wrapping range is EINVAL.
  const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  const uintptr_t base = a & ~uintptr_t(0x3FFF);
  const size_t span = (len + (a & 0x3FFF) + 0x3FFF) & ~uintptr_t(0x3FFF);
  uintptr_t end;
  if (__builtin_add_overflow(base, span, &end))
    return -SysError::eINVAL;

  // The kernel masks prot to 0x37 and applies it to every entry in the range.
  // We don't restrict host pages and don't fail over gaps (the linker mprotects
  // its own RELRO segments outside this table); update what we know, succeed.
  const u32 sceProt = static_cast<u32>(prot) & 0x37;
  proc->getVma().protectRange(reinterpret_cast<u8 *>(base), span,
                              static_cast<ppt>(sceProt &
                                               static_cast<u32>(ppt::rwx)),
                              sceProt);
  return 0;
}

// System services we do not host: a shm + named semaphore pair sharing the
// service's name. The list grows one name at a time as titles walk into the next.
bool isAbsentServiceChannel(const char *name) {
  static constexpr const char *kServices[] = {"SceNpTpip"};
  if (!name)
    return false;
  if (*name == '/')
    name++;
  const size_t n = std::strcspn(name, " ");  // the semaphore half is "<svc> <n>"
  for (const char *svc : kServices)
    if (std::strlen(svc) == n && std::strncmp(name, svc, n) == 0)
      return true;
  return false;
}

int PS4ABI sys_shm_open(const char *path, u32 flags, u16 mode) {
  auto *proc = proc::getActive();
  if (!proc || !path)
    return -SysError::eINVAL;

  // From the kernel's sys_shm_open: O_WRONLY is invalid, unknown bits among
  // {O_RDWR, O_CREAT, O_TRUNC, O_EXCL} are EINVAL. Only the low half is checked:
  // libkernel passes high descriptor flags the console accepts.
  if ((flags & 1) != 0)
    return -SysError::eINVAL;
  if ((flags & 0xF1FC) != 0)
    return -SysError::eINVAL;

  constexpr u32 kO_CREAT = 0x0200, kO_EXCL = 0x0800, kO_TRUNC = 0x0400;
  std::string name(path);
  shmRef backing;
  {
    std::lock_guard<std::mutex> lk(g_shmMutex);
    auto it = g_shmByName.find(name);
    if (it == g_shmByName.end()) {
      if (!(flags & kO_CREAT) && kShmNoAuto) {
        // DIAGNOSTIC: fail opens of system shms the guest didn't create
        // (tests whether auto-providing masks a missing ShellCore handshake).
        BASE_LOGI("shm_open", "NOAUTO: '{}' -> ENOENT", name.c_str());
        return -SysError::eNOENT;
      }
      if (!(flags & kO_CREAT)) {
        // Read-only open of a system shm (e.g. libSceAvSetting's settings
        // block): auto-provide a zeroed backing so the title fstats a real
        // size instead of failing init.
        backing = std::make_shared<shmBacking>();
        backing->size = 0x10000;  // 64 KiB, ample for a settings block
        backing->base = allocLowGuest(backing->size);
        if (backing->base) {
          std::memset(backing->base, 0, backing->size);
          proc->getVma().add(backing->base, backing->size, ppt::w);
        }
        BASE_LOGI("shm_open", "auto-provide system shm '{}' size={:#x}",
                  name.c_str(), backing->size);
        g_shmByName.emplace(name, backing);
      } else {
        // O_CREAT: fresh, empty backing; sized later by ftruncate.
        backing = std::make_shared<shmBacking>();
        g_shmByName.emplace(name, backing);
      }
    } else {
      if ((flags & kO_CREAT) && (flags & kO_EXCL))
        return -SysError::eEXIST;
      backing = it->second;
      // O_TRUNC|O_RDWR (0x402) truncates to zero; existing fds keep mapping the
      // empty backing, exactly like the real object.
      if ((flags & 0x403) == 0x402) {
        shmAudioTrace("trunc", name, backing->base, 0, 0);
        backing->size = 0;
        if (backing->base)
          audioDaemonNoticeShm(name.c_str(), backing->base, 0);
      }
    }
  }

  // A fresh fd per open, all sharing the named backing.
  auto *obj = new shmObject(proc->getObjTable(), std::move(name), std::move(backing));
  BASE_LOGI("shm_open", "'{}' flags={:#x} -> fd={}", path, flags,
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
  // Drop the name only; open fds hold shared refs to the backing, like the
  // kernel refcounting the shm object.
  g_shmByName.erase(it);
  return 0;
}

// Report a shm fd's backing size for fstat (shm objects aren't device-backed,
// so sys_fstat's fdToDevice path can't size them). Returns SIZE_MAX if `fd`
// isn't a shm, so the caller falls through to the normal path.
size_t shmFstatSize(u32 fd) {
  auto *proc = proc::getActive();
  if (!proc)
    return SIZE_MAX;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::shm)
    return SIZE_MAX;
  auto *shm = static_cast<shmObject *>(obj);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  return shm->backing->size;
}

int PS4ABI sys_ftruncate(u32 fd, i64 length) {
  auto *proc = proc::getActive();
  if (!proc || length < 0)
    return -SysError::eINVAL;
  auto *obj = proc->getObjTable().get(fd);
  if (!obj)
    return -SysError::eBADF;
  // The kernel dispatches to the file type's truncate method; a type without
  // one (e.g. a device) is EINVAL, not EBADF.
  if (obj->type() != kObject::oType::shm)
    return -SysError::eINVAL;

  auto *shm = static_cast<shmObject *>(obj);
  const size_t raw = static_cast<size_t>(length);
  const size_t want = (raw + 0x3FFF) & ~size_t(0x3FFF);
  std::lock_guard<std::mutex> lk(g_shmMutex);
  auto &b = *shm->backing;
  // The RAW length matters for the protocol spec (the rounded `want` hides it).
  shmAudioTrace("ftruncate", shm->shmName, b.base, raw, want);
  if (want < b.size) {
    // Shrink (the kernel frees the tail pages); the host block stays put, a
    // later grow reallocates and copies `size` bytes.
    b.size = want;
    audioDaemonNoticeShm(shm->shmName.c_str(), b.base, b.size);
    return 0;
  }
  if (want == b.size)
    return 0;
  u8 *nb = allocLowGuest(want);
  if (!nb)
    return -SysError::eNOMEM;
  if (b.base)
    std::memcpy(nb, b.base, b.size);  // grow before first mmap: preserve contents
  b.base = nb;
  b.size = want;
  proc->getVma().add(b.base, want, ppt::w);
  shmAudioTrace("sized", shm->shmName, b.base, want, 0);
  shmAudioPoison(shm->shmName, b.base, want);
  // The LLE audio regions become consumable here; the base can MOVE on a later
  // grow, hence the fresh pointer each time.
  audioDaemonNoticeShm(shm->shmName.c_str(), b.base, b.size);
  return 0;
}

int PS4ABI sys_mname(u8 *ptr, size_t len, const char *name, void *) {
  auto *proc = proc::getActive();
  if (!proc)
    return -SysError::eINVAL;

  // Kernel guards: the range lives inside user VA space and the name fits the
  // 32-byte tag buffer.
  const uintptr_t a = reinterpret_cast<uintptr_t>(ptr);
  if (a >= 0x10000000000ull || len > 0x10000000000ull - a)
    return -SysError::eINVAL;
  char tag[32];
  const size_t n = strnlen(name, sizeof(tag));
  if (n == sizeof(tag))
    return -SysError::eNAMETOOLONG;
  memcpy(tag, name, n);
  tag[n] = '\0';

  // Tag every entry the range covers, mirroring vm_map_set_name.
  proc->getVma().setRangeName(ptr, len, tag);
  // Titles name thread stacks this way; carry the tag onto the host thread
  // running on that stack (attribution).
  nameThreadsForRange(ptr, len, tag);
  return 0;
}

struct mdbg_property {
  i32 unk;
  i32 unk2;
  void *addr;
  size_t areaSize;
  i64 unk3;
  i64 unk4;
  char name[32];
};

static_assert(sizeof(mdbg_property) == 72);

// Stand-in for the kernel's per-process debug-raise qword (proc+2600); no
// debugger is attached, so a raise is recorded and reported delivered.
static std::atomic<u64> gMdbgFlags{0};

// Mirrors mdbg_service_raise; the suspend notification callback is
// unregistered, so report the raise as delivered.
static int mdbgServiceRaise(uintptr_t reason) {
  if (reason <= 0x7E)
    gMdbgFlags.fetch_or(static_cast<u64>(reason) << 32);
  gMdbgFlags.fetch_or(2);
  LOG_WARNING("mdbg raise: reason={} (no debugger attached, ignoring)", reason);
  return 0;
}

int PS4ABI sys_mdbg_service(u32 op, void *arg1, void *arg2, void *a3) {
  (void)arg2;
  (void)a3;
  switch (op) {
  case 0:
    // Kernel: copyout the proc's debug flags qword (proc+2600) to the caller.
    *static_cast<u64 *>(arg1) = gMdbgFlags.load();
    return 0;
  case 1: {
    // Copyin a 72-byte property and register a named object (a no-op here); log it.
    auto *info = static_cast<mdbg_property *>(arg1);
    char tag[32];
    tag[31] = 0; // kernel zeroes the last name byte after copyin
    memcpy(tag, info->name, sizeof(tag) - 1);
    LOG_WARNING("mdbg property {} for addr {} size {}", tag, info->addr,
                info->areaSize);
    return 0;
  }
  case 3:
    // Kernel: mdbg_service_raise with the raw second syscall argument as reason.
    return mdbgServiceRaise(reinterpret_cast<uintptr_t>(arg1));
  case 4:
    // Raise only while debug mode is allowed; both conditions hold for us.
    return mdbgServiceRaise(reinterpret_cast<uintptr_t>(arg1));
  case 7: {
    // The mdbg text facility: log the message.
    const char *msg = static_cast<const char *>(arg1);
    if (!msg)
      return -SysError::eINVAL;
    std::string s(msg, strnlen(msg, 0x1000));
    LOG_INFO("[mdbg-text] {}", s);
    return 0;
  }
  case 8:
    // libkernel's debug thread treats any error as "event arrived" and runs the
    // title's coredump callback on an UNINITIALIZED buffer (Demon's Souls faults
    // at boot). No debugger ever posts here; park the caller like the kernel.
    LOG_INFO("mdbg wait-event: parking caller (no debugger attached)");
    for (;;)
      ::pause();
  default:
    // The kernel returns 78 (eNOSYS in our table) for unknown ops.
    return -SysError::eNOSYS;
  }
}

// sys_dmem_container (586): 0xFFFFFFFF reads the container id, 0/1 sets it
// (needs privilege 0x2AD; the kernel keeps it at proc+2020). We only track it.
int PS4ABI sys_dmem_container(u32 op) {
  static std::atomic<u32> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  if (op > 1)
    return -SysError::eINVAL;
  current.store(op);
  return 0;
}
} // namespace krnl
