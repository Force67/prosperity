/*
 * PS4Delta: PS4 emulation and research project
 * Copyright 2019-2020 Force67. See LICENSE in the root of the source tree.
 */

#include <cstdio>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"
#include "logger/logger.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/crash.h"
#include "kern/guest_va_space.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/stub_log.h"
#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "kern/ps4/audio_daemon.h"
#include "kern/ps4/dev/dma_dev.h"  // DmemBackingFd/Size (shared physical dmem store)
#include "kern/ps5/dev/dma_dev.h"
#include "kern/thread_names.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kBlockpoolTrace, "DELTA_BLOCKPOOL_TRACE", false);
DELTA_OPTION(const char*, kShmFilter, "DELTA_SHM_AUDIO_FILTER", nullptr);
DELTA_OPTION(const char*, kShmPoison, "DELTA_SHM_AUDIO_POISON", nullptr);
DELTA_OPTION(const char*,
             kShmPoisonFilter,
             "DELTA_SHM_AUDIO_POISON_FILTER",
             nullptr);
DELTA_OPTION(bool, kShmRepoison, "DELTA_SHM_AUDIO_REPOISON", false);
DELTA_OPTION(const char*, kShmProbe, "DELTA_SHM_AUDIO_PROBE", nullptr);
DELTA_OPTION(const char*, kShmDump, "DELTA_SHM_AUDIO_DUMP", nullptr);
DELTA_OPTION(unsigned, kShmDumpMs, "DELTA_SHM_AUDIO_DUMP_MS", 10);
DELTA_OPTION(unsigned, kShmDumpN, "DELTA_SHM_AUDIO_DUMP_N", 400);
DELTA_OPTION(size_t, kShmDumpMax, "DELTA_SHM_AUDIO_DUMP_MAX", 1u << 20);
DELTA_OPTION(const char*, kMmapCaller, "DELTA_MMAP_CALLER", nullptr);
DELTA_OPTION(bool, kShmNoAuto, "DELTA_SHM_NOAUTO", false);
DELTA_OPTION(bool, kAllocTrace, "DELTA_ALLOC_TRACE", false);
DELTA_OPTION(bool, kGnmapTrace, "DELTA_GNMAP_TRACE", false);
DELTA_OPTION(bool, kMmapLog, "DELTA_MMAP_LOG", false);
DELTA_OPTION(bool, kMmapfdTrace, "DELTA_MMAPFD_TRACE", false);
DELTA_OPTION(bool, kShmAudioDumpDelta, "DELTA_SHM_AUDIO_DUMP_DELTA", false);
DELTA_OPTION(bool, kShmAudioTrace, "DELTA_SHM_AUDIO_TRACE", false);
}  // namespace

namespace kern {

using Ppt = host_memory::PageProtection;
using Alt = host_memory::AllocationType;

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
base::Mutex g_released_lock;
base::Vector<ReleasedRange> g_released;
}  // namespace

void NoteGuestReleased(u8* ptr, size_t size) {
  if (!ptr || !size)
    return;
  host_memory::ForgetMemoryMapping(ptr, size);
  const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  base::LockGuard<base::Mutex> lk(g_released_lock);
  for (auto& r : g_released) {
    if (base <= r.end && base + size >= r.base) {  // merge touching ranges
      r.base = base::Min(r.base, base);
      r.end = base::Max(r.end, base + size);
      return;
    }
  }
  if (g_released.size() < 4096)
    g_released.push_back({base, base + size});
}

void NoteGuestTaken(u8* ptr, size_t size) {
  if (!ptr || !size)
    return;
  host_memory::ForgetMemoryMapping(ptr, size);
  const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr), hi = lo + size;
  base::LockGuard<base::Mutex> lk(g_released_lock);
  for (auto it = g_released.begin(); it != g_released.end();) {
    if (lo < it->end && it->base < hi)
      g_released.erase(it);  // partially reused: no longer safe to reuse
    else
      ++it;
  }
}

bool WasGuestReleased(u8* ptr, size_t size) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
  base::LockGuard<base::Mutex> lk(g_released_lock);
  for (const auto& r : g_released)
    if (base >= r.base && base + size <= r.end)
      return true;
  return false;
}

// PS4 user-space pointers must live below 2^40: libc's sceLibcMspaceCreate (and
// other allocators) reject a base whose bits >= 40 are set. The host kernel
// hands mmap(NULL) addresses far above that ceiling, so when we have to pick an
// address ourselves, carve it from a dedicated low arena instead. Bump-only and
// MAP_FIXED_NOREPLACE so we never clobber an existing mapping.
namespace {
base::Atomic<uintptr_t> g_low_next{kGuestArenaFloor};
}

u8* AllocLowGuest(size_t size, size_t align) {
#ifdef __ANDROID__
  // 39-bit user VA: keep the guest arena clear of the module region (32..~224
  // GiB) and the FEX heap / bionic up top, and still under the PS4 2^40
  // ceiling.
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 256 GiB
  constexpr uintptr_t kCeil = 0x6000000000ull;    // 384 GiB
#else
  // Start at 512 GiB: titles fixed-map pools at round 64 GiB slots (GTA:SA put
  // a 128 MB pool exactly on the old 256 GiB floor, clobbering the primary
  // TCB).
  constexpr uintptr_t kFloor = kGuestArenaFloor;  // 512 GiB
  constexpr uintptr_t kCeil = 0x10000000000ull;   // 2^40, the PS4 user ceiling
#endif
  // 64 KiB, not just one page: GNM surfaces sub-allocated from a pool base
  // assert on weaker alignment (DOOM rhiTextureGnm).
  constexpr uintptr_t kAlign = 0x10000;

  size = (size + 0x3FFF) & ~uintptr_t(0x3FFF);
  // MAP_ALIGNED(n) is contractual: SotC indexes its streaming arenas by VA>>20,
  // so a weaker base breaks every lookup.
  const uintptr_t al = align > kAlign ? align : kAlign;
  for (int tries = 0; tries < 8192; tries++) {
    uintptr_t raw = g_low_next.load(base::memory_order_relaxed);
    uintptr_t base = (raw + (al - 1)) & ~(al - 1);  // align the base up
    if (base + size + 0x4000 > kCeil)
      return nullptr;  // doesn't fit; do NOT poison `next` (CAS, not fetch_add)
    if (!g_low_next.compare_exchange_weak(raw, base + size + 0x4000,
                                          base::memory_order_relaxed))
      continue;  // another thread advanced it; reload and retry
    void* p =
        ::mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == reinterpret_cast<void*>(base)) {
      host_memory::TrackMemoryMapping(p, size);
      if (kAllocTrace)
        BASE_LOGI("lowalloc", "{:#x} +{:#x}", (unsigned long)base,
                  (unsigned long)size);
      return static_cast<u8*>(p);
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
struct ShmBacking {
  u8* base = nullptr;
  size_t size = 0;
};
using ShmRef = base::SharedPointer<ShmBacking>;
base::Mutex g_shm_mutex;
// name -> backing. The backing outlives the name (shmObject holds a shared
// ref), like the kernel refcounting the shm object by fd.
base::HashMap<base::String, ShmRef> g_shm_by_name;

// ---------------------------------------------------------------------------
// RESEARCH INSTRUMENTATION (env-gated, default OFF), for reverse-engineering
// the LLE libSceAudioOut shared-memory mixer protocol (see the note at the
// head of runtime/vprx/ps4/libSceAudioOut):
//   DELTA_SHM_AUDIO_TRACE=1        log shm_open/ftruncate/mmap for matching
//   names DELTA_SHM_AUDIO_FILTER=<sub>   name substring to match (default
//   "shm_") DELTA_SHM_AUDIO_DUMP=<dir>     periodically snapshot every matching
//   region DELTA_SHM_AUDIO_DUMP_MS/N/MAX  period (10 ms), count (400), byte cap
//   (1 MiB)
// Snapshots append to <dir>/<name>.bin, indexed in <dir>/index.txt.
// ---------------------------------------------------------------------------
const char* ShmAudioFilter() {
  const char* v = kShmFilter;
  return (v && *v) ? v : "shm_";
}

bool ShmAudioMatch(const base::String& n) {
  return n.find(ShmAudioFilter()) != base::String::npos;
}

bool ShmAudioTraceOn() {
  return kShmAudioTrace;
}

u64 ShmAudioNowUs() {
  return (u64)(base::TickClock::NowNs() / 1000);
}

void ShmAudioTrace(const char* op,
                   const base::String& name,
                   const void* base,
                   size_t size,
                   size_t off) {
  if (!ShmAudioTraceOn() || !ShmAudioMatch(name))
    return;
  BASE_LOGI("shmaudio", "t={} tid={} {:<9} '{}' base={:p} size={:#x} off={:#x}",
            (unsigned long long)ShmAudioNowUs(), (long)gettid(), op,
            name.c_str(), base, size, off);
}

base::String ShmAudioSanitize(const base::String& n) {
  base::String s;
  for (char c : n)
    s += (c == '/' || c == '\\') ? '_' : c;
  return s;
}

// DELTA_SHM_AUDIO_POISON=<byte>: fill matching regions (filter default "_A") on
// first map; silence is otherwise indistinguishable from untouched.
bool ShmAudioPoisonQuiet(const base::String& name, u8* base, size_t size) {
  const char* pv = kShmPoison;
  if (!pv || !base || !size)
    return false;
  const char* pf = (kShmPoisonFilter && *kShmPoisonFilter.get())
                       ? kShmPoisonFilter.get()
                       : "_A";
  if (name.find(pf) == base::String::npos)
    return false;
  std::memset(base, (int)std::strtol(pv, nullptr, 0), size);
  return true;
}

void ShmAudioPoison(const base::String& name, u8* base, size_t size) {
  if (ShmAudioPoisonQuiet(name, base, size))
    BASE_LOGI("shmaudio", "poisoned '{}' {:p} +{:#x}", name.c_str(), base,
              size);
}

void ShmAudioRepoison(const base::String& name, u8* base, size_t size) {
  if (!kShmRepoison)
    return;
  ShmAudioPoisonQuiet(name, base, size);
}

base::Atomic<bool> g_shm_audio_dumper{false};

void ShmAudioDumperMain(base::String dir,
                        unsigned period_ms,
                        unsigned max_snaps,
                        size_t max_bytes,
                        bool delta_only) {
  base::HashMap<base::String, FILE*> files;
  base::HashMap<base::String, base::Vector<u8>> prev;
  FILE* idx = std::fopen((dir + "/index.txt").c_str(), "w");
  struct CloseFiles {
    base::HashMap<base::String, FILE*>& files;
    FILE* index;
    ~CloseFiles() {
      for (auto& [name, file] : files)
        if (file)
          std::fclose(file);
      if (index)
        std::fclose(index);
    }
  } close_files{files, idx};
  struct Reg {
    base::String n;
    u8* b;
    size_t sz;
  };
  base::Vector<u8> cur;
  for (unsigned seq = 0; seq < max_snaps; seq++) {
    base::Vector<Reg> regs;
    {
      base::LockGuard<base::Mutex> lk(g_shm_mutex);
      for (auto& kv : g_shm_by_name)
        if (kv.second && kv.second->base && kv.second->size &&
            ShmAudioMatch(kv.first))
          regs.push_back({kv.first, kv.second->base, kv.second->size});
    }
    const u64 t = ShmAudioNowUs();
    for (auto& r : regs) {
      FILE*& f = files[r.n];
      if (!f)
        f = std::fopen((dir + "/" + ShmAudioSanitize(r.n) + ".bin").c_str(),
                       "wb");
      if (!f)
        continue;
      const size_t len = r.sz < max_bytes ? r.sz : max_bytes;
      // Racy by construction: tearing shows up as torn samples, never a wrong
      // cursor value.
      cur.assign(r.b, r.b + len);
      auto& p = prev[r.n];
      const bool same =
          (p.size() == len) && std::memcmp(p.data(), cur.data(), len) == 0;
      // Per-tick fingerprint (non-zero count + FNV-1a) so a long run can be
      // judged without keeping its bytes.
      size_t nz = 0;
      u64 h = 1469598103934665603ull;
      for (size_t i = 0; i < len; i++) {
        nz += cur[i] != 0;
        h = (h ^ cur[i]) * 1099511628211ull;
      }
      long off = -1;
      if (!delta_only || !same) {
        off = std::ftell(f);
        std::fwrite(cur.data(), 1, len, f);
        p = cur;
      }
      // Repoison after sampling so the next snapshot shows exactly this tick's
      // writes; destroys the contents.
      ShmAudioRepoison(r.n, r.b, len);
      if (idx)
        std::fprintf(idx, "%u %llu %s %ld %zu %zu %016llx\n", seq,
                     (unsigned long long)t, r.n.c_str(), off, len, nz,
                     (unsigned long long)h);
    }
    if (idx)
      std::fflush(idx);
    guest::SleepForMilliseconds(period_ms);
  }
  BASE_LOGI("shmaudio", "dumper finished");
}

// ---------------------------------------------------------------------------
// DELTA_SHM_AUDIO_PROBE=<us>: SPEC VALIDATION HARNESS, default OFF. Plays
// nothing; performs the consumer half of the LLE libSceAudioOut handshake:
// walk the 26 port slots of "/shm_<pid>_C", read a block from the matching
// "_A" region, compute a peak, zero +0x00 to release the port. Pair with
// DELTA_AUDIOMIX_ACK. Correct layout => peaks track the HLE reference.
// ---------------------------------------------------------------------------
void ShmAudioProbeMain(long period_us) {
  constexpr size_t kHdr = 0x20, kStride = 0x250, kSlots = 26;
  u64 ticks = 0, blocks = 0;
  float peak_max = 0.f;
  auto last = ShmAudioNowUs();
  for (;;) {
    guest::SleepForMicroseconds(period_us);
    ticks++;
    u8* ctl_raw = nullptr;
    size_t ctl_size = 0;
    struct Areg {
      int idx;
      u8* b;
      size_t sz;
    };
    base::Vector<Areg> as;
    {
      base::LockGuard<base::Mutex> lk(g_shm_mutex);
      for (auto& kv : g_shm_by_name) {
        if (!kv.second || !kv.second->base)
          continue;
        const base::String& n = kv.first;
        if (n.size() > 2 && n.compare(n.size() - 2, 2, "_C") == 0 &&
            n.compare(0, 5, "/shm_") == 0) {
          ctl_raw = kv.second->base;
          ctl_size = kv.second->size;
        } else if (n.size() > 2 && n.compare(n.size() - 2, 2, "_A") == 0 &&
                   n.compare(0, 5, "/shm_") == 0) {
          // "/shm_<pid>_<idx>_A" -> idx
          size_t e = n.size() - 2;  // at the '_' of "_A"
          size_t s = n.rfind('_', e - 1);
          if (s != base::String::npos)
            as.push_back({std::atoi(n.c_str() + s + 1), kv.second->base,
                          kv.second->size});
        }
      }
    }
    if (!ctl_raw || ctl_size < kHdr + kSlots * kStride)
      continue;
    for (size_t k = 0; k < kSlots; k++) {
      u8* slot = ctl_raw + kHdr + k * kStride;
      auto word = [&](size_t o) {
        return *reinterpret_cast<volatile u32*>(slot + o);
      };
      const u32 bpf = word(0x08), rate = word(0x34), grain = word(0x60);
      const u32 stype = word(0x2c), state = word(0x90);
      if (!bpf || !grain)
        continue;  // slot never opened
      if (!word(0x00))
        continue;  // no block pending
      u8* src = nullptr;
      size_t src_sz = 0;
      for (auto& a : as)
        if (a.idx == (int)k) {
          src = a.b;
          src_sz = a.sz;
        }
      const size_t need = (size_t)grain * bpf;
      float peak = 0.f;
      if (src && src_sz >= need) {
        const u32 bps = (stype == 0) ? 2u : 4u;
        const u32 n = need / bps;
        if (stype == 1) {
          const float* p = reinterpret_cast<const float*>(src);
          for (u32 i = 0; i < n; i++) {
            float a = p[i] < 0 ? -p[i] : p[i];
            if (a > peak)
              peak = a;
          }
        } else if (stype == 0) {
          const i16* p = reinterpret_cast<const i16*>(src);
          for (u32 i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -p[i] : p[i]) / 32768.f;
            if (a > peak)
              peak = a;
          }
        } else {
          const i32* p = reinterpret_cast<const i32*>(src);
          for (u32 i = 0; i < n; i++) {
            float a = (p[i] < 0 ? -(float)p[i] : (float)p[i]) / 2147483648.f;
            if (a > peak)
              peak = a;
          }
        }
      }
      blocks++;
      if (peak > peak_max)
        peak_max = peak;
      const u64 now = ShmAudioNowUs();
      if (now - last > 2000000) {
        last = now;
        BASE_LOGI("shmprobe",
                  "port={} bpf={} type={} ch={} rate={} grain={} state={} "
                  "blocks={} peak={:.4f} peakMax={:.4f}",
                  k, bpf, stype, bpf / (stype == 0 ? 2u : 4u), rate, grain,
                  state, (unsigned long long)blocks, peak, peak_max);
      }
      // Release the port: the guest's next submit is gated on this being 0.
      *reinterpret_cast<volatile u64*>(slot + 0x00) = 0;
    }
    (void)ticks;
  }
}

void ShmAudioProbeMaybeStart() {
  const char* v = kShmProbe;
  if (!v || !*v)
    return;
  static base::Atomic<bool> started{false};
  bool e = false;
  if (!started.compare_exchange_strong(e, true))
    return;
  const long us = std::strtol(v, nullptr, 0);
  BASE_LOGI("shmprobe", "consumer probe every {}us", us);
  guest::SpawnThread("shmprobe",
                     [us] { ShmAudioProbeMain(us > 0 ? us : 10667); });
}

// Start the periodic dumper once, on the first matching shm we see.
void ShmAudioDumpMaybeStart() {
  const char* dir = kShmDump;
  if (!dir || !*dir)
    return;
  bool expected = false;
  if (!g_shm_audio_dumper.compare_exchange_strong(expected, true))
    return;
  const unsigned ms = kShmDumpMs;
  const unsigned n = kShmDumpN;
  const size_t mx = kShmDumpMax;
  BASE_LOGI("shmaudio", "dumper -> {} every {}ms x{} (<={:#x} B){}", dir, ms, n,
            mx, kShmAudioDumpDelta ? " delta-only" : "");
  guest::SpawnThread("shmaudio", [dir = base::String(dir), ms, n, mx,
                                  delta = kShmAudioDumpDelta.get()] {
    ShmAudioDumperMain(dir, ms, n, mx, delta);
  });
}

class ShmObject : public Object {
 public:
  ShmObject(ObjectTable& objects, base::String nm, ShmRef b)
      : Object(objects, Object::Type::kShm),
        shm_name(base::move(nm)),
        backing(base::move(b)) {}
  base::String shm_name;  // diagnostics / audio protocol key
  ShmRef backing;         // keeps the backing alive while this fd is open
};

// Backing block for a shm, grown to cover the requested range. Caller must not
// hold g_shmMutex; -1 on failure.
u8* ShmMap(ShmObject* shm, size_t size, size_t offset) {
  base::LockGuard<base::Mutex> lk(g_shm_mutex);
  auto& b = *shm->backing;
  size_t need = (offset + size + 0x3FFF) & ~size_t(0x3FFF);
  if (!b.base && need) {
    b.base = AllocLowGuest(need);
    if (!b.base)
      return reinterpret_cast<u8*>(-1);
    b.size = need;
    Process::GetActive()->GetVma().Add(b.base, need, Ppt::kW);
  }
  if (!b.base || offset > b.size)
    return reinterpret_cast<u8*>(-1);
  ShmAudioTrace("mmap", shm->shm_name, b.base + offset, size, offset);
  // Announce here too: a region mmap'd without ftruncate allocates above.
  AudioDaemonNoticeShm(shm->shm_name.c_str(), b.base, b.size);
  return b.base + offset;
}
}  // namespace

u8* PS4ABI
sys_mmap(void* addr, size_t size, u32 prot, u32 flags, u32 fd, size_t offset) {
  auto* proc = Process::GetActive();
  if (!proc)
    return reinterpret_cast<u8*>(-1);

  if (flags & MFlags::kStack || flags & MFlags::kNoextend)
    flags |= MFlags::kAnon;

  /*align the page*/
  size = (size + 0x3FFF) & 0xFFFFFFFFFFFFC000LL;

  // A zero-length mapping is invalid (BSD EINVAL); guests hit it on
  // error-recovery paths (mmap of an fd from a failed physhm_open/fstat).
  if (size == 0)
    return reinterpret_cast<u8*>(-SysError::eINVAL);

  // DELTA_MMAP_CALLER=<minMB>: scan the guest stack for module .text return
  // addresses to pin which guest code requested a big map.
  if (const char* mc = kMmapCaller) {
    size_t min_b =
        static_cast<size_t>(std::strtoull(mc, nullptr, 0)) * 1024 * 1024;
    if (min_b == 0)
      min_b = 64ull * 1024 * 1024;
    if (size >= min_b) {
      BASE_LOGI("mmap-caller", "size={:#x} prot={:#x} flags={:#x} fd={}:", size,
                prot, flags, fd);
      auto* sp = reinterpret_cast<uintptr_t*>(__builtin_frame_address(0));
      int shown = 0;
      for (int i = 0; i < 8192 && shown < 12; i++) {
        uintptr_t v = sp[i];
        for (auto& m : proc->GetModuleList()) {
          auto& mi = m->GetInfo();
          auto* t = mi.text_seg.addr;
          if (t && v >= (uintptr_t)t && v < (uintptr_t)t + mi.text_seg.size) {
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
  // and drops the offset; MAP_VOID forces prot 0; MAP_FIXED must be
  // page-aligned and not wrap. (Unbounded by VM_MAXUSER_ADDRESS: libkernel
  // fixed-maps a host stack guard far above the 2^40 guest ceiling.)
  if (flags & MFlags::kStack) {
    if (fd != static_cast<u32>(-1) || (prot & 3) != 3)
      return reinterpret_cast<u8*>(-SysError::eINVAL);
    offset = 0;
  }
  const bool void_reserve = (flags & 0x100) != 0;
  if (void_reserve)
    prot = 0;
  if (flags & MFlags::kFixed) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
    if ((a & 0x3FFF) != 0)
      return reinterpret_cast<u8*>(-SysError::eINVAL);
    uintptr_t a_end;
    if (__builtin_add_overflow(a, size, &a_end))
      return reinterpret_cast<u8*>(-SysError::eINVAL);
  }

  // addr is a hint unless MAP_FIXED: relocate it rather than alias an existing
  // map
  if (!(flags & MFlags::kFixed)) {
    if (!addr || proc->GetVma().Overlaps(static_cast<u8*>(addr), size))
      addr = nullptr;
  }

  // A hint inside the guest's user-stack region (or a VA range
  // ReserveGuestVaSpace() claimed) is guest-owned: our PROT_NONE placeholder
  // would relocate it and exhaust libkernel's internal arena.
  bool in_user_stack = false;
  if (addr) {
    auto& env = proc->GetEnv();
    auto a = reinterpret_cast<uintptr_t>(addr);
    auto lo = reinterpret_cast<uintptr_t>(env.user_stack);
    in_user_stack =
        env.user_stack && a >= lo && a + size <= lo + env.user_stack_size;
    in_user_stack = in_user_stack || IsGuestReservedVa(addr, size);
  }

  // A /dev/dmem mmap carries the direct-memory physical offset in `offset`.
  bool dmem_map = false;
  if (fd != -1) {
    auto obj = proc->GetObjTable().Get(fd);
    if (obj && obj->type() == Object::Type::kDevice &&
        static_cast<Device*>(obj.get())->IsDirectMemory())
      dmem_map = true;
    if (kMmapfdTrace)
      BASE_LOGI("mmapfd", "fd={} addr={:p} size={:#x} off={:#x} objType={}", fd,
                addr, size, offset, obj ? (int)obj->type() : -1);
    if (obj && obj->type() == Object::Type::kShm) {
      // Every mapper of this shm shares the backing (sized by ftruncate).
      return ShmMap(static_cast<ShmObject*>(obj.get()), size, offset);
    }
    if (obj) {
      // Device-backed mmap: use the region the device hands back; -1 = fall
      // through.
      auto* m =
          static_cast<Device*>(obj.get())->Map(addr, size, prot, flags, offset);
      if (m != reinterpret_cast<u8*>(-1)) {
        auto gprot = static_cast<Ppt>(prot & static_cast<u32>(Ppt::kRwx));
        if (dmem_map)
          proc->GetVma().AddDirect(m, size, gprot, prot, offset);
        else
          proc->GetVma().Add(m, size, gprot, prot);
        return m;
      }
    }
  }

  // MAP_ALIGNED(n): flags bits 31..24 carry log2 base alignment the kernel must
  // honor (SotC keys streaming-arena bookkeeping on it).
  const u32 align_log = (flags >> 24) & 0x1F;
  const size_t map_align =
      (align_log >= 14 && align_log < 40) ? (size_t(1) << align_log) : 0;
  if (map_align && addr && !(flags & MFlags::kFixed) &&
      (reinterpret_cast<uintptr_t>(addr) & (map_align - 1)))
    addr = nullptr;  // misaligned hint: pick our own aligned base instead

  // Keep prot-0 reservations out of the band titles MAP_FIXED their pools into:
  // Minecraft's direct-memory MAP_FIXED would silently replace the first page
  // of V8's pointer cage (and its read-only heap with it).
  if (addr && !(flags & MFlags::kFixed) && prot == 0 &&
      reinterpret_cast<uintptr_t>(addr) >= 0x1000000000ull &&
      reinterpret_cast<uintptr_t>(addr) < kGuestArenaFloor)
    addr = nullptr;

  void* ptr = nullptr;
  if (addr) {
    if (flags & MFlags::kFixed) {
      // MAP_FIXED stack maps [start-size, start): plant below the hint.
      void* want = addr;
      if (flags & MFlags::kStack)
        want = static_cast<u8*>(addr) - size;
      ForgetDmemVa(static_cast<u8*>(want), size);
      ptr = host_memory::AllocMem(want, size, Ppt::kW, Alt::kReservecommit);
      if (!ptr)
        ptr = host_memory::AllocMem(want, size, Ppt::kW,
                                    Alt::kCommit);  // maybe pre-reserved
    } else if (in_user_stack) {
      ptr = host_memory::AllocMem(addr, size, Ppt::kW, Alt::kCommit);
    } else if (host_memory::AllocMem(addr, size, Ppt::kW, Alt::kReserve)) {
      // A hint must never alias an existing mapping; reservecommit would
      // clobber it (a guest TLS hint destroyed a loaded module on Android).
      ptr = host_memory::AllocMem(addr, size, Ppt::kW, Alt::kCommit);
    } else if (WasGuestReleased(static_cast<u8*>(addr), size) &&
               !proc->GetVma().Overlaps(static_cast<u8*>(addr), size)) {
      // The probe only fails here because we kept the pages of a guest-unmapped
      // range; the address is free as far as the guest is concerned.
      ptr = host_memory::AllocMem(addr, size, Ppt::kW, Alt::kReservecommit);
    }
  }
  // No usable hint (or it was taken): pick a low (<2^40) address the guest's
  // own allocators will accept, not whatever high address the host hands out.
  if (!ptr)
    ptr = AllocLowGuest(size, map_align);
  if (!ptr)
    return reinterpret_cast<u8*>(-SysError::eNOMEM);

  // Track the guest-requested prot so VirtualQuery reports the truth; host
  // pages stay rwx (FEX reads guest memory directly, no protection faults
  // delivered).
  auto gprot = static_cast<Ppt>(prot & static_cast<u32>(Ppt::kRwx));

  // No zero-fill: every path above already returns kernel-zeroed anonymous
  // pages; self-fill faulted in multi-GiB pools (Minecraft, 43 GiB RSS).

  // File-backed mmap: copy the file's content in (Doom64 mmaps its WADs and
  // samples textures from the mapping); the read stops at EOF so an oversized
  // mapping keeps zeros past the file's end.
  if (fd != static_cast<u32>(-1)) {
    if (auto o = proc->GetObjTable().Get(fd))
      if (o->type() == Object::Type::kDevice) {
        // The kernel hands back base + (offset & 0x3FFF), so the fill starts at
        // the page-aligned offset for that contract to hold.
        const i64 file_off = static_cast<i64>(offset & ~size_t(0x3FFF));
        i64 got = static_cast<Device*>(o.get())->ReadAt(ptr, size, file_off);
        if (got > 0 && kMmapfdTrace)
          BASE_LOGI("mmapfd", "  filled {:p} from fd={} off={:#x} -> {} bytes",
                    ptr, fd, (unsigned long long)file_off, (long long)got);
      }
  }

  // MAP_VOID is a reservation (later MAP_FIXED commits punch it apart in the
  // VMA); virtual query must see it as reserved, not committed.
  if (dmem_map)
    proc->GetVma().AddDirect(static_cast<u8*>(ptr), size, gprot, prot, offset);
  else
    proc->GetVma().Add(static_cast<u8*>(ptr), size, gprot, prot, void_reserve);

  host_memory::ProtectMem(static_cast<void*>(ptr), size, Ppt::kRwx);

  // DELTA_GNMAP_TRACE: log mappings in the GNM/GPU aperture to pin how the AGC
  // ring buffers are mapped and by whom (coherency with guest PM4 writes is
  // an open item).
  if (kGnmapTrace) {
    u64 r = reinterpret_cast<u64>(ptr);
    if (r >= 0x8000000000ull && r < 0x8300000000ull) {
      // Walk the guest frame chain (libkernel keeps frame pointers) for the
      // real caller above libkernel's mmap wrapper.
      base::String callers;
      base::FormatTo(callers,
                     "{:p} size={:#x} fd={} flags={:#x}  callers:", ptr, size,
                     static_cast<int>(fd), flags);
      void* fp = __builtin_frame_address(0);
      for (int depth = 0; depth < 20 && fp; depth++) {
        auto* frame = static_cast<void**>(fp);
        void* ret = frame[1];
        if (!ret)
          break;
        char sym[160];
        kern::Symbolize(reinterpret_cast<uintptr_t>(ret), sym, sizeof(sym));
        base::FormatTo(callers, " [{}]", sym);
        fp = frame[0];
        if (reinterpret_cast<uintptr_t>(fp) < 0x10000)
          break;
      }
      BASE_LOGI("gnmap", "{}", callers.c_str());
    }
  }

  if (kMmapLog)
    BASE_LOGI("mmap",
              "mmap {:p}, {:x}, prot={:x} flags={:x} fd={} off={:#x}, {:p} -> "
              "{:p}",
              addr, size, prot, flags, static_cast<int>(fd), offset,
              __builtin_return_address(0), ptr);

  // Returned address = base + (offset & 0x3FFF), FreeBSD mmap semantics;
  // MAP_STACK returns the region top, like vm_map_stack.
  if (flags & MFlags::kStack)
    return &static_cast<u8*>(ptr)[size];

  return static_cast<u8*>(ptr) + (offset & 0x3FFF);
}

int PS4ABI sys_mprotect(u8* addr, size_t len, int prot) {
  auto* proc = Process::GetActive();
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
  const u32 sce_prot = static_cast<u32>(prot) & 0x37;
  proc->GetVma().ProtectRange(
      reinterpret_cast<u8*>(base), span,
      static_cast<Ppt>(sce_prot & static_cast<u32>(Ppt::kRwx)), sce_prot);
  return 0;
}

// System services we do not host: a shm + named semaphore pair sharing the
// service's name. The list grows one name at a time as titles walk into the
// next.
bool IsAbsentServiceChannel(const char* name) {
  static constexpr const char* kServices[] = {"SceNpTpip"};
  if (!name)
    return false;
  if (*name == '/')
    name++;
  const size_t n =
      std::strcspn(name, " ");  // the semaphore half is "<svc> <n>"
  for (const char* svc : kServices)
    if (std::strlen(svc) == n && std::strncmp(name, svc, n) == 0)
      return true;
  return false;
}

int PS4ABI sys_shm_open(const char* path, u32 flags, u16 mode) {
  auto* proc = Process::GetActive();
  if (!proc || !path)
    return -SysError::eINVAL;

  // From the kernel's sys_shm_open: O_WRONLY is invalid, unknown bits among
  // {O_RDWR, O_CREAT, O_TRUNC, O_EXCL} are EINVAL. Only the low half is
  // checked: libkernel passes high descriptor flags the console accepts.
  if ((flags & 1) != 0)
    return -SysError::eINVAL;
  if ((flags & 0xF1FC) != 0)
    return -SysError::eINVAL;

  constexpr u32 kO_CREAT = 0x0200, kO_EXCL = 0x0800, kO_TRUNC = 0x0400;
  base::String name(path);
  ShmRef backing;
  {
    base::LockGuard<base::Mutex> lk(g_shm_mutex);
    auto it = g_shm_by_name.find(name);
    if (it == g_shm_by_name.end()) {
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
        backing = base::MakeShared<ShmBacking>();
        backing->size = 0x10000;  // 64 KiB, ample for a settings block
        backing->base = AllocLowGuest(backing->size);
        if (backing->base) {
          std::memset(backing->base, 0, backing->size);
          proc->GetVma().Add(backing->base, backing->size, Ppt::kW);
        }
        BASE_LOGI("shm_open", "auto-provide system shm '{}' size={:#x}",
                  name.c_str(), backing->size);
        g_shm_by_name.emplace(name, backing);
      } else {
        // O_CREAT: fresh, empty backing; sized later by ftruncate.
        backing = base::MakeShared<ShmBacking>();
        g_shm_by_name.emplace(name, backing);
      }
    } else {
      if ((flags & kO_CREAT) && (flags & kO_EXCL))
        return -SysError::eEXIST;
      backing = it->second;
      // O_TRUNC|O_RDWR (0x402) truncates to zero; existing fds keep mapping the
      // empty backing, exactly like the real object.
      if ((flags & 0x403) == 0x402) {
        ShmAudioTrace("trunc", name, backing->base, 0, 0);
        backing->size = 0;
        if (backing->base)
          AudioDaemonNoticeShm(name.c_str(), backing->base, 0);
      }
    }
  }

  // A fresh fd per open, all sharing the named backing.
  auto* obj =
      new ShmObject(proc->GetObjTable(), base::move(name), base::move(backing));
  BASE_LOGI("shm_open", "'{}' flags={:#x} -> fd={}", path, flags,
            obj->handle());
  ShmAudioTrace("shm_open", obj->shm_name, nullptr, 0, flags);
  ShmAudioDumpMaybeStart();
  ShmAudioProbeMaybeStart();
  return obj->handle();
}

int PS4ABI sys_shm_unlink(const char* path) {
  if (!path)
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(g_shm_mutex);
  auto it = g_shm_by_name.find(path);
  if (it == g_shm_by_name.end())
    return -SysError::eNOENT;
  // Drop the name only; open fds hold shared refs to the backing, like the
  // kernel refcounting the shm object.
  g_shm_by_name.erase(it);
  return 0;
}

// Report a shm fd's backing size for fstat (shm objects aren't device-backed,
// so sys_fstat's fdToDevice path can't size them). Returns SIZE_MAX if `fd`
// isn't a shm, so the caller falls through to the normal path.
size_t ShmFstatSize(u32 fd) {
  auto* proc = Process::GetActive();
  if (!proc)
    return SIZE_MAX;
  auto obj = proc->GetObjTable().Get(fd);
  if (!obj || obj->type() != Object::Type::kShm)
    return SIZE_MAX;
  auto* shm = static_cast<ShmObject*>(obj.get());
  base::LockGuard<base::Mutex> lk(g_shm_mutex);
  return shm->backing->size;
}

int PS4ABI sys_ftruncate(u32 fd, i64 length) {
  auto* proc = Process::GetActive();
  if (!proc || length < 0)
    return -SysError::eINVAL;
  auto obj = proc->GetObjTable().Get(fd);
  if (!obj)
    return -SysError::eBADF;
  // The kernel dispatches to the file type's truncate method; a type without
  // one (e.g. a device) is EINVAL, not EBADF.
  if (obj->type() != Object::Type::kShm)
    return -SysError::eINVAL;

  auto* shm = static_cast<ShmObject*>(obj.get());
  const size_t raw = static_cast<size_t>(length);
  const size_t want = (raw + 0x3FFF) & ~size_t(0x3FFF);
  base::LockGuard<base::Mutex> lk(g_shm_mutex);
  auto& b = *shm->backing;
  // The RAW length matters for the protocol spec (the rounded `want` hides it).
  ShmAudioTrace("ftruncate", shm->shm_name, b.base, raw, want);
  if (want < b.size) {
    // Shrink (the kernel frees the tail pages); the host block stays put, a
    // later grow reallocates and copies `size` bytes.
    b.size = want;
    AudioDaemonNoticeShm(shm->shm_name.c_str(), b.base, b.size);
    return 0;
  }
  if (want == b.size)
    return 0;
  u8* nb = AllocLowGuest(want);
  if (!nb)
    return -SysError::eNOMEM;
  if (b.base)
    std::memcpy(nb, b.base,
                b.size);  // grow before first mmap: preserve contents
  b.base = nb;
  b.size = want;
  proc->GetVma().Add(b.base, want, Ppt::kW);
  ShmAudioTrace("sized", shm->shm_name, b.base, want, 0);
  ShmAudioPoison(shm->shm_name, b.base, want);
  // The LLE audio regions become consumable here; the base can MOVE on a later
  // grow, hence the fresh pointer each time.
  AudioDaemonNoticeShm(shm->shm_name.c_str(), b.base, b.size);
  return 0;
}

int PS4ABI sys_mname(u8* ptr, size_t len, const char* name, void*) {
  auto* proc = Process::GetActive();
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
  proc->GetVma().SetRangeName(ptr, len, tag);
  // Titles name thread stacks this way; carry the tag onto the host thread
  // running on that stack (attribution).
  NameThreadsForRange(ptr, len, tag);
  return 0;
}

struct mdbg_property {  // NOLINT(readability-identifier-naming): guest ABI name
  i32 unk;
  i32 unk2;
  void* addr;
  size_t area_size;
  i64 unk3;
  i64 unk4;
  char name[32];
};

static_assert(sizeof(mdbg_property) == 72);

// Stand-in for the kernel's per-process debug-raise qword (proc+2600); no
// debugger is attached, so a raise is recorded and reported delivered.
static base::Atomic<u64> g_mdbg_flags{0};

// Mirrors mdbg_service_raise; the suspend notification callback is
// unregistered, so report the raise as delivered.
static int MdbgServiceRaise(uintptr_t reason) {
  if (reason <= 0x7E)
    g_mdbg_flags.fetch_or(static_cast<u64>(reason) << 32);
  g_mdbg_flags.fetch_or(2);
  LOG_WARNING("mdbg raise: reason={} (no debugger attached, ignoring)", reason);
  return 0;
}

int PS4ABI sys_mdbg_service(u32 op, void* arg1, void* arg2, void* a3) {
  (void)arg2;
  (void)a3;
  switch (op) {
    case 0:
      // Kernel: copyout the proc's debug flags qword (proc+2600) to the caller.
      *static_cast<u64*>(arg1) = g_mdbg_flags.load();
      return 0;
    case 1: {
      // Copyin a 72-byte property and register a named object (a no-op here);
      // log it.
      auto* info = static_cast<mdbg_property*>(arg1);
      char tag[32];
      tag[31] = 0;  // kernel zeroes the last name byte after copyin
      memcpy(tag, info->name, sizeof(tag) - 1);
      LOG_WARNING("mdbg property {} for addr {} size {}", tag, info->addr,
                  info->area_size);
      return 0;
    }
    case 3:
      // Kernel: mdbg_service_raise with the raw second syscall argument as
      // reason.
      return MdbgServiceRaise(reinterpret_cast<uintptr_t>(arg1));
    case 4:
      // Raise only while debug mode is allowed; both conditions hold for us.
      return MdbgServiceRaise(reinterpret_cast<uintptr_t>(arg1));
    case 7: {
      // The mdbg text facility: log the message.
      const char* msg = static_cast<const char*>(arg1);
      if (!msg)
        return -SysError::eINVAL;
      base::String s(msg, strnlen(msg, 0x1000));
      LOG_INFO("[mdbg-text] {}", s);
      return 0;
    }
    case 8:
      // libkernel's debug thread treats any error as "event arrived" and runs
      // the title's coredump callback on an UNINITIALIZED buffer (Demon's Souls
      // faults at boot). No debugger ever posts here; park the caller like the
      // kernel.
      LOG_INFO("mdbg wait-event: parking caller (no debugger attached)");
      while (!guest::Stopping())
        base::SleepForMilliseconds(20);
      return -SysError::eINTR;
    default:
      // The kernel returns 78 (eNOSYS in our table) for unknown ops.
      return -SysError::eNOSYS;
  }
}

// sys_dmem_container (586): 0xFFFFFFFF reads the container id, 0/1 sets it
// (needs privilege 0x2AD; the kernel keeps it at proc+2020). We only track it.
int PS4ABI sys_dmem_container(u32 op) {
  static base::Atomic<u32> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  if (op > 1)
    return -SysError::eINVAL;
  current.store(op);
  return 0;
}
}  // namespace kern

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kVqTrace, "DELTA_VQ_TRACE", false);
}  // namespace

namespace kern {

// Our VM is a flat host-backed low arena, never compacted: the HOST pages stay
// mapped (host_memory::FreeMem takes no length; stale pointers read stable
// garbage instead of faulting). The BOOKKEEPING must still be released: titles
// churn VA and key allocator state off VirtualQuery's [start,end), so dead VMA
// entries report stale bounds.
int PS4ABI sys_munmap(void* addr, size_t len) {
  static base::Atomic<bool> warned{false};
  if (!warned.exchange(true))
    LOG_WARNING(
        "sys_munmap: host pages are retained (first was {} +{:#x}); "
        "only the VMA bookkeeping is released",
        static_cast<const void*>(addr), len);
  if (auto* proc = Process::GetActive(); proc && addr && len) {
    AudioDaemonForgetRange(addr, len);
    // Exception to "keep the host pages": a whole PROT_NONE reservation holds
    // no data and leaving it mapped makes the next reservation there relocate.
    // V8 reserves padded, frees, re-reserves exact; a stale pointer into the
    // padding should fault where the mistake is.
    auto region = proc->GetVma().Get(static_cast<u8*>(addr));
    if (region && region->ptr == addr && region->size == len &&
        region->sce_prot == 0)
      host_memory::FreeMem(addr, len);
    proc->GetVma().Remove(static_cast<u8*>(addr), len);
    NoteGuestReleased(static_cast<u8*>(addr), len);
    ForgetDmemVa(static_cast<u8*>(addr), len);
  }
  return 0;
}

// The guest libc allocates through mmap, so the brk is unused. A benign 0 keeps
// any stray caller satisfied without handing it a usable region.
i64 PS4ABI sys_obreak(void*) {
  return 0;
}
i64 PS4ABI sys_sbrk(intptr_t) {
  return 0;
}

// Anonymous host memory has no file backing, so there is nothing to flush.
int PS4ABI sys_msync(void*, size_t, int) {
  return 0;
}

int PS4ABI sys_madvise(void*, size_t, int) {
  return 0;
}

// The guest arena is fully committed and never pages out: report MINCORE_INCORE
// per page so a residency probe sees the truth.
int PS4ABI sys_mincore(void* addr, size_t len, char* vec) {
  if (!vec)
    return -SysError::eFAULT;
  size_t pages = (len + 0x3FFF) >> 14;
  std::memset(vec, 0x01 /*MINCORE_INCORE*/, pages);
  return 0;
}

// All our pages are committed host memory that never pages out, so locking is a
// no-op.
int PS4ABI sys_mlock(const void*, size_t) {
  return 0;
}
int PS4ABI sys_munlock(const void*, size_t) {
  return 0;
}
int PS4ABI sys_mlockall(int) {
  return 0;
}
int PS4ABI sys_munlockall() {
  return 0;
}

// minherit only matters across fork(), which we don't model.
int PS4ABI sys_minherit(void*, size_t, int) {
  return 0;
}

// sceKernelQueryMemoryProtection(addr, &start, &end, &prot). The libkernel
// wrapper passes one scratch struct as arg2 for the kernel to fill, then
// distributes: {start@0, end@8, prot@0x10}, per the wrapper at libkernel
// 0x17ef0 (prot masked 0x37). Unmapped addr = the EACCES the wrapper
// translates.
int PS4ABI sys_query_memory_protection(void* addr, void* info) {
  auto* proc = Process::GetActive();
  if (!proc || !info)
    return -SysError::eINVAL;
  auto region = proc->GetVma().Get(static_cast<u8*>(addr));
  if (!region)
    return -SysError::eACCES;

  auto* qp = static_cast<u8*>(info);
  std::memset(qp, 0, 0x18);
  void* start = region->ptr;
  void* end = region->ptr + region->size;
  // Full SCE prot (with GPU bits) if we kept it, else the host r/w/x bits.
  u32 prot =
      region->sce_prot ? region->sce_prot : static_cast<u32>(region->prot);
  std::memcpy(qp + 0x00, &start, sizeof(void*));
  std::memcpy(qp + 0x08, &end, sizeof(void*));
  std::memcpy(qp + 0x10, &prot, sizeof(u32));
  return 0;
}

// The host's own view of an address, for ranges the guest VMA never recorded.
struct HostMapping {
  u64 start = 0;
  u64 end = 0;
  u32 prot = 0;  // SCE r/w/x bits, 0 when the address is not mapped at all
};

HostMapping HostMappingOf(const void* addr) {
  HostMapping out;
  const u64 want = reinterpret_cast<u64>(addr);
  std::FILE* f = std::fopen("/proc/self/maps", "re");
  if (!f)
    return out;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) != 3)
      continue;
    if (want < lo || want >= hi)
      continue;
    out.start = lo;
    out.end = hi;
    out.prot = (perms[0] == 'r' ? 1u : 0u) | (perms[1] == 'w' ? 2u : 0u) |
               (perms[2] == 'x' ? 4u : 0u);
    break;
  }
  std::fclose(f);
  return out;
}

// sceKernelVirtualQuery(addr, flags, info, size); layout verified against the
// consumer at libkernel 0x2b9d0 (size 0x48, name at +0x21):
// 0x00 start 0x08 end 0x10 offset 0x18 protection 0x1C memoryType
// 0x20 bits (flexible/direct/stack/pooled/committed) 0x21 char[32] name.
// Our anon low-arena maps are flexible and committed.
int PS4ABI sys_virtual_query(const void* addr,
                             int /*flags*/,
                             void* info,
                             size_t info_size) {
  auto* proc = Process::GetActive();
  if (!proc || !info || info_size == 0)
    return -SysError::eINVAL;

  std::memset(info, 0, info_size);
  auto region =
      proc->GetVma().Get(const_cast<u8*>(static_cast<const u8*>(addr)));
  if (!region) {
    // Memory we allocated outside the guest VMA is still guest-used memory: a
    // library validating a caller's buffer with this call must not hear
    // "unmapped". libSce Videodec2 does exactly that, answering every AvPlayer
    // call with ARGUMENT_POINTER (Astro Bot's title screen sat behind a video
    // that never started).
    if (const HostMapping host = HostMappingOf(addr); host.prot) {
      auto* vq = static_cast<u8*>(info);
      std::memcpy(vq + 0x00, &host.start, sizeof(u64));
      std::memcpy(vq + 0x08, &host.end, sizeof(u64));
      std::memcpy(vq + 0x10, &host.start, sizeof(u64));
      if (info_size >= 0x1C + sizeof(int)) {
        // Host PROT_READ/WRITE happen to be SCE's CPU bits, but a host mapping
        // carries no GPU bits, and a library validating hardware-bound buffers
        // demands them. libSceVdecCore (+0x17c50) rejects any range without
        // prot & 0x30 (error 5, why Astro Bot's intro video never decoded). The
        // guest is identity-mapped for our GPU: report the GPU bits alongside
        // the CPU ones.
        int prot = static_cast<int>(host.prot);
        if (prot & 0x1)
          prot |= 0x10;  // GPU read
        if (prot & 0x2)
          prot |= 0x20;          // GPU write
        const int mem_type = 0;  // WB_ONION, like any other CPU mapping
        std::memcpy(vq + 0x18, &prot, sizeof(int));
        std::memcpy(vq + 0x1C, &mem_type, sizeof(int));
      }
      if (info_size >= 0x21)
        vq[0x20] = 0x01 | 0x10;  // flexible + committed
      if (kVqTrace)
        BASE_LOGI("vq", "addr={:p} host mapping [{:#x}..{:#x}) prot={:#x}",
                  addr, (unsigned long long)host.start,
                  (unsigned long long)host.end, host.prot);
      return 0;
    }
    // Worth seeing: a caller that walks its own heap this way reads the zeroed
    // struct as "not committed" and silently skips the range.
    if (kVqTrace)
      BASE_LOGI("vq", "addr={:p} NOT MAPPED", addr);
    return -SysError::eACCES;
  }

  // A region with NO protection at all is a reservation, and a title committing
  // sub-ranges inside one left us reporting the outer entry (Astro Bot's
  // decoder buffer sits in a 1.9 GiB reservation; libSceVdecCore wants prot &
  // 0x2/0x30). If the host has accessible pages there, that mapping is the
  // truth; an untouched PROT_NONE reservation still reads uncommitted.
  if (!region->sce_prot && !static_cast<u32>(region->prot)) {
    if (const HostMapping host = HostMappingOf(addr); host.prot) {
      auto* vq = static_cast<u8*>(info);
      std::memcpy(vq + 0x00, &host.start, sizeof(u64));
      std::memcpy(vq + 0x08, &host.end, sizeof(u64));
      std::memcpy(vq + 0x10, &host.start, sizeof(u64));
      if (info_size >= 0x1C + sizeof(int)) {
        int prot = static_cast<int>(host.prot);
        if (prot & 0x1)
          prot |= 0x10;  // GPU read: guest memory is identity-mapped for us
        if (prot & 0x2)
          prot |= 0x20;          // GPU write
        const int mem_type = 0;  // WB_ONION
        std::memcpy(vq + 0x18, &prot, sizeof(int));
        std::memcpy(vq + 0x1C, &mem_type, sizeof(int));
      }
      if (info_size >= 0x21)
        vq[0x20] = 0x01 | 0x10;  // flexible + committed
      if (kVqTrace)
        BASE_LOGI("vq",
                  "addr={:p} committed inside a reservation: "
                  "[{:#x}..{:#x}) prot={:#x}",
                  addr, (unsigned long long)host.start,
                  (unsigned long long)host.end, host.prot);
      return 0;
    }
  }

  auto* vq = static_cast<u8*>(info);
  void* start = region->ptr;
  void* end = region->ptr + region->size;
  std::memcpy(vq + 0x00, &start, sizeof(void*));
  std::memcpy(vq + 0x08, &end, sizeof(void*));
  // +0x10 = the region's direct-memory offset, exact for /dev/dmem ranges: SotC
  // turns (offset + addr - start) into a 64 KiB heap-map block index and marks
  // nothing when out of range. Elsewhere we have no dmem pool
  // (identity-mapped), but libSceVideoOut rejects a scanout buffer unless the
  // offset shares the VA's low 16 bits (tiling alignment), so report the VA
  // there; zero failed every scanout register.
  u64 offset =
      region->has_phys ? region->phys_offset : reinterpret_cast<u64>(start);
  std::memcpy(vq + 0x10, &offset, sizeof(u64));
  // GPU-accessible memory (guest asked bits 0x10/0x20) is direct/physical in
  // SCE terms: report WC_GARLIC (memType 3) + direct bit, what libSceVideoOut
  // checks before registering a scanout buffer; plain CPU memory stays flexible
  // WB_ONION.
  bool gpu = (region->sce_prot & 0x30) != 0;
  // For direct memory, the reservation's type is the truth (titles route
  // addresses to heaps by it); inferring GARLIC from GPU prot bits made a CPU
  // heap mapped GPU-visible look like the GPU one.
  int mem_type = region->has_phys ? DmemTypeForOffset(region->phys_offset) : -1;
  if (mem_type < 0)
    mem_type = gpu ? 3 : 0;  // 3 = SCE_KERNEL_WC_GARLIC, 0 = WB_ONION
  if (info_size >= 0x1C + sizeof(int)) {
    int prot = region->sce_prot ? static_cast<int>(region->sce_prot)
                                : static_cast<int>(region->prot);
    std::memcpy(vq + 0x18, &prot, sizeof(int));
    std::memcpy(vq + 0x1C, &mem_type, sizeof(int));
  }
  if (kVqTrace)
    BASE_LOGI("vq",
              "addr={:p} region=[{:p}..{:p}) sceProt={:#x} memType={} rsv={} "
              "off={:#x}{}",
              addr, start, end, region->sce_prot, mem_type,
              region->reserved ? 1 : 0, (unsigned long long)offset,
              region->has_phys ? " (dmem)" : "");
  if (info_size >= 0x21) {
    // flexible(0x01) | direct(0x02, GPU mem) | committed(0x10). A MAP_VOID
    // reservation is none of these; titles branch on isCommitted to decide
    // whether a range still needs a real commit.
    vq[0x20] = region->reserved
                   ? 0x00
                   : (0x01 | 0x10 | ((gpu || region->has_phys) ? 0x02 : 0x00));
    if (region->name) {
      size_t n = std::strlen(region->name);
      if (n > 31)
        n = 31;
      std::memcpy(vq + 0x21, region->name, n);
      vq[0x21 + n] = '\0';
    }
  }
  return 0;
}

// sceKernelBatchMap/2: a list of dmem map/unmap/protect ops in one call. Entry
// layout (libkernel wrapper): 0x00 start 0x08 offset (MAP_DIRECT) 0x10 length
// 0x18 protection 0x19 memoryType 0x1c operation; ops 0 MAP_DIRECT, 1 UNMAP,
// 2 PROTECT, 3 MAP_FLEXIBLE, 4 TYPE_PROTECT. The maps must really commit at
// `start`: SotC batch-maps its GPU pools, and the old ignore-stub left PM4
// referencing never-backed VAs, faulting the submit.
int PS4ABI sys_batch_map(u32 /*handle*/,
                         u32 /*flags*/,
                         void* entries,
                         int count,
                         int* processed) {
  struct BatchMapEntry {
    u64 start;
    u64 offset;
    u64 length;
    u8 prot;
    u8 type;
    u16 pad;
    u32 operation;
  };
  static_assert(sizeof(BatchMapEntry) == 0x20, "batch-map entry is 32 bytes");

  auto* e = static_cast<BatchMapEntry*>(entries);
  int done = 0;
  for (; e && done < count; done++) {
    const auto& op = e[done];
    if (kDmemTrace)
      BASE_LOGI("dmem",
                "batch[{}/{}] op={} start={:#x} off={:#x} len={:#x} prot={:#x} "
                "type={}",
                done, count, op.operation, (unsigned long long)op.start,
                (unsigned long long)op.offset, (unsigned long long)op.length,
                op.prot, op.type);
    switch (op.operation) {
      case 0:  // MAP_DIRECT: back the VA with the shared dmem store at
               // op.offset, MAP_FLEXIBLE: no physical offset, so map the shared
               // dmem backing (syscall 628 semantics): every VA mapping it
               // aliases the same bytes. Skyrim maps GPU pools this way then
               // waits on a GPU-written label; private pages never see it.
      case 3: {  // MAP_FLEXIBLE: no physical offset, plain anonymous memory
        if (!op.start || !op.length) {
          if (processed)
            *processed = done;
          return -SysError::eINVAL;
        }
        auto* pr = Process::GetActive();
        // Sharing the dmem backing is NOT safe for PS4 yet: SotC maps one
        // physical offset at 1664 successive VAs and never releases, so a
        // shared store aliases every historical mapping at once and the title's
        // memory dissolves (measured twice, both times SotC stopped rendering
        // even its intro). Needs understanding of what the guest does with that
        // offset first.
        const bool ps5 = pr && pr->GetPlatform() == Process::Platform::kPs5;
        const int fd = (op.operation == 0 && ps5) ? DmemBackingFd() : -1;
        if (fd >= 0 && op.offset + op.length <= DmemBackingSize()) {
          void* p = ::mmap(reinterpret_cast<void*>(op.start), op.length,
                           PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
                           static_cast<off_t>(op.offset));
          if (p != MAP_FAILED) {
            pr->GetVma().Add(reinterpret_cast<u8*>(p), op.length,
                             host_memory::PageProtection::kW);
            break;
          }
        }
        u8* p =
            sys_mmap(reinterpret_cast<void*>(op.start), op.length, op.prot,
                     MFlags::kFixed | MFlags::kAnon, static_cast<u32>(-1), 0);
        if (IsErrnoPtr(p)) {
          if (processed)
            *processed = done;
          return -SysError::eNOMEM;
        }
        break;
      }
      case 1:  // UNMAP: host pages retained, bookkeeping released (see
               // sys_munmap)
        if (auto* pr = Process::GetActive(); pr && op.start && op.length)
          pr->GetVma().Remove(reinterpret_cast<u8*>(op.start), op.length);
        break;
      case 2:  // PROTECT / TYPE_PROTECT: our flat arena stays permissive; the
      case 4:  // title only narrows protections it already owns
        break;
      default:
        LOG_WARNING("sys_batch_map: unknown op {} (entry {})", op.operation,
                    done);
        break;
    }
  }
  if (processed)
    *processed = done;
  return 0;
}

// sys_set_vm_container (559): arg == -1 returns the current vm container id;
// arg 0 or 1 selects it (requires privilege). The kernel validates: unsigned
// (arg+1) > 2 is EINVAL, and arg > 1 is EINVAL. We track the id (default 0).
int PS4ABI sys_set_vm_container(u32 op) {
  static base::Atomic<u32> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  if (op > 1)
    return -SysError::eINVAL;
  current.store(op);
  return 0;
}

// sceKernelMapDirectMemory (628). Real ABI (libkernel 01.14.00): rdi=VA hint,
// rsi=len, rdx=prot, rcx=flags (0x10 = FIXED),
// r8=packed(alignShift<<24|memType), r9=directMemoryStart (the PHYSICAL offset
// from AllocateMainDirectMemory). The physical offset is the source of truth:
// map the VA to the shared dmem backing at that offset (MAP_SHARED) so a
// CPU-written command buffer and the GPU's view alias the same bytes (without
// it the CP read all-zero DCBs). Falls back to anonymous memory; returns the
// mapped VA.
i64 PS4ABI sys_mmap_dmem(void* addr,
                         size_t len,
                         int prot,
                         int flags,
                         i64 /*packed*/,
                         i64 phys_offset) {
  const bool fixed_req = (flags & 0x10) != 0;
  auto* active = Process::GetActive();
  const bool ps5 = active && active->GetPlatform() == Process::Platform::kPs5;
  const int fd = ps5 ? DmemBackingFd() : -1;  // see the batch-map note above
  if (kDmemTrace)
    BASE_LOGI("dmem",
              "map628 va={:p} len={:#x} prot={:#x} flags={:#x} physOff={:#x} "
              "fixed={}",
              addr, len, prot, flags, (unsigned long long)phys_offset,
              fixed_req ? 1 : 0);
  if (fd >= 0 && phys_offset >= 0 &&
      static_cast<u64>(phys_offset) + len <= DmemBackingSize()) {
    const int mflags = MAP_SHARED | (fixed_req ? MAP_FIXED : 0);
    void* p = ::mmap(addr, len, PROT_READ | PROT_WRITE, mflags, fd,
                     static_cast<off_t>(phys_offset));
    if (p != MAP_FAILED) {
      Process::GetActive()->GetVma().AddDirect(
          reinterpret_cast<u8*>(p), len, host_memory::PageProtection::kW,
          static_cast<u32>(prot), static_cast<u64>(phys_offset));
      return reinterpret_cast<i64>(p);
    }
  }
  // Fallback: plain anonymous mapping (loses aliasing but keeps the region
  // live).
  u8* p = sys_mmap(addr, len, PROT_READ | PROT_WRITE,
                   MFlags::kAnon | (fixed_req ? MFlags::kFixed : 0),
                   static_cast<u32>(-1), 0);
  if (IsErrnoPtr(p))
    return -SysError::eNOMEM;
  // Still direct memory as far as the guest is concerned: it keys its own heap
  // map off the physical offset the query reports back.
  if (phys_offset >= 0)
    Process::GetActive()->GetVma().AddDirect(
        p, len, host_memory::PageProtection::kW, static_cast<u32>(prot),
        static_cast<u64>(phys_offset));
  return reinterpret_cast<i64>(p);
}

int PS4ABI sys_cpuset(void*, int, int, i64, size_t, void*) {
  return 0;
}

int PS4ABI sys_extend_page_table_pool() {
  return 0;
}

i64 PS4ABI sys_get_vm_map_timestamp() {
  return 0;
}

int PS4ABI sys_get_map_statistics(void* info) {
  if (info)
    std::memset(info, 0, 0x40);
  return 0;
}

// Thread stacks live in the leaked flat arena, so freeing one is a no-op.
int PS4ABI sys_free_stack(void*, size_t) {
  return 0;
}

// The JIT shm object the guest later mmaps to hold generated code. The real
// syscall returns an fd; we return an RWX anonymous region instead, so a guest
// that maps the result by fd (mmap(fd)) will NOT see this region. Loud once
// because that mismatch breaks runtime code generation if a title relies on it.
i64 PS4ABI sys_jitshm_create(size_t len, u32 flags) {
  (void)flags;
  static base::Atomic<bool> once{false};
  LogOnce(once,
          "jitshm_create returns a raw RWX region, not an fd; JIT may break");
  return (i64)sys_mmap(nullptr, len, 7 /*rwx*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_jitshm_alias() {
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_get_paging_stats_of_all_threads() {
  return 0;
}
int PS4ABI sys_get_paging_stats_of_all_objects() {
  return 0;
}

int PS4ABI sys_get_resident_count() {
  return 0;
}
int PS4ABI sys_get_resident_fmem_count() {
  return 0;
}

// Physically-contiguous shared memory; we don't back it, so deny and let the
// guest fall back to ordinary memory.
int PS4ABI sys_physhm_open() {
  return -SysError::eOPNOTSUPP;
}
int PS4ABI sys_physhm_unlink() {
  return 0;
}

int PS4ABI sys_set_phys_fmem_limit() {
  return 0;
}

int PS4ABI sys_get_kernel_mem_statistics(void* out) {
  if (out)
    std::memset(out, 0, 0x40);
  return 0;
}

// Carve a mapping out of a block pool. We don't model pools, so back it with a
// plain anonymous RW region of the requested length. Logged once because the
// pool handle and any accounting it implies are ignored.
i64 PS4ABI sys_blockpool_map(i64 pool, size_t len, u32 prot, u32 flags) {
  (void)pool;
  (void)prot;
  (void)flags;
  static base::Atomic<bool> once{false};
  LogOnce(once, "blockpool_map backs the pool with a plain anon region");
  return (i64)sys_mmap(nullptr, len, 3 /*rw*/, 0x1000 /*anon*/, -1, 0);
}

int PS4ABI sys_blockpool_unmap() {
  return 0;
}
i64 PS4ABI sys_blockpool_batch(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
  if (kBlockpoolTrace) {
    BASE_LOGI("blockpool_batch",
              "a0={:#x} a1={:#x} a2={:#x} a3={:#x} a4={:#x} a5={:#x}",
              (unsigned long long)a0, (unsigned long long)a1,
              (unsigned long long)a2, (unsigned long long)a3,
              (unsigned long long)a4, (unsigned long long)a5);
    // a1 commonly points at the command array; dump a few 64-bit words.
    if (a1 > 0x10000) {
      auto* w = reinterpret_cast<u64*>(a1);
      BASE_LOGI("blockpool_batch",
                "  cmd[0..7]: {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
                (unsigned long long)w[0], (unsigned long long)w[1],
                (unsigned long long)w[2], (unsigned long long)w[3],
                (unsigned long long)w[4], (unsigned long long)w[5],
                (unsigned long long)w[6], (unsigned long long)w[7]);
    }
  }
  return 0;
}

int PS4ABI sys_get_page_table_stats() {
  return 0;
}

// The PS4 uses a 16 KiB page size.
int PS4ABI sys_getpagesize() {
  return 16384;
}

// sys_blockpool_open (653): descriptor for the flexible-memory block pool.
// Unmodelled; the boot caller never feeds it to blockpool_map/mmap, so return a
// fixed positive non-stdio handle.
int PS4ABI sys_blockpool_open() {
  return 0x4000;
}

namespace {
const guest::SessionReset g_session_reset([] {
  g_low_next.store(kGuestArenaFloor);
  guest::ResetResource(g_released);
  guest::ResetResource(g_shm_by_name);
  guest::ResetResource(g_shm_audio_dumper);
  guest::ResetResource(g_mdbg_flags);
});
}  // namespace

}  // namespace kern
