/*
 * PS4Delta : GCN / Liverpool texture de-tiling (32bpp). See gcn_detile.h.
 *
 * Faithful port of the AMD AddrLib (Liverpool) tiling/de-tiling swizzle for
 * 32bpp, one-sample 2D and 2D-array textures, including mip chains and thick
 * microtile Z interleaving. XThick surfaces are degraded to Thick as AddrLib
 * requires when a 32bpp microtile exceeds Liverpool's 1 KiB row size.
 *
 * Unlike the previous version, every addressing parameter (array mode, micro
 * tile mode, pipe config, num_pipes/banks, bank_width/height, macro aspect,
 * tile split) is DERIVED from the surface's tiling_index via the standard
 * Liverpool GB_TILE_MODE + macro-tile-mode tables, rather than hardcoding one
 * config. That keeps Isaac's surfaces (Display2DThin / 1D micro,
 * P8_32x32_16x16, 16 banks) byte-identical to before while also covering the
 * other tiled modes a title like Doom64 can ask for (e.g. Thin* modes that use
 * P8_32x32_8x16).
 *
 * Tables follow the AMD AddrLib GB_TILE_MODE register decode for the Liverpool
 * (base-PS4) GPU, as documented in the AMD GCN/Sea Islands ISA + register
 * specs. Console reported as base PS4 (non-Neo) so num_pipes is 8, the standard
 * value for these modes.
 */

#include "gpu/gcn/gcn_detile.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDetileMt, "DELTA_GPU_DETILE_MT", true);
DELTA_OPTION(const char*, kDetileThreads, "DELTA_GPU_DETILE_THREADS", nullptr);
}  // namespace

namespace gpu::gcn {
namespace {

// --- Persistent worker pool. ----------------------------------------------
// Detile kernels submit independent 8-row microtile bands; format-conversion
// callers can submit ordinary row ranges. Threads are parked between regions,
// work is dynamically claimed, and the calling thread participates too.
class RowPool {
 public:
  static RowPool& get() {
    static RowPool inst;
    return inst;
  }

  void run(uint32_t units,
           uint64_t work_items,
           const std::function<void(uint32_t, uint32_t)>& fn) {
    // A callback can use another detile operation for one of its units. Running
    // that nested region inline avoids recursively taking run_mutex_ and keeps
    // the outer workers useful instead of deadlocking them at a second barrier.
    if (running_ == this) {
      if (units)
        fn(0, units);
      return;
    }
    // Serialize whole regions: the GPU pipeline is single-threaded, but this
    // guards the shared state should two callers ever overlap.
    std::lock_guard<std::mutex> serial(run_mutex_);
    if (!enabled_ || threads_.empty() || units < 2 ||
        work_items < kMinParallelItems) {
      invoke(fn, 0, units);
      return;
    }
    {
      std::unique_lock<std::mutex> lk(mtx_);
      fn_ = &fn;
      total_units_ = units;
      cursor_.store(0, std::memory_order_relaxed);
      const uint64_t useful_lanes = std::max<uint64_t>(
          2, (work_items + kItemsPerLane - 1) / kItemsPerLane);
      worker_count_ = std::min<uint32_t>(
          {static_cast<uint32_t>(threads_.size()), units - 1,
           static_cast<uint32_t>(std::min<uint64_t>(
               useful_lanes - 1, static_cast<uint64_t>(UINT32_MAX)))});
      // Aim for several chunks per lane so stealing balances uneven units.
      const uint32_t lanes = worker_count_ + 1;
      block_ = std::max(1u, units / (lanes * 4u));
      active_ = worker_count_;
      ++generation_;
      start_cv_.notify_all();
    }
    drain();  // the caller is a lane too
    {
      std::unique_lock<std::mutex> lk(mtx_);
      done_cv_.wait(lk, [this] { return active_ == 0; });
      fn_ = nullptr;
    }
  }

 private:
  static constexpr uint64_t kMinParallelItems = 32 * 1024;
  static constexpr uint64_t kItemsPerLane = 32 * 1024;

  RowPool() {
    enabled_ = kDetileMt;
    uint32_t hw = std::thread::hardware_concurrency();
    uint32_t n = std::min(8u, hw ? hw / 2u : 1u);
    if (const char* configured = kDetileThreads) {
      char* end = nullptr;
      const unsigned long lanes = std::strtoul(configured, &end, 10);
      if (end != configured && !*end)
        n = lanes > 1 ? static_cast<uint32_t>(std::min(lanes - 1, 63ul)) : 0;
    }
    if (!enabled_)
      n = 0;
    for (uint32_t i = 0; i < n; i++)
      threads_.emplace_back([this, i] { worker(i); });
  }

  ~RowPool() {
    {
      std::unique_lock<std::mutex> lk(mtx_);
      stop_ = true;
      start_cv_.notify_all();
    }
    for (auto& t : threads_)
      if (t.joinable())
        t.join();
  }

  // Claim and process chunks until the range is exhausted.
  void drain() {
    for (;;) {
      uint32_t s = cursor_.fetch_add(block_, std::memory_order_relaxed);
      if (s >= total_units_)
        break;
      uint32_t e = std::min(s + block_, total_units_);
      invoke(*fn_, s, e);
    }
  }

  void invoke(const std::function<void(uint32_t, uint32_t)>& fn,
              uint32_t first,
              uint32_t last) {
    if (first == last)
      return;
    RowPool* previous = running_;
    running_ = this;
    fn(first, last);
    running_ = previous;
  }

  void worker(uint32_t index) {
    uint32_t local_gen = 0;
    for (;;) {
      bool participate;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        start_cv_.wait(lk, [this, &local_gen] {
          return stop_ || generation_ != local_gen;
        });
        if (stop_)
          return;
        local_gen = generation_;
        participate = index < worker_count_;
      }
      if (!participate)
        continue;
      drain();
      {
        std::lock_guard<std::mutex> lk(mtx_);
        if (--active_ == 0)
          done_cv_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mtx_;
  std::mutex run_mutex_;
  std::condition_variable start_cv_, done_cv_;
  const std::function<void(uint32_t, uint32_t)>* fn_ = nullptr;
  std::atomic<uint32_t> cursor_{0};
  uint32_t total_units_ = 0;
  uint32_t block_ = 1;
  uint32_t generation_ = 0;
  uint32_t worker_count_ = 0;
  uint32_t active_ = 0;
  bool stop_ = false;
  bool enabled_ = true;
  inline static thread_local RowPool* running_ = nullptr;
};

// Linux defines the CIK array/tiling enums and GB_TILE_MODE fields, but not the
// Liverpool table contents or the byte-address equations implemented below:
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_sh_mask.h
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/gfx_v7_0.c

// --- GB_TILE_MODE decode (per tiling_index 0..31), Liverpool / base PS4. ----
// The AddrLib GB_TILE_MODE decode (array mode / micro tile mode / pipe config /
// sample split / tile split per index). We only need a compact form of each.

enum ArrayMode {
  kAmLinearGeneral = 0,
  kAmLinearAligned = 1,
  kAm1DThin1 = 2,
  kAm1DThick = 3,
  kAm2DThin1 = 4,
  kAmPrtThin1 = 5,
  kAmPrt2DThin1 = 6,
  kAm2DThick = 7,
  kAm2DXThick = 8,
  kAmPrtThick = 9,
  kAmPrt2DThick = 10,
  kAmPrt3DThin1 = 11,
  kAm3DThin1 = 12,
  kAm3DThick = 13,
  kAm3DXThick = 14,
  kAmPrt3DThick = 15,
};

enum MicroMode {
  kMmDisplay = 0,
  kMmThin = 1,
  kMmDepth = 2,
  kMmRotated = 3,
  kMmThick = 4
};

// Two 8-pipe pipe-config equations are used by the colour/depth modes on
// Liverpool: P8_32x32_16x16 and P8_32x32_8x16. (P2 only for LinearGeneral.)
enum PipeConfig { kPcP2 = 0, kPcP8_32x32_8x16 = 10, kPcP8_32x32_16x16 = 12 };

ArrayMode ArrayModeOf(uint32_t idx) {
  switch (idx) {
    case 5:   // Depth1DThin
    case 9:   // Display1DThin
    case 13:  // Thin1DThin
      return kAm1DThin1;
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:   // Depth2DThin*
    case 10:  // Display2DThin
    case 14:  // Thin2DThin
      return kAm2DThin1;
    case 11:  // DisplayThinPrt
    case 16:  // ThinThinPrt
      return kAmPrtThin1;
    case 6:   // Depth2DThinPrt256
    case 7:   // Depth2DThinPrt1K
    case 12:  // Display2DThinPrt
    case 17:  // Thin2DThinPrt
      return kAmPrt2DThin1;
    case 15:  // Thin3DThin
      return kAm3DThin1;
    case 18:  // Thin3DThinPrt
      return kAmPrt3DThin1;
    case 19:
      return kAm1DThick;  // Thick1DThick
    case 20:
      return kAm2DThick;  // Thick2DThick
    case 21:
      return kAm3DThick;  // Thick3DThick
    case 22:
      return kAmPrtThick;  // ThickThickPrt
    case 23:
      return kAmPrt2DThick;  // Thick2DThickPrt
    case 24:
      return kAmPrt3DThick;  // Thick3DThickPrt
    case 25:
      return kAm2DXThick;  // Thick2DXThick
    case 26:
      return kAm3DXThick;  // Thick3DXThick
    case 8:
      return kAmLinearAligned;  // DisplayLinearAligned
    case 31:
      return kAmLinearGeneral;  // DisplayLinearGeneral
    default:
      return kAmLinearGeneral;  // reserved; rejected by ValidTileMode()
  }
}

MicroMode MicroModeOf(uint32_t idx) {
  switch (idx) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      return kMmDepth;
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 31:
      return kMmDisplay;
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
      return kMmThin;
    default:  // 19..26
      return kMmThick;
  }
}

PipeConfig PipeConfigOf(uint32_t idx) {
  switch (idx) {
    // P8_32x32_8x16: DisplayThinPrt(11), Thin3DThin(15), ThinThinPrt(16),
    // Thick3DThick(21), ThickThickPrt(22), Thick3DThickPrt(24),
    // Thick3DXThick(26)
    case 11:
    case 15:
    case 16:
    case 21:
    case 22:
    case 24:
    case 26:
      return kPcP8_32x32_8x16;
    case 31:
      return kPcP2;
    default:
      return kPcP8_32x32_16x16;
  }
}

// SAMPLE_SPLIT (sample-split count) per GB_TILE_MODE; 2 for Display2DThin and
// the 2D/3D thin colour modes, 1 elsewhere. Affects the 32bpp tile_split (->
// 512).
uint32_t SampleSplitOf(uint32_t idx) {
  switch (idx) {
    case 10:
    case 11:
    case 12:  // Display2DThin, DisplayThinPrt, Display2DThinPrt
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:  // Thin 2D/3D/Prt
      return 2;
    default:
      return 1;
  }
}

// TILE_SPLIT (hw field) per GB_TILE_MODE; only the depth modes use the large
// values. Colour modes use 64 (then color_tile_split overrides for 32bpp).
uint32_t TileSplitHwOf(uint32_t idx) {
  switch (idx) {
    case 1:
      return 128;  // Depth2DThin128
    case 2:
    case 6:
      return 256;  // Depth2DThin256 / Prt256
    case 3:
      return 512;  // Depth2DThin512
    case 4:
    case 7:
      return 1024;  // Depth2DThin1K / Prt1K
    default:
      return 64;
  }
}

// --- Macro-tile-mode params (AddrLib GetNumBanks/BankWidth/BankHeight/...).
// --- The macro tile mode index for 32bpp/1-sample is derived from tile_split,
// then (per AMD AddrLib) yields num_banks/bank_width/bank_height/macro_aspect.
struct MacroParams {
  uint32_t num_banks, bank_width, bank_height, macro_aspect;
};

// AddrLib macro-tile-mode LUTs, indexed by MacroTileMode (0..15). For the
// non-Neo (base PS4) primary config we use the non-alt tables.
MacroParams MacroParamsForMode(uint32_t m) {
  // {num_banks, bank_width, bank_height, macro_aspect}
  static const MacroParams tbl[16] = {
      /*0  Mode_1x4_16     */ {16, 1, 4, 4},
      /*1  Mode_1x2_16     */ {16, 1, 2, 2},
      /*2  Mode_1x1_16     */ {16, 1, 1, 2},
      /*3  Mode_1x1_16_Dup */ {16, 1, 1, 2},
      /*4  Mode_1x1_8      */ {8, 1, 1, 1},
      /*5  Mode_1x1_4      */ {4, 1, 1, 1},
      /*6  Mode_1x1_2      */ {2, 1, 1, 1},
      /*7  Mode_1x1_2_Dup  */ {2, 1, 1, 1},
      /*8  Mode_1x8_16     */ {16, 1, 8, 4},
      /*9  Mode_1x4_16_Dup */ {16, 1, 4, 4},
      /*10 Mode_1x2_16_Dup */ {16, 1, 2, 2},
      /*11 Mode_1x1_16_Dup2*/ {16, 1, 1, 2},
      /*12 Mode_1x1_8_Dup  */ {8, 1, 1, 1},
      /*13 Mode_1x1_4_Dup  */ {4, 1, 1, 1},
      /*14 Mode_1x1_2_Dup2 */ {2, 1, 1, 1},
      /*15 Mode_1x1_2_Dup3 */ {2, 1, 1, 1},
  };
  return tbl[m & 15];
}

constexpr uint32_t kMicroW = 8, kMicroH = 8;
constexpr uint32_t kMicroTilePixels = kMicroW * kMicroH;
constexpr uint32_t kPipeInterleaveBits = 8;

// gfx10.3 (RDNA2) "standard" swizzle, encoded as kGfx10StdBase + log2 of the
// block size in KiB (256 B, 4 KiB, 64 KiB). Prospero titles describe their
// textures with these, not with Liverpool's GB_TILE_MODE indices, so they get
// their own id range rather than being squeezed into that table.
constexpr uint32_t kGfx10StdBase = 0x50;
constexpr uint32_t kGfx10StdModes = 3;

bool TilingIsGfx10Std(uint32_t idx) {
  return idx >= kGfx10StdBase && idx < kGfx10StdBase + kGfx10StdModes;
}

// Micro-tiles are always 256 B; their shape follows the element size.
void Gfx10MicroShape(uint32_t elem_bytes, uint32_t& mw, uint32_t& mh) {
  switch (elem_bytes) {
    case 1:
      mw = 16;
      mh = 16;
      break;
    case 2:
      mw = 16;
      mh = 8;
      break;
    case 4:
      mw = 8;
      mh = 8;
      break;
    case 8:
      mw = 8;
      mh = 4;
      break;
    default:
      mw = 4;
      mh = 4;
      break;  // 16
  }
}

// Micro-tiles per block edge: 1 (256 B), 4 (4 KiB) or 16 (64 KiB).
uint32_t Gfx10BlockTiles(uint32_t idx) {
  switch (idx - kGfx10StdBase) {
    case 0:
      return 1;
    case 1:
      return 4;
    default:
      return 16;
  }
}

bool ValidTileMode(uint32_t idx) {
  return idx <= 26 || idx == 31 || TilingIsGfx10Std(idx);
}

uint32_t EffectiveTileMode(uint32_t idx) {
  if (idx == 25)
    return 20;  // 2D XThick -> 2D Thick at 32bpp
  if (idx == 26)
    return 21;  // 3D XThick -> 3D Thick at 32bpp
  return idx;
}

uint32_t TileThickness(ArrayMode am) {
  switch (am) {
    case kAm1DThick:
    case kAm2DThick:
    case kAmPrtThick:
    case kAmPrt2DThick:
    case kAm3DThick:
    case kAmPrt3DThick:
      return 4;
    case kAm2DXThick:
    case kAm3DXThick:
      return 8;
    default:
      return 1;
  }
}

bool IsMacroTiled(ArrayMode am) {
  return am != kAmLinearGeneral && am != kAmLinearAligned && am != kAm1DThin1 &&
         am != kAm1DThick;
}

uint32_t NumPipesOf(PipeConfig pc) {
  return pc == kPcP2 ? 2u : 8u;
}
uint32_t NumPipeBitsOf(PipeConfig pc) {
  return pc == kPcP2 ? 1u : 3u;
}

bool IsPrt(ArrayMode am) {
  return am == kAmPrtThin1 || am == kAmPrtThick || am == kAmPrt2DThin1 ||
         am == kAmPrt2DThick || am == kAmPrt3DThin1 || am == kAmPrt3DThick;
}

// Effective tile size used to select the macro-tile table. Thick microtiles may
// exceed the 1 KiB DRAM-row split, but unlike thin multisample tiles they are
// not split into virtual slices by the address equation.
uint32_t TileSizeBytes(uint32_t idx,
                       MicroMode mm,
                       uint32_t thickness,
                       uint32_t elem) {
  const uint32_t tile_bytes_1x = kMicroTilePixels * elem * thickness;
  const uint32_t color_split = SampleSplitOf(idx) * tile_bytes_1x;
  const uint32_t cts = color_split < 256u ? 256u : color_split;
  uint32_t split = (mm == kMmDepth) ? TileSplitHwOf(idx) : cts;
  if (split > 1024u)
    split = 1024u;
  if (split > tile_bytes_1x)
    split = tile_bytes_1x;
  return split;
}

// CalculateMacrotileMode: mtm = log2(tile_split/64); +8 if PRT.
uint32_t MacroTileModeIndex(uint32_t idx,
                            MicroMode mm,
                            ArrayMode am,
                            uint32_t thickness,
                            uint32_t elem) {
  uint32_t split = TileSizeBytes(idx, mm, thickness, elem);
  uint32_t q = split / 64u;
  uint32_t mtm = 0;
  while ((1u << (mtm + 1)) <= q)
    mtm++;  // bit_width(q)-1
  if (IsPrt(am))
    mtm += 8;
  return mtm;
}

// Element index within an 8x8x{1,4,8} microtile.
inline uint32_t PixIdx(uint32_t x,
                       uint32_t y,
                       uint32_t z,
                       MicroMode m,
                       uint32_t thickness,
                       uint32_t elem) {
  uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
  uint32_t y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
  if (m == kMmDisplay) {
    if (elem == 2)
      return x0 | (x1 << 1) | (x2 << 2) | (y0 << 3) | (y1 << 4) | (y2 << 5);
    if (elem == 4)
      return x0 | (x1 << 1) | (y0 << 2) | (x2 << 3) | (y1 << 4) | (y2 << 5);
    if (elem == 8)
      return x0 | (y0 << 1) | (x1 << 2) | (x2 << 3) | (y1 << 4) | (y2 << 5);
    return y0 | (x0 << 1) | (x1 << 2) | (x2 << 3) | (y1 << 4) | (y2 << 5);
  }
  if (m != kMmThick)
    return x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (x2 << 4) | (y2 << 5);
  uint32_t z0 = z & 1, z1 = (z >> 1) & 1;
  uint32_t index;
  if (elem <= 2)
    index = x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (z0 << 4) | (z1 << 5) |
            (x2 << 6) | (y2 << 7);
  else if (elem == 4)
    index = x0 | (y0 << 1) | (x1 << 2) | (z0 << 3) | (y1 << 4) | (z1 << 5) |
            (x2 << 6) | (y2 << 7);
  else
    index = x0 | (y0 << 1) | (z0 << 2) | (x1 << 3) | (y1 << 4) | (z1 << 5) |
            (x2 << 6) | (y2 << 7);
  if (thickness == 8)
    index |= ((z >> 2) & 1) << 8;
  return index;
}

// ComputePipeFromCoord (tiling.comp), 8-pipe configs.
inline uint32_t PipeFromCoord(uint32_t x,
                              uint32_t y,
                              uint32_t slice,
                              PipeConfig pc,
                              ArrayMode am,
                              uint32_t thickness) {
  uint32_t tx = x >> 3, ty = y >> 3;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
  uint32_t pipe;
  if (pc == kPcP2)
    pipe = x3 ^ y3;
  else if (pc == kPcP8_32x32_8x16) {
    uint32_t p0 = x4 ^ y3 ^ x5;
    uint32_t p1 = x3 ^ y4;
    uint32_t p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  } else {
    // P8_32x32_16x16
    uint32_t p0 = x3 ^ y3 ^ x4;
    uint32_t p1 = x4 ^ y4;
    uint32_t p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  }
  if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick) {
    uint32_t rotation =
        std::max(1u, NumPipesOf(pc) / 2 - 1) * (slice / thickness);
    pipe ^= rotation & (NumPipesOf(pc) - 1);
  }
  return pipe;
}

// ComputeBankFromCoord (tiling.comp), parameterized by num_banks/widths.
inline uint32_t BankFromCoord(uint32_t x,
                              uint32_t y,
                              uint32_t slice,
                              const MacroParams& mp,
                              uint32_t num_pipes,
                              ArrayMode am,
                              uint32_t thickness,
                              uint32_t tile_split_slice) {
  uint32_t tx = (x >> 3) / (mp.bank_width * num_pipes);
  uint32_t ty = (y >> 3) / mp.bank_height;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1,
           x6 = (tx >> 3) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1,
           y6 = (ty >> 3) & 1;
  uint32_t bank = 0;
  switch (mp.num_banks) {
    case 16:
      bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) |
             ((x6 ^ y3) << 3);
      break;
    case 8:
      bank = (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2);
      break;
    case 4:
      bank = (x3 ^ y4) | ((x4 ^ y3) << 1);
      break;
    case 2:
      bank = (x3 ^ y3);
      break;
    default:
      bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) |
             ((x6 ^ y3) << 3);
      break;
  }
  uint32_t rotation = 0;
  if (am == kAm2DThin1 || am == kAm2DThick || am == kAm2DXThick)
    rotation = (mp.num_banks / 2 - 1) * (slice / thickness);
  else if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick)
    rotation =
        std::max(1u, num_pipes / 2 - 1) * (slice / thickness) / num_pipes;
  uint32_t tile_split_rotation = 0;
  if (am == kAm2DThin1 || am == kAm3DThin1 || am == kAmPrt2DThin1 ||
      am == kAmPrt3DThin1)
    tile_split_rotation = (mp.num_banks / 2 + 1) * tile_split_slice;
  return (bank ^ rotation ^ tile_split_rotation) & (mp.num_banks - 1);
}

struct Macro2D {
  ArrayMode am;
  MicroMode mm;
  PipeConfig pc;
  MacroParams mp;
  uint32_t num_pipes, num_pipe_bits, num_bank_bits;
  uint32_t thickness, micro_tile_bytes, slices_per_tile;
  uint32_t macro_pitch, macro_height, macro_tile_bytes, base_align;
};

inline uint32_t NumBankBitsOf(uint32_t num_banks) {
  uint32_t b = 0;
  while ((1u << (b + 1)) <= num_banks)
    b++;
  return b;  // bit_width(num_banks)-1
}

inline uint64_t SwizzleMacroOffset(uint64_t total_offset,
                                   uint32_t pipe,
                                   uint32_t bank,
                                   const Macro2D& c) {
  const uint64_t interleave_offset =
      total_offset & ((1u << kPipeInterleaveBits) - 1);
  const uint64_t offset = total_offset >> kPipeInterleaveBits;
  return interleave_offset |
         (static_cast<uint64_t>(pipe) << kPipeInterleaveBits) |
         (static_cast<uint64_t>(bank)
          << (kPipeInterleaveBits + c.num_pipe_bits)) |
         (offset << (kPipeInterleaveBits + c.num_pipe_bits + c.num_bank_bits));
}

bool ConfigureMacro2D(uint32_t tiling_idx,
                      ArrayMode am,
                      MicroMode mm,
                      uint32_t elem,
                      Macro2D& c) {
  c.am = am;
  c.mm = mm;
  c.pc = PipeConfigOf(tiling_idx);
  c.num_pipes = NumPipesOf(c.pc);
  c.num_pipe_bits = NumPipeBitsOf(c.pc);
  c.thickness = TileThickness(am);
  const uint32_t full_micro_tile_bytes = kMicroTilePixels * elem * c.thickness;
  const uint32_t tile_split_bytes =
      TileSizeBytes(tiling_idx, mm, c.thickness, elem);
  c.micro_tile_bytes = full_micro_tile_bytes;
  c.slices_per_tile = 1;
  // AddrLib addresses each split portion as a virtual slice with its own bank
  // rotation.
  if (full_micro_tile_bytes > tile_split_bytes && c.thickness == 1) {
    c.micro_tile_bytes = tile_split_bytes;
    c.slices_per_tile = full_micro_tile_bytes / tile_split_bytes;
  }
  uint32_t mtm = MacroTileModeIndex(tiling_idx, mm, am, c.thickness, elem);
  c.mp = MacroParamsForMode(mtm);
  c.num_bank_bits = NumBankBitsOf(c.mp.num_banks);
  c.macro_pitch = kMicroW * c.mp.bank_width * c.num_pipes * c.mp.macro_aspect;
  c.macro_height =
      kMicroH * c.mp.bank_height * c.mp.num_banks / c.mp.macro_aspect;
  c.macro_tile_bytes = c.micro_tile_bytes * (c.macro_pitch / kMicroW) *
                       (c.macro_height / kMicroH) /
                       (c.num_pipes * c.mp.num_banks);
  c.base_align = c.num_pipes * c.mp.bank_width * c.mp.num_banks *
                 c.mp.bank_height *
                 TileSizeBytes(tiling_idx, mm, c.thickness, elem);
  return c.macro_pitch && c.macro_height;
}

uint32_t AlignUp(uint32_t value, uint32_t align) {
  return (value + align - 1) & ~(align - 1);
}

uint64_t AlignUp(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

uint32_t BitCeil(uint32_t value) {
  if (value <= 1)
    return 1;
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1;
}

template <uint32_t Elem, bool Detile>
inline void CopyElement(uint8_t* tiled, uint8_t* linear) {
  if constexpr (Detile)
    std::memcpy(linear, tiled, Elem);
  else
    std::memcpy(tiled, linear, Elem);
}

template <uint32_t Elem, bool Detile>
void CopyLinearMip(uint8_t* tiled,
                   uint8_t* linear,
                   const TextureMipLayout32& level,
                   uint32_t layer,
                   size_t linear_row_bytes) {
  tiled +=
      static_cast<uint64_t>(layer) * level.pitch * level.stored_height * Elem;
  const size_t logical_row_bytes = static_cast<size_t>(level.width) * Elem;
  if (level.pitch == level.width && linear_row_bytes == logical_row_bytes) {
    const size_t bytes = logical_row_bytes * level.height;
    if constexpr (Detile)
      std::memcpy(linear, tiled, bytes);
    else
      std::memcpy(tiled, linear, bytes);
    return;
  }

  RowPool::get().run(
      level.height, static_cast<uint64_t>(level.width) * level.height,
      [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; ++y) {
          uint8_t* tiled_row =
              tiled + static_cast<size_t>(y) * level.pitch * Elem;
          uint8_t* linear_row =
              linear + static_cast<size_t>(y) * linear_row_bytes;
          if constexpr (Detile)
            std::memcpy(linear_row, tiled_row, logical_row_bytes);
          else
            std::memcpy(tiled_row, linear_row, logical_row_bytes);
        }
      });
}

template <uint32_t Elem, bool Detile>
void CopyMicroTiledMip(uint8_t* tiled,
                       uint8_t* linear,
                       const TextureMipLayout32& level,
                       uint32_t layer,
                       size_t linear_row_bytes,
                       MicroMode mm) {
  std::array<uint32_t, kMicroTilePixels> element_offsets{};
  for (uint32_t y = 0; y < kMicroH; ++y)
    for (uint32_t x = 0; x < kMicroW; ++x)
      element_offsets[y * kMicroW + x] =
          PixIdx(x, y, layer, mm, level.thickness, Elem) * Elem;

  const uint32_t tile_columns = (level.width + kMicroW - 1) / kMicroW;
  const uint32_t tile_rows = (level.height + kMicroH - 1) / kMicroH;
  const uint32_t physical_tiles_per_row = level.pitch / kMicroW;
  const uint64_t micro_tile_bytes =
      static_cast<uint64_t>(kMicroTilePixels) * Elem * level.thickness;
  const uint64_t group_bytes = static_cast<uint64_t>(level.pitch) *
                               level.stored_height * level.thickness * Elem;
  const uint64_t slice_offset = (layer / level.thickness) * group_bytes;

  RowPool::get().run(
      tile_rows, static_cast<uint64_t>(level.width) * level.height,
      [&](uint32_t tile_y0, uint32_t tile_y1) {
        for (uint32_t tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {
          const uint32_t first_y = tile_y * kMicroH;
          const uint32_t copy_height =
              std::min(kMicroH, level.height - first_y);
          for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const uint32_t first_x = tile_x * kMicroW;
            const uint32_t copy_width =
                std::min(kMicroW, level.width - first_x);
            const uint64_t tile_offset =
                slice_offset +
                (static_cast<uint64_t>(tile_y) * physical_tiles_per_row +
                 tile_x) *
                    micro_tile_bytes;
            for (uint32_t y = 0; y < copy_height; ++y) {
              uint8_t* linear_row =
                  linear + static_cast<size_t>(first_y + y) * linear_row_bytes +
                  static_cast<size_t>(first_x) * Elem;
              for (uint32_t x = 0; x < copy_width; ++x)
                CopyElement<Elem, Detile>(
                    tiled + tile_offset + element_offsets[y * kMicroW + x],
                    linear_row + static_cast<size_t>(x) * Elem);
            }
          }
        }
      });
}

template <uint32_t Elem, bool Detile, bool Split>
void CopyMacroTiledMip(uint8_t* tiled,
                       uint8_t* linear,
                       const TextureMipLayout32& level,
                       uint32_t layer,
                       size_t linear_row_bytes,
                       const Macro2D& c) {
  std::array<uint32_t, kMicroTilePixels> element_offsets{};
  std::array<uint16_t, kMicroTilePixels> split_slices{};
  for (uint32_t y = 0; y < kMicroH; ++y) {
    for (uint32_t x = 0; x < kMicroW; ++x) {
      const uint32_t i = y * kMicroW + x;
      const uint32_t raw_offset =
          PixIdx(x, y, layer, c.mm, c.thickness, Elem) * Elem;
      if constexpr (Split) {
        split_slices[i] =
            static_cast<uint16_t>(raw_offset / c.micro_tile_bytes);
        element_offsets[i] = raw_offset % c.micro_tile_bytes;
      } else {
        element_offsets[i] = raw_offset;
      }
    }
  }

  const uint32_t tile_columns = (level.width + kMicroW - 1) / kMicroW;
  const uint32_t tile_rows = (level.height + kMicroH - 1) / kMicroH;
  const uint32_t macro_tiles_per_row = level.pitch / c.macro_pitch;
  const uint32_t macro_tiles_per_slice =
      macro_tiles_per_row * (level.stored_height / c.macro_height);
  const uint64_t slice_bytes =
      static_cast<uint64_t>(macro_tiles_per_slice) * c.macro_tile_bytes;
  const uint64_t slice_group =
      static_cast<uint64_t>(c.slices_per_tile) * (layer / c.thickness);

  RowPool::get().run(
      tile_rows, static_cast<uint64_t>(level.width) * level.height,
      [&](uint32_t tile_y0, uint32_t tile_y1) {
        for (uint32_t tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {
          const uint32_t first_y = tile_y * kMicroH;
          const uint32_t copy_height =
              std::min(kMicroH, level.height - first_y);
          const uint64_t macro_row =
              static_cast<uint64_t>(first_y / c.macro_height) *
              macro_tiles_per_row;
          const uint32_t tile_row = tile_y % c.mp.bank_height;
          for (uint32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const uint32_t first_x = tile_x * kMicroW;
            const uint32_t copy_width =
                std::min(kMicroW, level.width - first_x);
            const uint64_t macro_tile_offset =
                (macro_row + first_x / c.macro_pitch) * c.macro_tile_bytes;
            const uint32_t tile_col = (tile_x / c.num_pipes) % c.mp.bank_width;
            const uint64_t tile_offset =
                static_cast<uint64_t>(tile_row * c.mp.bank_width + tile_col) *
                c.micro_tile_bytes;
            const uint64_t common_offset = macro_tile_offset + tile_offset;

            uint32_t swizzle_x = first_x;
            uint32_t swizzle_y = first_y;
            if (c.am == kAmPrtThin1 || c.am == kAmPrtThick) {
              swizzle_x %= c.macro_pitch;
              swizzle_y %= c.macro_height;
            }
            const uint32_t pipe = PipeFromCoord(swizzle_x, swizzle_y, layer,
                                                c.pc, c.am, c.thickness);

            std::array<uint64_t, 16> split_bases{};
            std::array<uint32_t, 16> split_banks{};
            uint64_t base = 0;
            uint32_t bank = 0;
            if constexpr (Split) {
              for (uint32_t split = 0; split < c.slices_per_tile; ++split) {
                split_bases[split] =
                    slice_bytes * (slice_group + split) + common_offset;
                split_banks[split] =
                    BankFromCoord(swizzle_x, swizzle_y, layer, c.mp,
                                  c.num_pipes, c.am, c.thickness, split);
              }
            } else {
              base = slice_bytes * slice_group + common_offset;
              bank = BankFromCoord(swizzle_x, swizzle_y, layer, c.mp,
                                   c.num_pipes, c.am, c.thickness, 0);
            }

            for (uint32_t y = 0; y < copy_height; ++y) {
              uint8_t* linear_row =
                  linear + static_cast<size_t>(first_y + y) * linear_row_bytes +
                  static_cast<size_t>(first_x) * Elem;
              for (uint32_t x = 0; x < copy_width; ++x) {
                const uint32_t i = y * kMicroW + x;
                uint64_t tiled_offset;
                if constexpr (Split) {
                  const uint32_t split = split_slices[i];
                  tiled_offset = SwizzleMacroOffset(
                      split_bases[split] + element_offsets[i], pipe,
                      split_banks[split], c);
                } else {
                  tiled_offset = SwizzleMacroOffset(base + element_offsets[i],
                                                    pipe, bank, c);
                }
                CopyElement<Elem, Detile>(
                    tiled + tiled_offset,
                    linear_row + static_cast<size_t>(x) * Elem);
              }
            }
          }
        }
      });
}

// gfx10.3 standard swizzle: 256 B micro-tiles stored row-major internally, the
// micro-tiles of a block interleaved in Morton order (x bit first), and the
// blocks themselves row-major across the surface.
template <uint32_t Elem, bool Detile>
void CopyGfx10StdMip(uint8_t* tiled,
                     uint8_t* linear,
                     const TextureMipLayout32& level,
                     uint32_t layer,
                     size_t linear_row_bytes,
                     uint32_t tiling_idx) {
  uint32_t mw = 0, mh = 0;
  Gfx10MicroShape(Elem, mw, mh);
  const uint32_t n = Gfx10BlockTiles(tiling_idx);
  const uint32_t bw = mw * n, bh = mh * n;
  const uint32_t blocks_per_row = level.pitch / bw;
  const uint64_t block_bytes = static_cast<uint64_t>(bw) * bh * Elem;
  const uint64_t slice_bytes =
      static_cast<uint64_t>(level.pitch) * level.stored_height * Elem;
  uint8_t* slice = tiled + slice_bytes * layer;
  // Within a block: the 256 B micro-tile has its own bit pattern, and above it
  // the micro-tiles interleave Y first. For 32bpp the micro-tile is 8x8 laid
  // out x0,x1,y0,y1,y2,x2 -- a 4x4 row-major quadrant, quadrants ordered Y
  // first. Solved from the surfaces themselves: a 64 KiB block only ever holds
  // non-zero elements inside the WxH image, and over two dozen differently
  // shaped Minecraft UI surfaces this is the only bit order that never places a
  // non-zero element outside one.
  uint32_t x_bits = 0, y_bits = 0;
  while ((1u << x_bits) < bw)
    x_bits++;
  while ((1u << y_bits) < bh)
    y_bits++;
  uint32_t micro_src[6] = {}, micro_is_x[6] = {}, micro_n = 0;
  if constexpr (Elem == 4) {
    const uint32_t sx[6] = {0, 1, 0, 1, 2, 2};
    const uint32_t is_x[6] = {1, 1, 0, 0, 0, 1};
    for (uint32_t i = 0; i < 6; i++) {
      micro_src[i] = sx[i];
      micro_is_x[i] = is_x[i];
    }
    micro_n = 6;
  }
  DetileParallelRows(level.height, [&](uint32_t y0, uint32_t y1) {
    for (uint32_t y = y0; y < y1; y++) {
      uint8_t* linear_row = linear + static_cast<size_t>(y) * linear_row_bytes;
      const uint32_t by = y / bh, iy = y % bh;
      for (uint32_t x = 0; x < level.width; x++) {
        const uint32_t bx = x / bw, ix = x % bw;
        uint32_t element = 0, bit = 0, xb = 0, yb = 0;
        for (uint32_t i = 0; i < micro_n; i++) {
          const uint32_t v = micro_is_x[i] ? ix : iy;
          element |= ((v >> micro_src[i]) & 1) << bit++;
          (micro_is_x[i] ? xb : yb)++;
        }
        while (xb < x_bits || yb < y_bits) {
          if (yb < y_bits)
            element |= ((iy >> yb++) & 1) << bit++;
          if (xb < x_bits)
            element |= ((ix >> xb++) & 1) << bit++;
        }
        const uint64_t off =
            (static_cast<uint64_t>(by) * blocks_per_row + bx) * block_bytes +
            static_cast<uint64_t>(element) * Elem;
        uint8_t* t = slice + off;
        uint8_t* l = linear_row + static_cast<size_t>(x) * Elem;
        if (Detile)
          std::memcpy(l, t, Elem);
        else
          std::memcpy(t, l, Elem);
      }
    }
  });
}

template <uint32_t Elem, bool Detile>
bool CopyTextureMip(uint8_t* tiled_image,
                    uint8_t* linear,
                    size_t linear_row_bytes,
                    const TextureLayout32& layout,
                    uint32_t mip,
                    uint32_t layer) {
  const TextureMipLayout32& level = layout.mips[mip];
  uint8_t* tiled = tiled_image + level.offset;
  if (TilingIsLinear(layout.tiling_idx)) {
    CopyLinearMip<Elem, Detile>(tiled, linear, level, layer, linear_row_bytes);
    return true;
  }

  if (TilingIsGfx10Std(layout.tiling_idx)) {
    CopyGfx10StdMip<Elem, Detile>(tiled, linear, level, layer, linear_row_bytes,
                                  layout.tiling_idx);
    return true;
  }

  const ArrayMode am = ArrayModeOf(layout.tiling_idx);
  const MicroMode mm = MicroModeOf(layout.tiling_idx);
  if (!level.macro_tiled) {
    CopyMicroTiledMip<Elem, Detile>(tiled, linear, level, layer,
                                    linear_row_bytes, mm);
    return true;
  }

  Macro2D macro{};
  if (!ConfigureMacro2D(layout.tiling_idx, am, mm, Elem, macro) ||
      macro.slices_per_tile > 16)
    return false;
  if (macro.slices_per_tile == 1)
    CopyMacroTiledMip<Elem, Detile, false>(tiled, linear, level, layer,
                                           linear_row_bytes, macro);
  else
    CopyMacroTiledMip<Elem, Detile, true>(tiled, linear, level, layer,
                                          linear_row_bytes, macro);
  return true;
}

}  // namespace

void DetileParallelRows(uint32_t rows,
                        const std::function<void(uint32_t, uint32_t)>& fn) {
  RowPool::get().run(rows, static_cast<uint64_t>(rows) * 1024, fn);
}

bool TilingIsLinear(uint32_t tiling_idx) {
  // Per the Liverpool GB_TILE_MODE table only DisplayLinearAligned(8) and
  // DisplayLinearGeneral(31) are genuinely linear (ArrayLinearAligned/General).
  // Keep the same set the previous code used so Isaac's linear surfaces and
  // small UI textures are still straight-copied.
  return tiling_idx == 8 || tiling_idx == 31;
}

bool BuildTextureLayout32(TextureLayout32& out,
                          uint32_t width,
                          uint32_t height,
                          uint32_t pitch,
                          uint32_t layers,
                          uint32_t mip_levels,
                          uint32_t tiling_idx,
                          bool pow2_pad,
                          uint32_t elem_bytes) {
  out = {};
  if (!width || !height || !layers || !mip_levels ||
      mip_levels > out.mips.size() || width > 16384 || height > 16384 ||
      pitch > 16384 || layers > 8192 || !ValidTileMode(tiling_idx))
    return false;
  if (elem_bytes != 1 && elem_bytes != 2 && elem_bytes != 4 &&
      elem_bytes != 8 && elem_bytes != 16)
    return false;

  tiling_idx = EffectiveTileMode(tiling_idx);
  out.mip_levels = mip_levels;
  out.layers = layers;
  out.tiling_idx = tiling_idx;
  out.elem_bytes = elem_bytes;
  const ArrayMode am = ArrayModeOf(tiling_idx);
  const MicroMode mm = MicroModeOf(tiling_idx);
  const uint32_t thickness = TileThickness(am);
  Macro2D macro{};
  if (IsMacroTiled(am) &&
      !ConfigureMacro2D(tiling_idx, am, mm, elem_bytes, macro))
    return false;

  uint64_t end = 0;
  for (uint32_t mip = 0; mip < mip_levels; mip++) {
    TextureMipLayout32& level = out.mips[mip];
    level.width = std::max(width >> mip, 1u);
    level.height = std::max(height >> mip, 1u);
    uint32_t raw_pitch = std::max(std::max(pitch >> mip, 1u), level.width);
    uint32_t storage_height = level.height;
    if (pow2_pad) {
      raw_pitch = BitCeil(raw_pitch);
      storage_height = BitCeil(storage_height);
    }
    level.thickness = thickness;

    uint64_t base_align = 1;
    if (TilingIsGfx10Std(tiling_idx)) {
      // Each mip is padded to whole blocks and starts on a block boundary.
      uint32_t mw = 0, mh = 0;
      Gfx10MicroShape(elem_bytes, mw, mh);
      const uint32_t n = Gfx10BlockTiles(tiling_idx);
      const uint32_t bw = mw * n, bh = mh * n;
      level.pitch = AlignUp(raw_pitch, bw);
      level.stored_height = AlignUp(storage_height, bh);
      base_align = static_cast<uint64_t>(bw) * bh * elem_bytes;
      level.macro_tiled = false;
    } else if (tiling_idx == 31) {
      level.pitch = raw_pitch;
      level.stored_height = storage_height;
    } else if (tiling_idx == 8) {
      level.pitch = AlignUp(raw_pitch, 16);
      level.stored_height = storage_height;
      while ((static_cast<uint64_t>(level.pitch) * level.stored_height) % 64)
        level.pitch += 16;
      base_align = 256;
    } else {
      level.macro_tiled = IsMacroTiled(am) &&
                          !(mip > 0 && (raw_pitch < macro.macro_pitch ||
                                        storage_height < macro.macro_height));
      if (level.macro_tiled) {
        level.pitch = AlignUp(raw_pitch, macro.macro_pitch);
        level.stored_height = AlignUp(storage_height, macro.macro_height);
        base_align = macro.base_align;
      } else {
        level.pitch = AlignUp(raw_pitch, kMicroW);
        level.stored_height = AlignUp(storage_height, kMicroH);
        base_align = 256;
      }
    }

    const uint32_t stored_layers = AlignUp(layers, thickness);
    level.offset = AlignUp(end, base_align);
    level.size = static_cast<uint64_t>(level.pitch) * level.stored_height *
                 stored_layers * elem_bytes;
    if (!level.size || level.offset > UINT64_MAX - level.size)
      return false;
    end = level.offset + level.size;
  }
  out.size = end;
  return out.size != 0;
}

bool DetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        uint32_t mip,
                        uint32_t layer) {
  if (mip >= layout.mip_levels)
    return false;
  return DetileTextureMip32Pitched(
      src, dst, static_cast<size_t>(layout.mips[mip].width) * layout.elem_bytes,
      layout, mip, layer);
}

bool DetileTextureMip32Pitched(const void* src,
                               void* dst,
                               size_t dst_row_bytes,
                               const TextureLayout32& layout,
                               uint32_t mip,
                               uint32_t layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32& level = layout.mips[mip];
  if (dst_row_bytes < static_cast<size_t>(level.width) * layout.elem_bytes ||
      (level.height > 1 && dst_row_bytes > SIZE_MAX / (level.height - 1)))
    return false;
  auto* tiled = const_cast<uint8_t*>(static_cast<const uint8_t*>(src));
  auto* linear = static_cast<uint8_t*>(dst);
  switch (layout.elem_bytes) {
    case 1:
      return CopyTextureMip<1, true>(tiled, linear, dst_row_bytes, layout, mip,
                                     layer);
    case 2:
      return CopyTextureMip<2, true>(tiled, linear, dst_row_bytes, layout, mip,
                                     layer);
    case 4:
      return CopyTextureMip<4, true>(tiled, linear, dst_row_bytes, layout, mip,
                                     layer);
    case 8:
      return CopyTextureMip<8, true>(tiled, linear, dst_row_bytes, layout, mip,
                                     layer);
    case 16:
      return CopyTextureMip<16, true>(tiled, linear, dst_row_bytes, layout, mip,
                                      layer);
    default:
      return false;
  }
}

bool RetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        uint32_t mip,
                        uint32_t layer) {
  if (mip >= layout.mip_levels)
    return false;
  return RetileTextureMip32Pitched(
      src, static_cast<size_t>(layout.mips[mip].width) * layout.elem_bytes, dst,
      layout, mip, layer);
}

bool RetileTextureMip32Pitched(const void* src,
                               size_t src_row_bytes,
                               void* dst,
                               const TextureLayout32& layout,
                               uint32_t mip,
                               uint32_t layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32& level = layout.mips[mip];
  if (src_row_bytes < static_cast<size_t>(level.width) * layout.elem_bytes ||
      (level.height > 1 && src_row_bytes > SIZE_MAX / (level.height - 1)))
    return false;
  auto* tiled = static_cast<uint8_t*>(dst);
  auto* linear = const_cast<uint8_t*>(static_cast<const uint8_t*>(src));
  switch (layout.elem_bytes) {
    case 2:
      return CopyTextureMip<2, false>(tiled, linear, src_row_bytes, layout, mip,
                                      layer);
    case 4:
      return CopyTextureMip<4, false>(tiled, linear, src_row_bytes, layout, mip,
                                      layer);
    case 8:
      return CopyTextureMip<8, false>(tiled, linear, src_row_bytes, layout, mip,
                                      layer);
    case 16:
      return CopyTextureMip<16, false>(tiled, linear, src_row_bytes, layout,
                                       mip, layer);
    default:
      return false;
  }
}

}  // namespace gpu::gcn
