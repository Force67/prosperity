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
#include "base/arch.h"

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

  void run(u32 units,
           u64 work_items,
           const std::function<void(u32, u32)>& fn) {
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
      const u64 useful_lanes = std::max<u64>(
          2, (work_items + kItemsPerLane - 1) / kItemsPerLane);
      worker_count_ = std::min<u32>(
          {static_cast<u32>(threads_.size()), units - 1,
           static_cast<u32>(std::min<u64>(
               useful_lanes - 1, static_cast<u64>(UINT32_MAX)))});
      // Aim for several chunks per lane so stealing balances uneven units.
      const u32 lanes = worker_count_ + 1;
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
  static constexpr u64 kMinParallelItems = 32 * 1024;
  static constexpr u64 kItemsPerLane = 32 * 1024;

  RowPool() {
    enabled_ = kDetileMt;
    u32 hw = std::thread::hardware_concurrency();
    u32 n = std::min(8u, hw ? hw / 2u : 1u);
    if (const char* configured = kDetileThreads) {
      char* end = nullptr;
      const unsigned long lanes = std::strtoul(configured, &end, 10);
      if (end != configured && !*end)
        n = lanes > 1 ? static_cast<u32>(std::min(lanes - 1, 63ul)) : 0;
    }
    if (!enabled_)
      n = 0;
    for (u32 i = 0; i < n; i++)
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
      u32 s = cursor_.fetch_add(block_, std::memory_order_relaxed);
      if (s >= total_units_)
        break;
      u32 e = std::min(s + block_, total_units_);
      invoke(*fn_, s, e);
    }
  }

  void invoke(const std::function<void(u32, u32)>& fn,
              u32 first,
              u32 last) {
    if (first == last)
      return;
    RowPool* previous = running_;
    running_ = this;
    fn(first, last);
    running_ = previous;
  }

  void worker(u32 index) {
    u32 local_gen = 0;
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
  const std::function<void(u32, u32)>* fn_ = nullptr;
  std::atomic<u32> cursor_{0};
  u32 total_units_ = 0;
  u32 block_ = 1;
  u32 generation_ = 0;
  u32 worker_count_ = 0;
  u32 active_ = 0;
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

ArrayMode ArrayModeOf(u32 idx) {
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

MicroMode MicroModeOf(u32 idx) {
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

PipeConfig PipeConfigOf(u32 idx) {
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
u32 SampleSplitOf(u32 idx) {
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
u32 TileSplitHwOf(u32 idx) {
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
  u32 num_banks, bank_width, bank_height, macro_aspect;
};

// AddrLib macro-tile-mode LUTs, indexed by MacroTileMode (0..15). For the
// non-Neo (base PS4) primary config we use the non-alt tables.
MacroParams MacroParamsForMode(u32 m) {
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

constexpr u32 kMicroW = 8, kMicroH = 8;
constexpr u32 kMicroTilePixels = kMicroW * kMicroH;
constexpr u32 kPipeInterleaveBits = 8;

// gfx10.3 (RDNA2) "standard" swizzle, encoded as kGfx10StdBase + log2 of the
// block size in KiB (256 B, 4 KiB, 64 KiB). Prospero titles describe their
// textures with these, not with Liverpool's GB_TILE_MODE indices, so they get
// their own id range rather than being squeezed into that table.
constexpr u32 kGfx10StdBase = 0x50;
constexpr u32 kGfx10StdModes = 3;

bool TilingIsGfx10Std(u32 idx) {
  return idx >= kGfx10StdBase && idx < kGfx10StdBase + kGfx10StdModes;
}

// Micro-tiles are always 256 B; their shape follows the element size.
void Gfx10MicroShape(u32 elem_bytes, u32& mw, u32& mh) {
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
u32 Gfx10BlockTiles(u32 idx) {
  switch (idx - kGfx10StdBase) {
    case 0:
      return 1;
    case 1:
      return 4;
    default:
      return 16;
  }
}

// --- gfx10.3 (RDNA2 / Prospero) swizzle modes. -----------------------------
// A PS5 T# names a 5-bit gfx10 swizzle mode; it reaches this module as
// kGfx10TilingBase + sw_mode. Each mode maps to one address equation family
// ("pattern set"); the modes with no 2D equation are refused outright so a
// surface is never addressed with a neighbouring mode's pattern.
constexpr u32 kGfx10Modes = 32;

enum Gfx10Set : u8 {
  kG10_256S,
  kG10_256D,
  kG10_4kS,
  kG10_4kD,
  kG10_4kSX,
  kG10_4kDX,
  kG10_64kS,
  kG10_64kD,
  kG10_64kST,
  kG10_64kDT,
  kG10_64kSX,
  kG10_64kDX,
  kG10_64kZX,
  kG10_64kRX,
  kG10_SetCount,
  kG10_Linear = 0xfe,
  kG10_None = 0xff,
};

struct Gfx10ModeInfo {
  u8 set;
  u8 blk_log2;
};

bool TilingIsGfx10(u32 idx) {
  return idx >= kGfx10TilingBase && idx < kGfx10TilingBase + kGfx10Modes;
}

Gfx10ModeInfo Gfx10ModeOf(u32 sw_mode) {
  switch (sw_mode) {
    case 0:   // LINEAR
    case 31:  // LINEAR_GENERAL
      return {kG10_Linear, 0};
    case 1:
      return {kG10_256S, 8};
    case 2:
      return {kG10_256D, 8};
    case 5:
      return {kG10_4kS, 12};
    case 6:
      return {kG10_4kD, 12};
    case 9:
      return {kG10_64kS, 16};
    case 10:
      return {kG10_64kD, 16};
    case 17:
      return {kG10_64kST, 16};
    case 18:
      return {kG10_64kDT, 16};
    case 21:
      return {kG10_4kSX, 12};
    case 22:
      return {kG10_4kDX, 12};
    case 24:
      return {kG10_64kZX, 16};
    case 25:
      return {kG10_64kSX, 16};
    case 26:
      return {kG10_64kDX, 16};
    case 27:
      return {kG10_64kRX, 16};
    // The sub-64-KiB _R variants, the block-variable modes and the reserved
    // encodings are not part of the gfx10 2D swizzle set and have no equation.
    default:
      return {kG10_None, 0};
  }
}

bool ValidTileMode(u32 idx) {
  if (TilingIsGfx10(idx))
    return Gfx10ModeOf(idx - kGfx10TilingBase).set != kG10_None;
  return idx <= 26 || idx == 31 || TilingIsGfx10Std(idx);
}

u32 EffectiveTileMode(u32 idx) {
  if (idx == 25)
    return 20;  // 2D XThick -> 2D Thick at 32bpp
  if (idx == 26)
    return 21;  // 3D XThick -> 3D Thick at 32bpp
  return idx;
}

u32 TileThickness(ArrayMode am) {
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

u32 NumPipesOf(PipeConfig pc) {
  return pc == kPcP2 ? 2u : 8u;
}
u32 NumPipeBitsOf(PipeConfig pc) {
  return pc == kPcP2 ? 1u : 3u;
}

bool IsPrt(ArrayMode am) {
  return am == kAmPrtThin1 || am == kAmPrtThick || am == kAmPrt2DThin1 ||
         am == kAmPrt2DThick || am == kAmPrt3DThin1 || am == kAmPrt3DThick;
}

// Effective tile size used to select the macro-tile table. Thick microtiles may
// exceed the 1 KiB DRAM-row split, but unlike thin multisample tiles they are
// not split into virtual slices by the address equation.
u32 TileSizeBytes(u32 idx,
                       MicroMode mm,
                       u32 thickness,
                       u32 elem) {
  const u32 tile_bytes_1x = kMicroTilePixels * elem * thickness;
  const u32 color_split = SampleSplitOf(idx) * tile_bytes_1x;
  const u32 cts = color_split < 256u ? 256u : color_split;
  u32 split = (mm == kMmDepth) ? TileSplitHwOf(idx) : cts;
  if (split > 1024u)
    split = 1024u;
  if (split > tile_bytes_1x)
    split = tile_bytes_1x;
  return split;
}

// CalculateMacrotileMode: mtm = log2(tile_split/64); +8 if PRT.
u32 MacroTileModeIndex(u32 idx,
                            MicroMode mm,
                            ArrayMode am,
                            u32 thickness,
                            u32 elem) {
  u32 split = TileSizeBytes(idx, mm, thickness, elem);
  u32 q = split / 64u;
  u32 mtm = 0;
  while ((1u << (mtm + 1)) <= q)
    mtm++;  // bit_width(q)-1
  if (IsPrt(am))
    mtm += 8;
  return mtm;
}

// Element index within an 8x8x{1,4,8} microtile.
inline u32 PixIdx(u32 x,
                       u32 y,
                       u32 z,
                       MicroMode m,
                       u32 thickness,
                       u32 elem) {
  u32 x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
  u32 y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
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
  u32 z0 = z & 1, z1 = (z >> 1) & 1;
  u32 index;
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
inline u32 PipeFromCoord(u32 x,
                              u32 y,
                              u32 slice,
                              PipeConfig pc,
                              ArrayMode am,
                              u32 thickness) {
  u32 tx = x >> 3, ty = y >> 3;
  u32 x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
  u32 y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
  u32 pipe;
  if (pc == kPcP2)
    pipe = x3 ^ y3;
  else if (pc == kPcP8_32x32_8x16) {
    u32 p0 = x4 ^ y3 ^ x5;
    u32 p1 = x3 ^ y4;
    u32 p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  } else {
    // P8_32x32_16x16
    u32 p0 = x3 ^ y3 ^ x4;
    u32 p1 = x4 ^ y4;
    u32 p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  }
  if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick) {
    u32 rotation =
        std::max(1u, NumPipesOf(pc) / 2 - 1) * (slice / thickness);
    pipe ^= rotation & (NumPipesOf(pc) - 1);
  }
  return pipe;
}

// ComputeBankFromCoord (tiling.comp), parameterized by num_banks/widths.
inline u32 BankFromCoord(u32 x,
                              u32 y,
                              u32 slice,
                              const MacroParams& mp,
                              u32 num_pipes,
                              ArrayMode am,
                              u32 thickness,
                              u32 tile_split_slice) {
  u32 tx = (x >> 3) / (mp.bank_width * num_pipes);
  u32 ty = (y >> 3) / mp.bank_height;
  u32 x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1,
           x6 = (tx >> 3) & 1;
  u32 y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1,
           y6 = (ty >> 3) & 1;
  u32 bank = 0;
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
  u32 rotation = 0;
  if (am == kAm2DThin1 || am == kAm2DThick || am == kAm2DXThick)
    rotation = (mp.num_banks / 2 - 1) * (slice / thickness);
  else if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick)
    rotation =
        std::max(1u, num_pipes / 2 - 1) * (slice / thickness) / num_pipes;
  u32 tile_split_rotation = 0;
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
  u32 num_pipes, num_pipe_bits, num_bank_bits;
  u32 thickness, micro_tile_bytes, slices_per_tile;
  u32 macro_pitch, macro_height, macro_tile_bytes, base_align;
};

inline u32 NumBankBitsOf(u32 num_banks) {
  u32 b = 0;
  while ((1u << (b + 1)) <= num_banks)
    b++;
  return b;  // bit_width(num_banks)-1
}

inline u64 SwizzleMacroOffset(u64 total_offset,
                                   u32 pipe,
                                   u32 bank,
                                   const Macro2D& c) {
  const u64 interleave_offset =
      total_offset & ((1u << kPipeInterleaveBits) - 1);
  const u64 offset = total_offset >> kPipeInterleaveBits;
  return interleave_offset |
         (static_cast<u64>(pipe) << kPipeInterleaveBits) |
         (static_cast<u64>(bank)
          << (kPipeInterleaveBits + c.num_pipe_bits)) |
         (offset << (kPipeInterleaveBits + c.num_pipe_bits + c.num_bank_bits));
}

bool ConfigureMacro2D(u32 tiling_idx,
                      ArrayMode am,
                      MicroMode mm,
                      u32 elem,
                      Macro2D& c) {
  c.am = am;
  c.mm = mm;
  c.pc = PipeConfigOf(tiling_idx);
  c.num_pipes = NumPipesOf(c.pc);
  c.num_pipe_bits = NumPipeBitsOf(c.pc);
  c.thickness = TileThickness(am);
  const u32 full_micro_tile_bytes = kMicroTilePixels * elem * c.thickness;
  const u32 tile_split_bytes =
      TileSizeBytes(tiling_idx, mm, c.thickness, elem);
  c.micro_tile_bytes = full_micro_tile_bytes;
  c.slices_per_tile = 1;
  // AddrLib addresses each split portion as a virtual slice with its own bank
  // rotation.
  if (full_micro_tile_bytes > tile_split_bytes && c.thickness == 1) {
    c.micro_tile_bytes = tile_split_bytes;
    c.slices_per_tile = full_micro_tile_bytes / tile_split_bytes;
  }
  u32 mtm = MacroTileModeIndex(tiling_idx, mm, am, c.thickness, elem);
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

u32 AlignUp(u32 value, u32 align) {
  return (value + align - 1) & ~(align - 1);
}

u64 AlignUp(u64 value, u64 align) {
  return (value + align - 1) & ~(align - 1);
}

u32 BitCeil(u32 value) {
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

// gfx10 address equations, one row per (pattern set, bytes-per-element). Bit b
// of the byte offset inside a block is the XOR of every coordinate bit selected
// by entry b: lane 0 (bits 0..15) picks x bits, lane 1 (16..31) y bits and
// lane 2 (32..47) array-slice bits. Only the low blk_log2 entries of a row are
// used, so the shorter rows simply end early.
//
// This is the AMD address-library gfx10 swizzle pattern data for a 16-pipe,
// non-RB+ configuration. That is the PS5's: its render-target and depth
// layouts, read back from the console, reproduce exactly this pipe count and
// pattern family across all five element sizes, and no other entry in the
// library's table set matches them. The non-XOR (_S/_D without _X/_T) rows are
// pipe count independent, so only the XOR rows depend on that choice.
const u64 kGfx10Patterns[kG10_SetCount][5][16] = {
    // 256B_S: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2},
    },
    // 256B_D: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000},
    },
    // 4KB_S: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x10,
         0x200000, 0x20},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80000, 0x8, 0x100000,
         0x10},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4, 0x40000, 0x8, 0x80000, 0x10},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2, 0x40000, 0x4, 0x80000, 0x8},
    },
    // 4KB_D: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000, 0x100000, 0x10,
         0x200000, 0x20},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80000, 0x8, 0x100000,
         0x10},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x40000, 0x8, 0x80000, 0x10},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x40000, 0x4, 0x80000, 0x8},
    },
    // 4KB_S_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000, 0x800100080,
         0x400800010, 0x200200040, 0x100400020},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x800080080,
         0x400400010, 0x200100040, 0x100200020},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x800080040,
         0x400400008, 0x200100020, 0x100200010},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4, 0x800040040, 0x400200008,
         0x200080020, 0x100100010},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2, 0x800040020, 0x400200004,
         0x200080010, 0x100100008},
    },
    // 4KB_D_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000, 0x800100080,
         0x400800010, 0x200200040, 0x100400020},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x800080080,
         0x400400010, 0x200100040, 0x100200020},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x800080040,
         0x400400008, 0x200100020, 0x100200010},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x800040040, 0x400200008,
         0x200080020, 0x100100010},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x800040020, 0x400200004,
         0x200080010, 0x100100008},
    },
    // 64KB_S: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x10,
         0x200000, 0x20, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80000, 0x8, 0x100000,
         0x10, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2, 0x40000, 0x4, 0x80000, 0x8,
         0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_D: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000, 0x100000, 0x10,
         0x200000, 0x20, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80000, 0x8, 0x100000,
         0x10, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x40000, 0x8, 0x80000, 0x10,
         0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x40000, 0x4, 0x80000, 0x8,
         0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_S_T: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000, 0x100080,
         0x800010, 0x200040, 0x400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80080, 0x400010,
         0x100040, 0x200020, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80040, 0x400008,
         0x100020, 0x200010, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4, 0x40040, 0x200008, 0x80020,
         0x100010, 0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2, 0x40020, 0x200004, 0x80010,
         0x100008, 0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_D_T: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000, 0x100080,
         0x800010, 0x200040, 0x400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x80080, 0x400010,
         0x100040, 0x200020, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x80040, 0x400008,
         0x100020, 0x200010, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x40040, 0x200008, 0x80020,
         0x100010, 0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x40020, 0x200004, 0x80010,
         0x100008, 0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_S_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x8, 0x10000, 0x20000, 0x40000, 0x80000, 0x800100080,
         0x400800010, 0x200200040, 0x100400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x800080080,
         0x400400010, 0x200100040, 0x100200020, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x800080040,
         0x400400008, 0x200100020, 0x100200010, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x20000, 0x2, 0x4, 0x800040040, 0x400200008,
         0x200080020, 0x100100010, 0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x10000, 0x20000, 0x1, 0x2, 0x800040020, 0x400200004,
         0x200080010, 0x100100008, 0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_D_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x80000, 0x800100080,
         0x400800010, 0x200200040, 0x100400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x800080080,
         0x400400010, 0x200100040, 0x100200020, 0x200000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x800080040,
         0x400400008, 0x200100020, 0x100200010, 0x200000, 0x20, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x800040040, 0x400200008,
         0x200080020, 0x100100010, 0x100000, 0x20, 0x200000, 0x40},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x800040020, 0x400200004,
         0x200080010, 0x100100008, 0x100000, 0x10, 0x200000, 0x20},
    },
    // 64KB_Z_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x10000, 0x2, 0x20000, 0x4, 0x40000, 0x8, 0x100000, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x10000, 0x2, 0x20000, 0x4, 0x40000, 0x8, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x100000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x4, 0x40000, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x80000, 0x10, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x4, 0x800080008, 0x400100010,
         0x200200040, 0x100400020, 0x40000, 0x8, 0x100000, 0x40},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x800080008, 0x400100010,
         0x200200040, 0x100400020, 0x40000, 0x4, 0x80000, 0x10},
    },
    // 64KB_R_X: 1/2/4/8/16 bytes per element
    {
        {0x1, 0x2, 0x4, 0x20000, 0x10000, 0x40000, 0x8, 0x100000, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x400000, 0x40, 0x800000, 0x80},
        {0, 0x1, 0x2, 0x4, 0x10000, 0x20000, 0x40000, 0x8, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x100000, 0x40, 0x400000, 0x80},
        {0, 0, 0x1, 0x2, 0x10000, 0x20000, 0x40000, 0x4, 0x800080008,
         0x400100010, 0x200200040, 0x100400020, 0x80000, 0x10, 0x400000, 0x40},
        {0, 0, 0, 0x1, 0x10000, 0x2, 0x4, 0x20000, 0x800080008, 0x400100010,
         0x200200040, 0x100400020, 0x40000, 0x8, 0x100000, 0x40},
        {0, 0, 0, 0, 0x1, 0x10000, 0x2, 0x20000, 0x800080008, 0x400100010,
         0x200200040, 0x100400020, 0x40000, 0x4, 0x80000, 0x10},
    },
};

u32 Gfx10ElemLog2(u32 elem_bytes) {
  switch (elem_bytes) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    default:
      return 4;  // 16
  }
}

// A block always holds 2^blk_log2 bytes; its element extent is square, or twice
// as wide as it is tall when the element count is an odd power of two.
void Gfx10BlockDim(u32 blk_log2, u32 elem_log2, u32& w, u32& h) {
  const u32 elems_log2 = blk_log2 - elem_log2;
  const u32 w_log2 = (elems_log2 + 1) / 2;
  w = 1u << w_log2;
  h = 1u << (elems_log2 - w_log2);
}

u32 ShiftCeil(u32 value, u32 shift) {
  return (value >> shift) + ((value & ((1u << shift) - 1)) ? 1u : 0u);
}

// The equation XORs independent contributions from x, y and the array slice,
// so the whole thing collapses to one table lookup per axis.
class Gfx10Addresser {
 public:
  bool Init(u32 sw_mode, u32 elem_bytes) {
    const Gfx10ModeInfo mode = Gfx10ModeOf(sw_mode);
    if (mode.set >= kG10_SetCount)
      return false;
    blk_log2_ = mode.blk_log2;
    const u32 elem_log2 = Gfx10ElemLog2(elem_bytes);
    Gfx10BlockDim(blk_log2_, elem_log2, block_w_, block_h_);
    const u64* pattern = kGfx10Patterns[mode.set][elem_log2];
    for (u32 bit = 0; bit < blk_log2_; bit++)
      for (u32 axis = 0; axis < 3; axis++)
        for (u32 i = 0; i < 16; i++)
          if ((pattern[bit] >> (axis * 16 + i)) & 1)
            contrib_[axis][i] |= 1u << bit;
    return true;
  }

  u32 Offset(u32 axis, u32 coord) const {
    u32 offset = 0;
    for (u32 i = 0; coord; coord >>= 1, i++)
      if (coord & 1)
        offset ^= contrib_[axis][i];
    return offset;
  }

  u32 block_w() const { return block_w_; }
  u32 block_h() const { return block_h_; }
  u64 block_bytes() const { return 1ull << blk_log2_; }

 private:
  u32 contrib_[3][16] = {};
  u32 blk_log2_ = 0;
  u32 block_w_ = 0;
  u32 block_h_ = 0;
};

// gfx10 stores a mip chain smallest-level-first and packs every level that fits
// into one shared "mip tail" block, where a level is placed by an x/y offset
// fed into the swizzle rather than by a byte offset. Array slices repeat the
// whole chain, so a layer is one constant stride away at every level.
bool BuildGfx10Layout(TextureLayout32& out,
                      u32 width,
                      u32 height,
                      u32 pitch,
                      u32 layers,
                      u32 mip_levels,
                      u32 sw_mode,
                      u32 elem) {
  const Gfx10ModeInfo mode = Gfx10ModeOf(sw_mode);
  if (mode.set == kG10_None)
    return false;

  for (u32 mip = 0; mip < mip_levels; mip++) {
    out.mips[mip].width = std::max(width >> mip, 1u);
    out.mips[mip].height = std::max(height >> mip, 1u);
  }

  if (mode.set == kG10_Linear) {
    // LINEAR pads every row to 256 bytes; LINEAR_GENERAL pads nothing. Only a
    // single-level surface may carry a caller-supplied pitch.
    const u32 pitch_align = sw_mode == 0 ? 256 / elem : 1;
    u64 chain = 0;
    for (u32 mip = mip_levels; mip-- > 0;) {
      TextureMipLayout32& level = out.mips[mip];
      const u32 raw_pitch =
          mip_levels > 1 ? ShiftCeil(width, mip) : std::max(width, pitch);
      level.pitch = AlignUp(raw_pitch, pitch_align);
      level.stored_height =
          mip_levels > 1 ? ShiftCeil(height, mip) : level.height;
      level.offset = chain;
      level.size =
          static_cast<u64>(level.pitch) * level.stored_height * elem;
      chain += level.size;
    }
    out.layer_stride = chain;
    out.size = chain * layers;
    return out.size != 0;
  }

  u32 block_w = 0, block_h = 0;
  const u32 elem_log2 = Gfx10ElemLog2(elem);
  Gfx10BlockDim(mode.blk_log2, elem_log2, block_w, block_h);
  const u64 block_bytes = 1ull << mode.blk_log2;

  // 256 B blocks are addressed without a mip tail; the macro block sizes hold
  // 8 (4 KiB) and 12 (64 KiB) levels in theirs.
  const u32 max_in_tail = mode.blk_log2 > 8 ? mode.blk_log2 - 4 : 0;
  u32 first_tail = mip_levels;
  if (mode.blk_log2 > 8) {
    for (u32 mip = 0; mip < mip_levels; mip++) {
      if (ShiftCeil(width, mip) <= block_w / 2 &&
          ShiftCeil(height, mip) <= block_h &&
          mip_levels - mip <= max_in_tail) {
        first_tail = mip;
        break;
      }
    }
  }

  for (u32 mip = 0; mip < first_tail; mip++) {
    TextureMipLayout32& level = out.mips[mip];
    level.pitch = AlignUp(ShiftCeil(width, mip), block_w);
    level.stored_height = AlignUp(ShiftCeil(height, mip), block_h);
    level.size = static_cast<u64>(level.pitch) * level.stored_height * elem;
  }
  u64 chain = first_tail < mip_levels ? block_bytes : 0;
  for (u32 mip = first_tail; mip-- > 0;) {
    out.mips[mip].offset = chain;
    chain += out.mips[mip].size;
  }

  u32 micro_w = 0, micro_h = 0;
  Gfx10MicroShape(elem, micro_w, micro_h);
  u32 tail_w = block_w / 2, tail_h = block_h;
  for (u32 mip = first_tail; mip < mip_levels; mip++) {
    TextureMipLayout32& level = out.mips[mip];
    const u32 order = max_in_tail - 1 - (mip - first_tail);
    const u32 tail_offset = order > 6 ? (16u << order) : (order << 8);
    level.pitch = tail_w;
    level.stored_height = tail_h;
    level.size = static_cast<u64>(tail_w) * tail_h * elem;
    // Every tail level shares one block at the start of the chain; its
    // position inside that block is a coordinate bias, interleaved x/y in the
    // level's own offset. Both supported block sizes are an even power of two,
    // so the odd-block-size axis swap never applies.
    u32 tail_x = 0, tail_y = 0;
    for (u32 i = 0; i < 6; i++) {
      tail_x |= ((tail_offset >> (9 + 2 * i)) & 1) << i;
      tail_y |= ((tail_offset >> (8 + 2 * i)) & 1) << i;
    }
    level.mip_tail_x = tail_x * micro_w;
    level.mip_tail_y = tail_y * micro_h;
    tail_w = std::max(tail_w >> 1, micro_w);
    tail_h = std::max(tail_h >> 1, micro_h);
  }

  out.layer_stride = chain;
  out.size = chain * layers;
  return out.size != 0;
}

template <u32 Elem, bool Detile>
inline void CopyElement(u8* tiled, u8* linear) {
  if constexpr (Detile)
    std::memcpy(linear, tiled, Elem);
  else
    std::memcpy(tiled, linear, Elem);
}

template <u32 Elem, bool Detile>
void CopyLinearMip(u8* tiled,
                   u8* linear,
                   const TextureMipLayout32& level,
                   u64 layer_offset,
                   size_t linear_row_bytes) {
  tiled += layer_offset;
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
      level.height, static_cast<u64>(level.width) * level.height,
      [&](u32 y0, u32 y1) {
        for (u32 y = y0; y < y1; ++y) {
          u8* tiled_row =
              tiled + static_cast<size_t>(y) * level.pitch * Elem;
          u8* linear_row =
              linear + static_cast<size_t>(y) * linear_row_bytes;
          if constexpr (Detile)
            std::memcpy(linear_row, tiled_row, logical_row_bytes);
          else
            std::memcpy(tiled_row, linear_row, logical_row_bytes);
        }
      });
}

template <u32 Elem, bool Detile>
void CopyMicroTiledMip(u8* tiled,
                       u8* linear,
                       const TextureMipLayout32& level,
                       u32 layer,
                       size_t linear_row_bytes,
                       MicroMode mm) {
  std::array<u32, kMicroTilePixels> element_offsets{};
  for (u32 y = 0; y < kMicroH; ++y)
    for (u32 x = 0; x < kMicroW; ++x)
      element_offsets[y * kMicroW + x] =
          PixIdx(x, y, layer, mm, level.thickness, Elem) * Elem;

  const u32 tile_columns = (level.width + kMicroW - 1) / kMicroW;
  const u32 tile_rows = (level.height + kMicroH - 1) / kMicroH;
  const u32 physical_tiles_per_row = level.pitch / kMicroW;
  const u64 micro_tile_bytes =
      static_cast<u64>(kMicroTilePixels) * Elem * level.thickness;
  const u64 group_bytes = static_cast<u64>(level.pitch) *
                               level.stored_height * level.thickness * Elem;
  const u64 slice_offset = (layer / level.thickness) * group_bytes;

  RowPool::get().run(
      tile_rows, static_cast<u64>(level.width) * level.height,
      [&](u32 tile_y0, u32 tile_y1) {
        for (u32 tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {
          const u32 first_y = tile_y * kMicroH;
          const u32 copy_height =
              std::min(kMicroH, level.height - first_y);
          for (u32 tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const u32 first_x = tile_x * kMicroW;
            const u32 copy_width =
                std::min(kMicroW, level.width - first_x);
            const u64 tile_offset =
                slice_offset +
                (static_cast<u64>(tile_y) * physical_tiles_per_row +
                 tile_x) *
                    micro_tile_bytes;
            for (u32 y = 0; y < copy_height; ++y) {
              u8* linear_row =
                  linear + static_cast<size_t>(first_y + y) * linear_row_bytes +
                  static_cast<size_t>(first_x) * Elem;
              for (u32 x = 0; x < copy_width; ++x)
                CopyElement<Elem, Detile>(
                    tiled + tile_offset + element_offsets[y * kMicroW + x],
                    linear_row + static_cast<size_t>(x) * Elem);
            }
          }
        }
      });
}

template <u32 Elem, bool Detile, bool Split>
void CopyMacroTiledMip(u8* tiled,
                       u8* linear,
                       const TextureMipLayout32& level,
                       u32 layer,
                       size_t linear_row_bytes,
                       const Macro2D& c) {
  std::array<u32, kMicroTilePixels> element_offsets{};
  std::array<u16, kMicroTilePixels> split_slices{};
  for (u32 y = 0; y < kMicroH; ++y) {
    for (u32 x = 0; x < kMicroW; ++x) {
      const u32 i = y * kMicroW + x;
      const u32 raw_offset =
          PixIdx(x, y, layer, c.mm, c.thickness, Elem) * Elem;
      if constexpr (Split) {
        split_slices[i] =
            static_cast<u16>(raw_offset / c.micro_tile_bytes);
        element_offsets[i] = raw_offset % c.micro_tile_bytes;
      } else {
        element_offsets[i] = raw_offset;
      }
    }
  }

  const u32 tile_columns = (level.width + kMicroW - 1) / kMicroW;
  const u32 tile_rows = (level.height + kMicroH - 1) / kMicroH;
  const u32 macro_tiles_per_row = level.pitch / c.macro_pitch;
  const u32 macro_tiles_per_slice =
      macro_tiles_per_row * (level.stored_height / c.macro_height);
  const u64 slice_bytes =
      static_cast<u64>(macro_tiles_per_slice) * c.macro_tile_bytes;
  const u64 slice_group =
      static_cast<u64>(c.slices_per_tile) * (layer / c.thickness);

  RowPool::get().run(
      tile_rows, static_cast<u64>(level.width) * level.height,
      [&](u32 tile_y0, u32 tile_y1) {
        for (u32 tile_y = tile_y0; tile_y < tile_y1; ++tile_y) {
          const u32 first_y = tile_y * kMicroH;
          const u32 copy_height =
              std::min(kMicroH, level.height - first_y);
          const u64 macro_row =
              static_cast<u64>(first_y / c.macro_height) *
              macro_tiles_per_row;
          const u32 tile_row = tile_y % c.mp.bank_height;
          for (u32 tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const u32 first_x = tile_x * kMicroW;
            const u32 copy_width =
                std::min(kMicroW, level.width - first_x);
            const u64 macro_tile_offset =
                (macro_row + first_x / c.macro_pitch) * c.macro_tile_bytes;
            const u32 tile_col = (tile_x / c.num_pipes) % c.mp.bank_width;
            const u64 tile_offset =
                static_cast<u64>(tile_row * c.mp.bank_width + tile_col) *
                c.micro_tile_bytes;
            const u64 common_offset = macro_tile_offset + tile_offset;

            u32 swizzle_x = first_x;
            u32 swizzle_y = first_y;
            if (c.am == kAmPrtThin1 || c.am == kAmPrtThick) {
              swizzle_x %= c.macro_pitch;
              swizzle_y %= c.macro_height;
            }
            const u32 pipe = PipeFromCoord(swizzle_x, swizzle_y, layer,
                                                c.pc, c.am, c.thickness);

            std::array<u64, 16> split_bases{};
            std::array<u32, 16> split_banks{};
            u64 base = 0;
            u32 bank = 0;
            if constexpr (Split) {
              for (u32 split = 0; split < c.slices_per_tile; ++split) {
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

            for (u32 y = 0; y < copy_height; ++y) {
              u8* linear_row =
                  linear + static_cast<size_t>(first_y + y) * linear_row_bytes +
                  static_cast<size_t>(first_x) * Elem;
              for (u32 x = 0; x < copy_width; ++x) {
                const u32 i = y * kMicroW + x;
                u64 tiled_offset;
                if constexpr (Split) {
                  const u32 split = split_slices[i];
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
template <u32 Elem, bool Detile>
void CopyGfx10StdMip(u8* tiled,
                     u8* linear,
                     const TextureMipLayout32& level,
                     u32 layer,
                     size_t linear_row_bytes,
                     u32 tiling_idx) {
  u32 mw = 0, mh = 0;
  Gfx10MicroShape(Elem, mw, mh);
  const u32 n = Gfx10BlockTiles(tiling_idx);
  const u32 bw = mw * n, bh = mh * n;
  const u32 blocks_per_row = level.pitch / bw;
  const u64 block_bytes = static_cast<u64>(bw) * bh * Elem;
  const u64 slice_bytes =
      static_cast<u64>(level.pitch) * level.stored_height * Elem;
  u8* slice = tiled + slice_bytes * layer;
  // Within a block: the 256 B micro-tile has its own bit pattern, and above it
  // the micro-tiles interleave Y first. For 32bpp the micro-tile is 8x8 laid
  // out x0,x1,y0,y1,y2,x2 -- a 4x4 row-major quadrant, quadrants ordered Y
  // first. Solved from the surfaces themselves: a 64 KiB block only ever holds
  // non-zero elements inside the WxH image, and over two dozen differently
  // shaped Minecraft UI surfaces this is the only bit order that never places a
  // non-zero element outside one.
  u32 x_bits = 0, y_bits = 0;
  while ((1u << x_bits) < bw)
    x_bits++;
  while ((1u << y_bits) < bh)
    y_bits++;
  u32 micro_src[6] = {}, micro_is_x[6] = {}, micro_n = 0;
  if constexpr (Elem == 4) {
    const u32 sx[6] = {0, 1, 0, 1, 2, 2};
    const u32 is_x[6] = {1, 1, 0, 0, 0, 1};
    for (u32 i = 0; i < 6; i++) {
      micro_src[i] = sx[i];
      micro_is_x[i] = is_x[i];
    }
    micro_n = 6;
  }
  DetileParallelRows(level.height, [&](u32 y0, u32 y1) {
    for (u32 y = y0; y < y1; y++) {
      u8* linear_row = linear + static_cast<size_t>(y) * linear_row_bytes;
      const u32 by = y / bh, iy = y % bh;
      for (u32 x = 0; x < level.width; x++) {
        const u32 bx = x / bw, ix = x % bw;
        u32 element = 0, bit = 0, xb = 0, yb = 0;
        for (u32 i = 0; i < micro_n; i++) {
          const u32 v = micro_is_x[i] ? ix : iy;
          element |= ((v >> micro_src[i]) & 1) << bit++;
          (micro_is_x[i] ? xb : yb)++;
        }
        while (xb < x_bits || yb < y_bits) {
          if (yb < y_bits)
            element |= ((iy >> yb++) & 1) << bit++;
          if (xb < x_bits)
            element |= ((ix >> xb++) & 1) << bit++;
        }
        const u64 off =
            (static_cast<u64>(by) * blocks_per_row + bx) * block_bytes +
            static_cast<u64>(element) * Elem;
        u8* t = slice + off;
        u8* l = linear_row + static_cast<size_t>(x) * Elem;
        if (Detile)
          std::memcpy(l, t, Elem);
        else
          std::memcpy(t, l, Elem);
      }
    }
  });
}

// gfx10 element address: the block index picks the block, the equation picks
// the byte inside it. Both are separable in x and y, so the per-row and
// per-column terms are tabulated once and the inner loop is an XOR and an add.
template <u32 Elem, bool Detile>
void CopyGfx10Mip(u8* tiled,
                  u8* linear,
                  const TextureLayout32& layout,
                  u32 mip,
                  u32 layer,
                  size_t linear_row_bytes,
                  const Gfx10Addresser& addr) {
  const TextureMipLayout32& level = layout.mips[mip];
  const u32 block_w = addr.block_w(), block_h = addr.block_h();
  const u64 block_bytes = addr.block_bytes();
  const u32 blocks_per_row = level.pitch / block_w;
  const u32 slice_offset = addr.Offset(2, layer);

  std::vector<u32> x_offsets(level.width), y_offsets(level.height);
  std::vector<u64> x_blocks(level.width);
  for (u32 x = 0; x < level.width; x++) {
    x_offsets[x] = addr.Offset(0, x + level.mip_tail_x);
    x_blocks[x] = static_cast<u64>(x / block_w) * block_bytes;
  }
  for (u32 y = 0; y < level.height; y++)
    y_offsets[y] = addr.Offset(1, y + level.mip_tail_y) ^ slice_offset;

  u8* slice = tiled + layout.layer_stride * layer;
  DetileParallelRows(level.height, [&](u32 y0, u32 y1) {
    for (u32 y = y0; y < y1; y++) {
      u8* linear_row = linear + static_cast<size_t>(y) * linear_row_bytes;
      const u64 row_base =
          static_cast<u64>(y / block_h) * blocks_per_row * block_bytes;
      const u32 y_offset = y_offsets[y];
      for (u32 x = 0; x < level.width; x++)
        CopyElement<Elem, Detile>(
            slice + row_base + x_blocks[x] + (x_offsets[x] ^ y_offset),
            linear_row + static_cast<size_t>(x) * Elem);
    }
  });
}

template <u32 Elem, bool Detile>
bool CopyTextureMip(u8* tiled_image,
                    u8* linear,
                    size_t linear_row_bytes,
                    const TextureLayout32& layout,
                    u32 mip,
                    u32 layer) {
  const TextureMipLayout32& level = layout.mips[mip];
  u8* tiled = tiled_image + level.offset;
  if (TilingIsLinear(layout.tiling_idx)) {
    const u64 layer_offset =
        layout.layer_stride ? layout.layer_stride * layer
                            : static_cast<u64>(layer) * level.pitch *
                                  level.stored_height * Elem;
    CopyLinearMip<Elem, Detile>(tiled, linear, level, layer_offset,
                                linear_row_bytes);
    return true;
  }

  if (TilingIsGfx10(layout.tiling_idx)) {
    Gfx10Addresser addr;
    if (!addr.Init(layout.tiling_idx - kGfx10TilingBase, Elem))
      return false;
    CopyGfx10Mip<Elem, Detile>(tiled, linear, layout, mip, layer,
                               linear_row_bytes, addr);
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

void DetileParallelRows(u32 rows,
                        const std::function<void(u32, u32)>& fn) {
  RowPool::get().run(rows, static_cast<u64>(rows) * 1024, fn);
}

bool TilingIsLinear(u32 tiling_idx) {
  // Per the Liverpool GB_TILE_MODE table only DisplayLinearAligned(8) and
  // DisplayLinearGeneral(31) are genuinely linear (ArrayLinearAligned/General).
  // Keep the same set the previous code used so Isaac's linear surfaces and
  // small UI textures are still straight-copied.
  if (TilingIsGfx10(tiling_idx))
    return Gfx10ModeOf(tiling_idx - kGfx10TilingBase).set == kG10_Linear;
  return tiling_idx == 8 || tiling_idx == 31;
}

bool TilingSupported(u32 tiling_idx) {
  return ValidTileMode(EffectiveTileMode(tiling_idx));
}

bool BuildTextureLayout32(TextureLayout32& out,
                          u32 width,
                          u32 height,
                          u32 pitch,
                          u32 layers,
                          u32 mip_levels,
                          u32 tiling_idx,
                          bool pow2_pad,
                          u32 elem_bytes) {
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
  if (TilingIsGfx10(tiling_idx))
    return BuildGfx10Layout(out, width, height, pitch, layers, mip_levels,
                            tiling_idx - kGfx10TilingBase, elem_bytes);
  const ArrayMode am = ArrayModeOf(tiling_idx);
  const MicroMode mm = MicroModeOf(tiling_idx);
  const u32 thickness = TileThickness(am);
  Macro2D macro{};
  if (IsMacroTiled(am) &&
      !ConfigureMacro2D(tiling_idx, am, mm, elem_bytes, macro))
    return false;

  u64 end = 0;
  for (u32 mip = 0; mip < mip_levels; mip++) {
    TextureMipLayout32& level = out.mips[mip];
    level.width = std::max(width >> mip, 1u);
    level.height = std::max(height >> mip, 1u);
    u32 raw_pitch = std::max(std::max(pitch >> mip, 1u), level.width);
    u32 storage_height = level.height;
    if (pow2_pad) {
      raw_pitch = BitCeil(raw_pitch);
      storage_height = BitCeil(storage_height);
    }
    level.thickness = thickness;

    u64 base_align = 1;
    if (TilingIsGfx10Std(tiling_idx)) {
      // Each mip is padded to whole blocks and starts on a block boundary.
      u32 mw = 0, mh = 0;
      Gfx10MicroShape(elem_bytes, mw, mh);
      const u32 n = Gfx10BlockTiles(tiling_idx);
      const u32 bw = mw * n, bh = mh * n;
      level.pitch = AlignUp(raw_pitch, bw);
      level.stored_height = AlignUp(storage_height, bh);
      base_align = static_cast<u64>(bw) * bh * elem_bytes;
      level.macro_tiled = false;
    } else if (tiling_idx == 31) {
      level.pitch = raw_pitch;
      level.stored_height = storage_height;
    } else if (tiling_idx == 8) {
      level.pitch = AlignUp(raw_pitch, 16);
      level.stored_height = storage_height;
      while ((static_cast<u64>(level.pitch) * level.stored_height) % 64)
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

    const u32 stored_layers = AlignUp(layers, thickness);
    level.offset = AlignUp(end, base_align);
    level.size = static_cast<u64>(level.pitch) * level.stored_height *
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
                        u32 mip,
                        u32 layer) {
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
                               u32 mip,
                               u32 layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32& level = layout.mips[mip];
  if (dst_row_bytes < static_cast<size_t>(level.width) * layout.elem_bytes ||
      (level.height > 1 && dst_row_bytes > SIZE_MAX / (level.height - 1)))
    return false;
  auto* tiled = const_cast<u8*>(static_cast<const u8*>(src));
  auto* linear = static_cast<u8*>(dst);
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
                        u32 mip,
                        u32 layer) {
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
                               u32 mip,
                               u32 layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32& level = layout.mips[mip];
  if (src_row_bytes < static_cast<size_t>(level.width) * layout.elem_bytes ||
      (level.height > 1 && src_row_bytes > SIZE_MAX / (level.height - 1)))
    return false;
  auto* tiled = static_cast<u8*>(dst);
  auto* linear = const_cast<u8*>(static_cast<const u8*>(src));
  switch (layout.elem_bytes) {
    case 1:
      return CopyTextureMip<1, false>(tiled, linear, src_row_bytes, layout, mip,
                                      layer);
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
