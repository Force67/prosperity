/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// State and helpers the compute_*.cc files share. Private to them.

#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/mutex.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/render/compute.h"
#include "gpu/render/compute_hazard.h"
#include "gpu/render/renderer.h"
#include "gpu/rhi/device.h"
#include "options/options.h"

namespace gpu::render::cs {
using render::ComputeInfo;
}  // namespace gpu::render::cs

namespace gpu::render {
using namespace cs;
}  // namespace gpu::render

namespace gpu::render::cs {

extern base::Option<u64> kDetileDump;
extern base::Option<bool> kCsList;
extern base::Option<int> kCsHist;
extern base::Option<bool> kCsRtTrace;
extern base::Option<bool> kCsRename;

extern base::Option<bool> kCsRtCache;
extern base::Option<bool> kCsSubset;

extern base::Option<bool> kCsSkipRetile;
extern base::Option<bool> kCsSkipUpload;
extern u64 g_cs_skip_n;
extern base::Option<bool> kCsWbFull;
extern base::Option<bool> kCsWbAudit;

extern base::Option<bool> kCsVram;
extern base::Option<bool> kCsGpuTiling;

extern base::Option<bool> kCsTexBridge;
extern base::Option<bool> kCsInFrame;
extern bool g_cs_in_frame;

extern base::Option<bool> kCsImageLazyWb;

extern base::Option<bool> kCsTruth;
extern base::Option<bool> kCsMergeOverlap;
extern base::Option<bool> kCsKeepOverlap;
extern base::Option<u32> kCsBatchCap;
extern base::Option<bool> kCsTexBridgeTrace;
extern base::Option<bool> kFrameChunks;
extern base::Option<u64> kCsFlushTrace;
extern base::Option<int> kCsOverlapTrace;
extern base::Option<bool> kCsSyncReport;
extern base::Option<int> kCsFlushFrom;
extern base::Option<bool> kCsEagerReadback;

extern base::Option<bool> kCsTrack;
extern base::Option<bool> kCsAged;
extern base::Option<bool> kGpuCsgpuVerbose;
bool CsSyncReporting();
extern u64 g_cs_image_staged;

extern u64 g_out_retile_ns, g_out_rt_ns, g_out_tail_ns;
extern u64 g_out_buffer_ns;
extern u64 g_tex_bridge_n, g_tex_bridge_bytes, g_cs_chunk_splits;
extern u64 g_buf_bridge_n, g_buf_bridge_bytes, g_buf_bridge_declined;
extern u64 g_vtx_bridge_n;
extern u64 g_guest_override_n, g_guest_mirror_tiles, g_cs_rt_variants_made;
extern u64 g_gds_imported_n, g_guest_misses_n, g_cs_indirect_n;
extern u64 g_stage_ro_bytes, g_stage_rw_bytes, g_stage_img_bytes;
extern u64 g_stage_guest_detile_bytes, g_stage_guest_detile_n;
extern u64 g_mirror_wb_bytes, g_mirror_wb_n, g_mirror_own_bytes;
extern base::Atomic<u64> g_mirror_resolves;
extern base::HashSet<u64> g_mirror_cached;
extern base::Map<u64, u64> g_mirror_uncached;
extern base::Map<u64, u64> g_mirror_wb_union;
extern base::Option<bool> kMirror;
extern base::Option<bool> kCsListProf;
extern u64 g_in_hash_ns, g_in_detile_ns, g_in_rt_ns, g_in_copy_ns;
extern u64 g_in_hash_n, g_in_rt_n, g_in_wt_skips;
extern u64 g_in_hash_bytes, g_wb_hash_bytes, g_wb_hash_ns;
extern u64 g_in_overlap_ns, g_in_reshape_ns, g_in_alloc_ns;
extern u64 g_in_membuf_ns, g_in_shadow_ns;
extern u64 g_in_overlap_n, g_in_reshape_n, g_in_alloc_n;
extern u64 g_truth_stage_bytes, g_truth_import_n, g_truth_detile_n,
    g_truth_retile_n, g_truth_parent_n, g_truth_fold_n, g_truth_rt_n;

extern base::Map<base::String, u64> g_wb_why;
extern u64 g_wb_why_bytes;
extern u64 g_lazy_kept_n, g_lazy_kept_bytes;
extern u64 g_out_retile_n, g_out_rt_submits;
extern u64 g_gpu_tiling_ns[2], g_gpu_tiling_n[2];

extern u64 g_tile_hin_ns, g_tile_sync_ns, g_tile_hout_ns;
extern u64 g_tile_hin_bytes, g_tile_hout_bytes;

// Staging a 4K surface is tens of megabytes, and one core copies ~4 GB/s of
// memory the GPU has just written. Split anything big enough to be worth the
// handoff across the detiler's pool; below that the call is a plain memcpy.
constexpr u64 kParallelCopyMin = 512 * 1024;

void ParallelCopy(void* dst, const void* src, u64 bytes);

// Per-range staging accounting (DELTA_GPU_CSSYNC). A frame that stages half a
// gigabyte says nothing about what to fix until the bytes are attributed to a
// base and a REASON: guest memory changed, the same base was rebound
// with a different shape, or a render target had to be bridged.
struct StageStat {
  u64 bytes = 0;
  u32 n = 0, first = 0, shape = 0, hash = 0, rt = 0, wait = 0;
  u64 wait_ns = 0;
};

extern base::HashMap<u64, StageStat> g_stage_stats;

// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding
// layout depend only on the code, so only the descriptor set + push constants +
// storage buffers are rebuilt per dispatch.
struct CsPipe {
  rhi::Pipeline* pipe = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::BindGroupLayout* set_layout = nullptr;
  u32 num_res = 0;
  int gds_binding = -1;
};

// The GDS scratchpad: one small buffer for the whole device, which is what the
// hardware's global data share is. Created on first use and left alone
// afterwards; the counters in it are the title's to manage.
struct GdsBuffer {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  static constexpr u64 kBytes = 64 * 1024;
};

extern GdsBuffer g_gds;

// Every read and write: the dependency the old ALL_COMMANDS/MEMORY barriers
// expressed.
constexpr u32 kAccessAll = rhi::kAccessAllRead | rhi::kAccessAllWrite;

constexpr u32 kAccessComputeRW =
    rhi::kAccessComputeRead | rhi::kAccessComputeWrite;

constexpr u32 kAccessCopyRW = rhi::kAccessCopyRead | rhi::kAccessCopyWrite;

bool SplitVram();

rhi::Buffer* CreateCsBuffer(u64 size, u32 usage, rhi::MemoryKind memory);

rhi::BindGroupLayout* CsSetLayout(u32 num_res,
                                  int gds_binding,
                                  int guest_memory_binding,
                                  const base::Vector<int>& textures);

rhi::PipelineLayout* CsPipelineLayout(rhi::BindGroupLayout* set_layout);

rhi::Pipeline* CreateCsPipeline(rhi::PipelineLayout* layout,
                                const base::Vector<u32>& spirv);

// Pipelines compiled ahead of their first dispatch
// (render::PrebuildComputePipeline), by module content and layout shape.
struct PrebuiltCs {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
};

// Filled in once by the thread building it; a consumer waits for `ready`.
struct PrebuiltSlot {
  base::Mutex mutex;
  base::ConditionVariable built;
  bool ready = false;
  PrebuiltCs value;
};

extern base::Mutex g_prebuilt_mutex;
extern base::HashMap<u64, base::SharedPointer<PrebuiltSlot>> g_prebuilt;
u64 PrebuiltKey(const base::Vector<u32>& spirv,
                u32 num_res,
                int guest_memory_binding);

CsPipe* GetCsPipe(const ComputeInfo& ci);

// Persistent compute staging. Dispatches are serialized behind the batch, so
// one set of staging buffers (+ command list) is reused across every
// dispatch: this avoids the per-dispatch buffer/memory/command-buffer churn
// (~3ms/frame). The buffers are host-cached so the copy-BACK read after the
// dispatch hits cache instead of stalling on write-combined memory (which was
// ~25ms/frame for Doom64's 8 MB atlas: the dominant compute cost).
struct CsStage {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
};

extern CsStage g_cs_stage[ComputeInfo::kMaxResources];
extern rhi::CommandList* g_cs_list;
bool BuildCsImageLayouts(const ComputeInfo::Res& res,
                         gcn::TextureLayout32& tiled,
                         gcn::TextureLayout32& linear);

void UnpackR11G11B10(u32 packed, u8* dst);

u32 PackR11G11B10(const u8* src);

bool StageCsImage(const ComputeInfo::Res& res, void* dst);

bool WritebackCsImage(const ComputeInfo::Res& res, const void* src);

// GPU-resident compute working set. Each guest range a CS touches gets a
// persistent host-visible storage buffer keyed by its base address. Staged
// content persists across dispatches and frames: a range the GPU wrote
// (gpu_dirty) is the newest copy and is bound directly with no re-staging;
// guest-sourced ranges revalidate against a content hash at most once per
// frame. Writebacks to guest memory (the expensive image retile) happen
// LAZILY, only when a draw / DMA / frame boundary needs guest memory to be
// current (FlushCsWrites), not after every dispatch.
struct CsRange {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
  // DELTA_GPU_CSVRAM: `buf` is in VRAM (what the shaders bind) and `map` is a
  // separate host-cached mirror, moved across by DMA at the staging edges. The
  // shaders then read and write at VRAM bandwidth instead of over PCIe, which
  // is worth ~6x of the GPU time in a SotC frame; the CPU keeps a cached
  // pointer, which the mapped-VRAM alternative does not.
  rhi::Buffer* host_buf = nullptr;
  bool device_local = false;
  bool readback_pending = false;  // copy recorded, not yet waited on
  bool mirror_current = false;    // map holds the buffer's contents
  u64 size = 0;                   // active staged (linear) byte size
  u64 guest_bytes = 0;            // guest footprint (hash + overlap checks)
  u64 hash = 0;                   // TexHash of guest content when last in sync
  int last_validated_frame = -1;
  int last_used_frame = -1;
  bool gpu_dirty = false;      // buffer newer than guest memory
  bool pending_batch = false;  // referenced by a batch not yet seen complete
  u64 batch_id = 0;            // that batch (see CsBatchWaitId)
  bool image_staging = false;
  // Buffer memory IS the guest pages (an imported host allocation): no
  // staging copy in, no writeback out. See CsRangeImportGuest.
  bool imported = false;
  u64 imported_base = 0;
  u64 imported_offset = 0;  // base - page-aligned import base
  // Staged from a live render-target image rather than guest memory; content
  // changes every frame regardless of the guest bytes, so validity is
  // per-frame (last_rt_frame), never the guest content hash.
  bool rt_sourced = false;
  int last_rt_frame = -1;
  u64 rt_serial = 0;  // the live target's render_serial the staged copy holds
  // The target's render_serial when it took this range's pixels in frame
  // (CsRefreshRtInFrame): rendered past it, the target is the newer copy.
  u64 refreshed_serial = 0;
  ComputeInfo::Res res;  // writeback needs the full layout description
  // A copy of the guest bytes as they were staged IN, kept so the writeback can
  // tell "the shader wrote this word" from "the shader never touched it".
  // Without it the writeback has to assume the whole range is GPU output and
  // copies the stale snapshot back over anything the CPU changed meanwhile –
  // see CsRangeFlushOne.
  base::Vector<u8> shadow;
  bool shadow_valid = false;
  // Bumped whenever the buffer's contents change (a dispatch wrote it, or it
  // was staged in again), so a texture bridged from it knows when to re-copy.
  u64 write_seq = 1;
  // The last frame whose command buffer reads `buf` (CsSupplyTexture): the
  // buffer outlives that frame. `chunk_ref` is the chunk the read was
  // recorded into; a rewrite while that chunk is still open submits it first.
  int frame_ref = -1;
  u64 chunk_ref = 0;
  // The buffer holds the guest bytes of a gfx10 image footprint as they are
  // (DELTA_GPU_CS_TRUTH); `size` == `guest_bytes`, and image bindings read a
  // detiled scratch instead of it.
  bool truth = false;
  u64 rt_seq = 0;  // write_seq the aliased render target was refreshed at
  // A guest-memory reader other than the frame end has pulled this range back
  // before (a CMASK or HTILE the RT bind inspects): its readback is recorded
  // with the dispatch that writes it, so the reader finds the mirror current
  // instead of paying a submit and a wait for the copy alone.
  bool cpu_reader = false;
  // Guest CPU writes into the range while it held dispatch output that is
  // not written back (CsNoteGuestWrites), page aligned and coalesced. They
  // came after the dispatch, so they win over its output.
  base::Vector<base::Pair<u64, u64>> cpu_writes;
  // Guest bytes a fill wrote over a linear image's footprint after its
  // dispatch: the writeback keeps them, as it keeps cpu_writes.
  base::Vector<base::Pair<u64, u64>> fill_kept;
  int dirty_frame = -1;  // frame the range last went dirty
  // Guest pages armed in the write tracker before the last hash or copy, and
  // no write reported since: the staging still holds what guest memory does.
  bool wt_clean = false;
  // When the staging last copied guest memory (g_cs_dirty_stamp's clock).
  u64 staged_stamp = 0;
  // The image in guest layout, tiled on the GPU for pointer-chasing
  // dispatches (GuestMirror), as of write_seq `mirror_seq` through
  // `mirror_table`.
  rhi::Buffer* mirror = nullptr;
  u64 mirror_cap = 0, mirror_seq = 0;
  const struct TileTable* mirror_table = nullptr;
  // When the GPU last wrote it, in dispatch order across all ranges. Ranges
  // whose footprints overlap (a title reusing transient memory) must reach
  // guest memory oldest first, so the newest bytes land last.
  u64 dirty_stamp = 0;
  // g_render_serial at that write: a target at this address rendered past it
  // holds newer bytes than the range does.
  u64 dirty_render_serial = 0;
};

void SnapshotCsShadow(CsRange& range, const ComputeInfo::Res& res, u64 bytes);

// The tiled-truth conversions (compute_tiling.cc).
struct TileTable;

const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed = false,
                              bool pack11 = false);
void RecordImageTiling(rhi::CommandList* c,
                       const TileTable& t,
                       rhi::Buffer* tiled,
                       u64 tiled_off,
                       u64 tiled_bytes,
                       rhi::Buffer* linear,
                       u64 linear_bytes,
                       bool detile,
                       u32 mips = 0,
                       u32 layers = 0,
                       bool depth16 = false);

rhi::Buffer* AcquireView(u64 bytes,
                         u64 base,
                         u64 off,
                         const TileTable* table,
                         u64 seq,
                         bool* fresh);

void StampView(rhi::Buffer* view,
               u64 base,
               u64 off,
               const TileTable* table,
               u64 seq);

extern CsRange g_truth_bridge_scratch;
extern u64 g_cs_scratch_owner;
extern u64 g_cs_scratch_bytes;

extern base::Vector<rhi::Buffer*> g_cs_frame_retired;
bool CsFrameReferenced(const CsRange& e);

extern u64 g_cs_free_bytes, g_cs_recycled_n;

void RecordStagingCopy(rhi::CommandList* c,
                       CsRange& e,
                       u64 bytes,
                       bool to_device);

bool SameCsResourceShape(const ComputeInfo::Res& a, const ComputeInfo::Res& b);

// A live image (color RT or depth target) aliasing a CS resource's guest
// range. Both bridge directions (StageCsRangeFromRt / UploadCsRangeToRt) use
// the same lookup, shape checks and barrier recipe.
struct CsAliasedImage {
  rhi::Texture* texture = nullptr;
  u32 w = 0, h = 0;
  u32 elem_bytes = 0;
  u8 aspect = rhi::kAspectColor;
  rhi::TextureState submitted = rhi::TextureState::kUndefined;
  bool is_depth = false;
  bool is_stencil = false;
  u32 layers = 1;
};

bool FindCsAliasedImage(u64 base,
                        CsAliasedImage& out,
                        bool for_write = false,
                        bool prefer_depth = false);

bool LiveTargetOfSize(u64 base, u32 w, u32 h);

bool CsAliasedBase(u64 base);

u64 AliasedImageRenderSerial(u64 base);

void AliasedImageBarrier(rhi::CommandList* c,
                         const CsAliasedImage& img,
                         rhi::TextureState from,
                         rhi::TextureState to);

// How a bridge copy moves texels between a live image and the CS staging
// buffer. The two descriptions of one surface disagree routinely and for
// reasons that are not defects: the IMAGE is sized from CB_COLOR's
// pitch/slice while the CS descriptor carries the surface's own width and
// height, and a target the shader reads as R11G11B10F is staged as four
// floats per texel because that is what StageCsImage produces on the
// guest-memory path. Demanding exact agreement rejected GTA:SA's scene
// target on both counts, so its whole compute post chain read the guest
// bytes under the target, which draws never write, i.e. black.
struct AliasedCopyPlan {
  u32 w = 0;
  u32 h = 0;
  bool unpack = false;   // image holds packed 11/11/10, staging holds float4
  bool widen = false;    // image holds 8/16-bit texels, staging holds u32
  bool depth16 = false;  // image holds float depth, the surface UNORM16
  u32 layers = 1;
};

bool PlanAliasedCopy(const CsAliasedImage& img,
                     const ComputeInfo::Res& res,
                     const char* dir,
                     AliasedCopyPlan& plan);

// A host-visible scratch the format-converting bridge copies through: the
// image side is packed at the image's own element size, the staging side
// holds the unpacked floats, and no image<->buffer copy converts between the
// two in one step.
struct CsBridgeScratch {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
};

extern CsBridgeScratch g_bridge;

extern bool g_cs_failed;

// A float scratch image the packed-float bridge blits through: no
// buffer<->image copy converts R11G11B10 to the float4 staging layout, and a
// blit converts any colour format to any other.
struct BridgeFloatImage {
  rhi::Texture* texture = nullptr;
  u32 w = 0, h = 0;
};

extern base::Vector<BridgeFloatImage> g_bridge_float_images;
extern u64 g_in_frame_bridge_n;
bool BlitsBetween(rhi::Format a, rhi::Format b);

rhi::Texture* GetBridgeFloatImage(u32 w, u32 h);

bool StageCsRangeFromRtInFrame(const ComputeInfo::Res& res, CsRange& e);

extern base::HashSet<u64> g_rt_unbridgeable;
u64 RtShapeKey(const ComputeInfo::Res& res);

bool StageCsRangeFromRt(const ComputeInfo::Res& res, CsRange& e);

bool UploadCsRangeToRt(u64 base, CsRange& e);

extern base::HashMap<u64, CsRange> g_cs_ranges;
extern decltype(g_cs_ranges)::iterator g_cs_evict_next;
extern u64 g_cs_dirty_stamp;
void SortByDirtyOrder(base::Vector<u64>& bases);

extern base::Vector<CsRange*> g_cs_pending;
extern u64 g_cs_range_bytes;

constexpr u32 kCsDirtyPageShift = 16;

extern base::HashMap<u64, base::Vector<u64>> g_cs_dirty_pages;

bool DirtyFilterClear(u64 base, u64 end);

extern base::HashMap<u64, u64> g_cs_dirty_bases;

extern base::HashMap<u64, u64> g_cs_dirty_lo;
u64 DirtyLo(u64 base);

extern u64 g_cs_writeback_gen;
u64 RangeEnd(u64 base, u64 bytes);

void IndexDirtySpan(u64 base, u64 lo, u64 end);

void IndexDirtyRange(u64 base, u64 bytes);

void UnindexDirtyRange(u64 base, u64 bytes);

// Every range by the 1 MiB blocks its guest footprint covers, dirty or not:
// what a CPU write to guest memory looks up instead of walking every range.
constexpr u32 kCsRangeBlockShift = 20;

extern base::HashMap<u64, base::Vector<u64>> g_cs_range_blocks;
void IndexCsRange(u64 base, u64 bytes);

void UnindexCsRange(u64 base, u64 bytes);

// Calls fn(other) for every dirty range overlapping [base, base+bytes) until
// it returns true; a range spanning several index pages may be offered more
// than once. Allocation free, for the per-draw paths.
template <typename Fn>
bool AnyDirtyOverlapping(u64 base, u64 bytes, Fn&& fn) {
  if (g_cs_dirty_pages.empty() || !bytes)
    return false;
  const u64 end = RangeEnd(base, bytes);
  if (DirtyFilterClear(base, end))
    return false;
  const auto hits = [&](u64 other) {
    auto indexed = g_cs_dirty_bases.find(other);
    if (indexed == g_cs_dirty_bases.end() || DirtyLo(other) >= end ||
        base >= indexed->second)
      return false;
    auto range = g_cs_ranges.find(other);
    return range != g_cs_ranges.end() && range->second.gpu_dirty && fn(other);
  };
  // Same smaller-side walk as DirtyRangesOverlapping, and safe for the same
  // reason: every candidate's overlap is rechecked exactly.
  const u64 first_page = base >> kCsDirtyPageShift;
  const u64 last_page = (end - 1) >> kCsDirtyPageShift;
  if (last_page - first_page + 1 > g_cs_dirty_pages.size()) {
    for (const auto& [page, bases] : g_cs_dirty_pages)
      for (u64 other : bases)
        if (hits(other))
          return true;
    return false;
  }
  for (u64 page = first_page; page <= last_page; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found == g_cs_dirty_pages.end())
      continue;
    for (u64 other : found->second)
      if (hits(other))
        return true;
  }
  return false;
}

base::Vector<u64> DirtyRangesOverlapping(u64 base,
                                         u64 bytes,
                                         u64 exclude = UINT64_MAX);

bool EnsureGdsBuffer();

base::HashMap<u64, CsRange>::iterator EraseCsRange(
    base::HashMap<u64, CsRange>::iterator it);

bool CsRangeEnsureBuffer(CsRange& e, u64 size);

bool CsRangeRename(CsRange& e, u64 size);

}  // namespace gpu::render::cs

namespace gpu::render::cs {

#ifdef DELTA_HAVE_SPIRV_BACKEND
struct TileTable {
  rhi::Buffer* buf = nullptr;
  u64 bytes = 0;
  base::Vector<gcn::ImageTilingParams> params;  // per mip; `detile` patched
  u32 layers = 0;
  bool expanded = false;
};

extern base::HashMap<u64, TileTable> g_tile_tables;

// The tiling shader, its descriptors pushed into whichever command list
// records a conversion.
struct TilingState {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
  CsRange buffers[3];
};

extern TilingState g_tiling;

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed,
                 bool pack11);

// Transient detiled views. A scratch is reused within the frame as soon as
// the pass that consumed it has been recorded (the conversions' own
// barriers order the accesses) and never by the dispatch being recorded.
// A scratch that goes cold is retired through the frame ring.
struct CsScratch {
  rhi::Buffer* buf = nullptr;
  u64 cap = 0;
  u64 owner = 0;
  int last_frame = -1;
  // The view it holds: which bytes, converted how, at which revision. A
  // later binding of the same unchanged truth reuses it as it is.
  u64 view_base = 0;
  u64 view_off = 0;
  const TileTable* view_table = nullptr;
  u64 view_seq = 0;
};

extern base::Vector<CsScratch> g_cs_scratch;
extern u64 g_truth_view_hits;

rhi::Buffer* AcquireScratch(u64 bytes);
#endif

}  // namespace gpu::render::cs

namespace gpu::render::cs {

bool TruthEligible(const ComputeInfo::Res& r);

bool CanGpuTile(const ComputeInfo::Res& res, const CsRange& e);

bool ConvertCsRangeImage(const ComputeInfo::Res& res, CsRange& e, bool detile);

extern bool g_cs_batch_open;
extern u32 g_cs_batch_count;

// Batches are submitted without waiting and retired in order as their
// submissions complete; a reader waits for the one batch its range was last
// recorded into (CsBatchWaitId), and only then. The GPU then executes batch N
// while the walk records N+1, where a synchronous flush idled both in turn.
constexpr u32 kCsBatchRing = 6;

extern u64 g_cs_batch_id;
extern u64 g_cs_batch_next_id;
extern u64 g_cs_batch_done;

// GPU writes into guest memory a batch has not finished (CopyGdsToGuest).
struct PendingGuestWrite {
  u64 base, end, batch;
};

extern base::Vector<PendingGuestWrite> g_pending_guest_writes;
extern u64 g_publish_batch;
extern u32 g_cs_batch_cur;
extern u64 g_cs_submit_n;

// What the open batch contains. A device loss names no dispatch on its own, and
// a batch is up to 128 of them, so the shader that killed the queue is
// otherwise unattributable.
struct BatchedDispatch {
  u64 cs_addr;
  u32 groups[3];
  u32 num_res;
  u64 res_base[4];
  u64 res_size[4];
  u8 res_img;           // bit i: resource i is an image
  u8 res_write;         // bit i: resource i is written
  bool traced = false;  // also a zone on the Tracy GPU timeline
  u16 query = 0;
};

extern base::Vector<BatchedDispatch> g_cs_batch_log;
extern rhi::TimestampPool* g_cs_timestamps;

struct CsGpuTime {
  double ns = 0;
  u64 count = 0;
};

extern base::HashMap<u64, CsGpuTime> g_cs_gpu_times;
extern base::Option<u32> kCsHeavyUs;
extern base::HashSet<u64> g_cs_heavy;
extern base::HashMap<rhi::Buffer*, ComputeBufferAccess> g_cs_batch_access;

struct CsBatch {
  rhi::CommandList* list = nullptr;
  bool in_frame = false;  // recorded into the frame list (g_cs_in_frame)
  rhi::TimestampPool* timestamps = nullptr;
  u64 submission = 0;  // the device submission carrying it
  u64 id = 0;
  bool submitted = false;
  u32 count = 0;
  base::Vector<BatchedDispatch> log;
};

extern CsBatch g_cs_batches[kCsBatchRing];
void MarkPending(CsRange& e);

bool CsBatchInit();

// DELTA_GPU_CSSYNC=1: how many submit+wait round trips a frame, and what asked
// for each. The wait itself is the single biggest term in a SotC frame, so the
// question that matters is which call site is forcing it.
enum CsSyncWhy {
  kSyncImported,
  kSyncWriteback,
  kSyncScratchGrow,
  kSyncRangeGrow,
  kSyncStageHazard,
  kSyncBatchCap,
  kSyncFrameEnd,
  kSyncChunk,
  kSyncRingFull,
  kSyncCount
};

const char* const kCsSyncName[kSyncCount] = {
    "imported",  "writeback", "scratch-grow", "range-grow", "stage-hazard",
    "batch-cap", "frame-end", "chunk",        "ring-full"};

extern bool g_cs_chunk_needs_batch;
extern u64 g_cs_sync_n[kSyncCount];
extern u64 g_cs_sync_ns[kSyncCount];
extern const char* g_cs_wait_reader;
extern base::Map<base::String, base::Pair<u64, u64>> g_cs_reader_waits;
extern base::Map<base::String, base::Pair<u64, u64>> g_cs_reader_wbs;

struct WaitReaderScope {
  explicit WaitReaderScope(const char* why) { g_cs_wait_reader = why; }
  ~WaitReaderScope() { g_cs_wait_reader = prev; }
  const char* prev = g_cs_wait_reader;
};

void CsBatchBeginImpl();

void CsCopyStaging(CsRange& e, u64 bytes, bool to_device);

extern u64 g_submit_why[6];

struct SubmittedBatch {
  u64 id = 0, submission = 0;
};

extern SubmittedBatch g_submitted[64];
extern u64 g_last_submitted_id;

bool CsBatchSubmit();

bool CsBatchWaitRange(const CsRange& e, CsSyncWhy why);

bool CsBatchFlush(CsSyncWhy why = kSyncWriteback);
// Reads the written-window flags of a pointer-chasing dispatch nothing waited
// for (compute_dispatch.cc), waiting for it first.
bool SettleUnwaitedWrites();

bool CsLazyKept(u64 base, const CsRange& e);

void CsStageReadback(u64 base, CsRange& e, bool all);

base::Vector<u64> DirtyBases(bool ordered);

void CsStageReadbacks(bool all = true);

bool CsSplitFrameChunk();

bool CsRangeFlushOne(u64 base, CsRange& e);

bool CsEnsureStage(u32 i, u64 size);

struct ScopeCs {
  u64 t0 = NowNs();
  ~ScopeCs() {
    g_ns_cs += NowNs() - t0;
    g_cs_count++;
  }
};

}  // namespace gpu::render::cs

namespace gpu::render {

bool MirrorSupplyTexture(u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear,
                         rhi::Texture* img,
                         rhi::TextureState state,
                         u64* seq);

bool CsRefreshRtInFrame(u64 base, CsRange& e);

extern base::HashMap<u64, rhi::Buffer*> g_guest_page_imports;
extern bool g_labels_deferrable;

constexpr u64 kGuestPageImport = 64 * 1024;

rhi::Buffer* ImportGuestPage(u64 page);

// DELTA_GPU_MIRROR: plain buffers a dispatch binds live in one sparse device
// buffer at offset (guest address - kMirrorBase), a write-through cache of
// guest memory. Before a dispatch the GPU copies the bytes in from the guest
// pages (imported, so no CPU copy and no wait), and after it copies what the
// dispatch wrote back out, as a pending guest write the labels wait for. The
// dispatch itself runs out of VRAM; ranges alias each other because they are
// the same memory.
constexpr u64 kMirrorBase = 0x300000000ull;

constexpr u64 kMirrorSize = 0x1000000000ull;  // through 0x1300000000

extern rhi::Buffer* g_mirror;
extern bool g_mirror_tried;
extern base::HashMap<u64, rhi::Buffer*> g_mirror_imports;

bool MirrorEligible(const ComputeInfo::Res& r, u64 bytes);

// Blocks whose mirror bytes are the guest's: copied in with the block armed
// in the write tracker, and not written by the CPU since. The mirror's own
// write backs are guest writes it already holds.
constexpr u64 kMirrorBlock = 64 * 1024;

extern base::HashSet<u64> g_mirror_clean;
extern base::Vector<base::Pair<u64, u64>> g_mirror_self;

constexpr u64 kOwnBlocks = kMirrorSize / kMirrorBlock;

constexpr u64 kOwnProtected = 1ull << 63, kOwnPending = 1ull << 62,
              kOwnBusy = 1ull << 61, kOwnSubmission = kOwnBusy - 1;

struct OwnBlock {
  base::Atomic<u64> state;  // submission | flags; 0 = the host's
  rhi::Buffer* import;      // the guest pages, as a device buffer
  u64 import_offset;
  u32 lo, hi;  // the span dispatches wrote, block-relative
  base::Atomic<u32> faults;
  base::Atomic<u8> cpu_dirty;  // the CPU wrote it since the mirror copied it
};

extern OwnBlock* g_own;
extern base::Vector<u64> g_own_unsubmitted;
OwnBlock& OwnAt(u64 block);

bool InMirror(u64 address);

void MirrorProtectSubmitted(u64 submission);

bool MirrorCopyIn(Renderer& renderer, u64 base, u64 bytes);

// What dispatches wrote through the mirror: base -> end, and a revision a
// texture made from it compares against.
struct MirrorWrite {
  u64 end = 0, seq = 0;
};

extern base::Map<u64, MirrorWrite> g_mirror_writes;
extern u64 g_mirror_seq;
const MirrorWrite* MirrorWritten(u64 base, u64 bytes);

void MirrorWriteBack(u64 base, u64 bytes);

u64 MirrorRevision(u64 base, u64 bytes);

bool MirrorSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off);

void ReportGpuMemory();

}  // namespace gpu::render

namespace gpu::render::cs {

bool CsBatchOpenOutOfFrame();

}  // namespace gpu::render::cs
