/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

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
#include "gpu/render/compute/compute.h"
#include "gpu/render/compute/hazard.h"
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

extern base::Option<bool> kCsList;
extern base::Option<int> kCsHist;
extern base::Option<bool> kCsRtTrace;
extern base::Option<bool> kCsRename;

extern base::Option<bool> kCsRtCache;
extern base::Option<bool> kCsSubset;

extern base::Option<bool> kCsSkipUpload;
extern u64 g_cs_skip_n;

extern base::Option<bool> kCsVram;

extern base::Option<bool> kCsTexBridge;
extern base::Option<bool> kCsInFrame;
extern bool g_cs_in_frame;

extern base::Option<bool> kCsImageLazyWb;

extern base::Option<bool> kCsTruth;
extern base::Option<bool> kCsMergeOverlap;
extern base::Option<bool> kCsKeepOverlap;
extern base::Option<u32> kCsBatchCap;
extern base::Option<bool> kCsTexBridgeTrace;
extern base::Option<u64> kCsFlushTrace;
extern base::Option<int> kCsOverlapTrace;
extern base::Option<bool> kCsSyncReport;
extern base::Option<int> kCsFlushFrom;

extern base::Option<bool> kCsTrack;
extern base::Option<bool> kCsAged;
bool CsSyncReporting();

extern u64 g_tex_bridge_n, g_tex_bridge_bytes, g_cs_chunk_splits;
extern u64 g_buf_bridge_n, g_buf_bridge_bytes, g_buf_bridge_declined;
extern u64 g_vtx_bridge_n;
extern u64 g_guest_override_n, g_guest_mirror_tiles, g_cs_rt_variants_made;
extern u64 g_gds_imported_n, g_guest_misses_n, g_cs_indirect_n;
extern u64 g_indirect_staged_n;
extern rhi::Buffer* g_indirect_ring;
extern rhi::Buffer* g_predicate;
extern u64 g_predicate_offset;
extern u64 g_indirect_head, g_indirect_frame_n;
extern int g_indirect_frame;
extern u64 g_stage_ro_bytes, g_stage_rw_bytes, g_stage_img_bytes;
extern u64 g_stage_guest_detile_bytes, g_stage_guest_detile_n;
extern u64 g_mirror_wb_bytes, g_mirror_wb_n, g_mirror_own_bytes;
extern base::Atomic<u64> g_mirror_resolves;
extern base::HashSet<u64> g_mirror_cached;
extern base::Vector<base::Pair<u64, u64>> g_mirror_stale;
extern base::HashSet<u64> g_mirror_hot;
extern base::HashSet<u64> g_mirror_was_owned;
constexpr u64 kMirrorHotPage = 4096;
extern base::Map<u64, u64> g_mirror_wb_union;
extern base::Option<bool> kMirror;
extern base::Option<bool> kCsListProf;
extern u64 g_in_hash_ns, g_in_detile_ns, g_in_rt_ns, g_in_copy_ns;
extern u64 g_in_hash_n, g_in_rt_n, g_in_wt_skips;
extern u64 g_in_hash_bytes, g_wb_hash_bytes, g_wb_hash_ns;
extern u64 g_in_overlap_ns, g_in_reshape_ns, g_in_alloc_ns;
extern u64 g_in_membuf_ns;
extern u64 g_in_overlap_n, g_in_reshape_n, g_in_alloc_n;
extern u64 g_truth_stage_bytes, g_truth_import_n, g_truth_detile_n,
    g_truth_retile_n, g_truth_parent_n, g_truth_fold_n, g_truth_rt_n;

extern base::Map<base::String, u64> g_wb_why;
extern u64 g_wb_why_bytes;
extern u64 g_lazy_kept_n, g_lazy_kept_bytes;
extern u64 g_out_rt_submits;

struct StageStat {
  u64 bytes = 0;
  u32 n = 0, first = 0, shape = 0, hash = 0, rt = 0, wait = 0;
  u64 wait_ns = 0;
};

extern base::HashMap<u64, StageStat> g_stage_stats;

struct CsPipe {
  rhi::Pipeline* pipe = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::BindGroupLayout* set_layout = nullptr;
  u32 num_res = 0;
  int gds_binding = -1;
};

struct GdsBuffer {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  static constexpr u64 kBytes = 64 * 1024;
};

extern GdsBuffer g_gds;

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

struct PrebuiltCs {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
};

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

struct CsRange {
  rhi::Buffer* buf = nullptr;
  u64 cap = 0;
  bool device_local = false;
  u64 size = 0;
  u64 guest_bytes = 0;
  u64 hash = 0;
  int last_validated_frame = -1;
  int last_used_frame = -1;
  bool gpu_dirty = false;
  bool pending_batch = false;
  u64 batch_id = 0;
  bool image_staging = false;
  bool imported = false;
  u64 imported_offset = 0;
  bool rt_sourced = false;
  int last_rt_frame = -1;
  u64 rt_serial = 0;
  u64 refreshed_serial = 0;
  ComputeInfo::Res res;
  u64 write_seq = 1;
  int frame_ref = -1;
  u64 chunk_ref = 0;
  bool truth = false;
  u64 rt_seq = 0;
  bool cpu_reader = false;
  base::Vector<base::Pair<u64, u64>> cpu_writes;
  base::Vector<base::Pair<u64, u64>> fill_kept;
  int dirty_frame = -1;
  bool wt_clean = false;
  bool staged_clean = false;
  u64 staged_stamp = 0;
  rhi::Buffer* mirror = nullptr;
  u64 mirror_cap = 0, mirror_seq = 0;
  const struct TileTable* mirror_table = nullptr;
  u64 dirty_stamp = 0;
  u64 dirty_render_serial = 0;
};

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

extern u64 g_cs_scratch_owner;
extern u64 g_cs_scratch_bytes;

extern base::Vector<rhi::Buffer*> g_cs_frame_retired;
bool CsFrameReferenced(const CsRange& e);

extern u64 g_cs_free_bytes, g_cs_recycled_n;

bool SameCsResourceShape(const ComputeInfo::Res& a, const ComputeInfo::Res& b);

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

bool LiveTargetOfSize(u64 base, u32 w, u32 h);

bool CsAliasedBase(u64 base);

u64 AliasedImageRenderSerial(u64 base);

void AliasedImageBarrier(rhi::CommandList* c,
                         const CsAliasedImage& img,
                         rhi::TextureState from,
                         rhi::TextureState to);

struct AliasedCopyPlan {
  u32 w = 0;
  u32 h = 0;
  bool unpack = false;
  bool widen = false;
  bool depth16 = false;
  u32 layers = 1;
};

bool PlanAliasedCopy(const CsAliasedImage& img,
                     const ComputeInfo::Res& res,
                     const char* dir,
                     AliasedCopyPlan& plan);

extern bool g_cs_failed;

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

constexpr u32 kCsRangeBlockShift = 20;

extern base::HashMap<u64, base::Vector<u64>> g_cs_range_blocks;
void IndexCsRange(u64 base, u64 bytes);

void UnindexCsRange(u64 base, u64 bytes);

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
  base::Vector<gcn::ImageTilingParams> params;
  u32 layers = 0;
  bool expanded = false;
};

extern base::HashMap<u64, TileTable> g_tile_tables;

struct TilingState {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
};

extern TilingState g_tiling;

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed,
                 bool pack11);

struct CsScratch {
  rhi::Buffer* buf = nullptr;
  u64 cap = 0;
  u64 owner = 0;
  int last_frame = -1;
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

bool StageImageOnGpu(const ComputeInfo::Res& res, CsRange& e);

extern bool g_cs_batch_open;
extern u32 g_cs_batch_count;

constexpr u32 kCsBatchRing = 128;

extern u64 g_cs_batch_id;
extern u64 g_cs_batch_next_id;
extern u64 g_cs_batch_done;

struct PendingGuestWrite {
  u64 base, end, batch;
};

extern base::Vector<PendingGuestWrite> g_pending_guest_writes;
void NotePendingGuestWrite(u64 base, u64 end);
extern u64 g_publish_batch;
extern u32 g_cs_batch_cur;
extern u64 g_cs_submit_n;

struct BatchedDispatch {
  u64 cs_addr;
  u32 groups[3];
  u32 num_res;
  u64 res_base[4];
  u64 res_size[4];
  u8 res_img;
  u8 res_write;
  bool traced = false;
  bool stamped = false;
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
extern base::HashMap<u64, int> g_cs_heavy_sampled;
extern base::HashMap<rhi::Buffer*, ComputeBufferAccess> g_cs_batch_access;

struct CsBatch {
  rhi::CommandList* list = nullptr;
  bool in_frame = false;
  rhi::TimestampPool* timestamps = nullptr;
  u64 submission = 0;
  u64 id = 0;
  bool submitted = false;
  u32 count = 0;
  base::Vector<BatchedDispatch> log;
};

extern CsBatch g_cs_batches[kCsBatchRing];
void MarkPending(CsRange& e);

bool CsBatchInit();

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

extern u64 g_submit_why[6];

struct SubmittedBatch {
  u64 id = 0, submission = 0;
};

extern SubmittedBatch g_submitted[64];
extern u64 g_last_submitted_id;

bool CsBatchSubmit();

bool CsBatchWaitRange(const CsRange& e, CsSyncWhy why);
bool CsBatchWaitId(u64 id, CsSyncWhy why);
void CsBatchReap();

bool CsBatchFlush(CsSyncWhy why = kSyncWriteback);
void HarvestGuestTables();

bool CsLazyKept(u64 base, const CsRange& e);

base::Vector<u64> DirtyBases(bool ordered);

bool CsSplitFrameChunk();

bool CsRangeFlushOne(u64 base, CsRange& e);

bool CsEnsureStage(u32 i, u64 size);

struct ScopeCs {
  u64 t0 = NowNs();
  ~ScopeCs() {
    g_stats.ns_cs += NowNs() - t0;
    g_stats.cs_count++;
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

constexpr u64 kMirrorPs4Base = 0, kMirrorPs5Base = 0x8000000000ull;
constexpr u64 kMirrorLowWindow = 0x4000000000ull;
constexpr u64 kMirrorWindow = 0x1000000000ull;
constexpr u64 kMirrorSize = kMirrorLowWindow + kMirrorWindow;
constexpr u64 kNotMirrored = ~0ull;
constexpr u64 MirrorOffset(u64 address) {
  if (address - kMirrorPs4Base < kMirrorLowWindow)
    return address - kMirrorPs4Base;
  if (address - kMirrorPs5Base < kMirrorWindow)
    return kMirrorLowWindow + (address - kMirrorPs5Base);
  return kNotMirrored;
}
constexpr bool MirrorCovers(u64 base, u64 bytes) {
  const u64 at = MirrorOffset(base);
  return bytes && at != kNotMirrored &&
         MirrorOffset(base + bytes - 1) == at + bytes - 1;
}

extern rhi::Buffer* g_mirror;
extern bool g_mirror_tried;
extern base::HashMap<u64, rhi::Buffer*> g_mirror_imports;

bool MirrorEligible(const ComputeInfo::Res& r, u64 bytes);

constexpr u64 kMirrorBlock = 64 * 1024;

extern base::HashSet<u64> g_mirror_clean;
extern base::Vector<base::Pair<u64, u64>> g_mirror_self;

constexpr u64 kOwnBlocks = kMirrorSize / kMirrorBlock;

constexpr u64 kOwnProtected = 1ull << 63, kOwnPending = 1ull << 62,
              kOwnBusy = 1ull << 61, kOwnSubmission = kOwnBusy - 1;

constexpr u32 kOwnSpans = 8;

struct OwnBlock {
  base::Atomic<u64> state;
  rhi::Buffer* import;
  u64 import_offset;
  u32 spans;
  u32 lo[kOwnSpans], hi[kOwnSpans];
  base::Atomic<u32> faults;
  base::Atomic<u8> cpu_dirty;
};

extern OwnBlock* g_own;
extern base::Vector<u64> g_own_unsubmitted;
OwnBlock& OwnAt(u64 block);

bool InMirror(u64 address);

void MirrorProtectSubmitted(u64 submission);

bool MirrorCopyIn(Renderer& renderer, u64 base, u64 bytes);
bool MirrorCopyInGuest(u64 base, u64 bytes);
bool GpuWriteBack(u64 base, CsRange& e, bool* kept);
void MirrorFrameRead();
bool MirrorOwned(u64 block);
bool MirrorOwnable(u64 base, u64 bytes);
bool MirrorOwnsAny(u64 first, u64 end);
bool MirrorFillable(u64 base, u64 bytes);
bool GuestImportable(u64 base, u64 bytes);
bool MirrorFill(u64 base, u64 bytes, u32 value);
bool GpuFill(u64 base, u64 bytes, u32 lo, u32 hi);
bool GpuCopy(u64 dst, u64 src, u64 bytes);
bool GpuWriteGuest(u64 dst, u64 bytes, rhi::Buffer* src, u64 src_off);
bool RecordGuestRead(rhi::CommandList* list,
                     u64 base,
                     u64 bytes,
                     rhi::Buffer* dst,
                     u64 dst_off = 0);
u64 PendingGuestWriteBatch(u64 base, u64 bytes);
void CheckSpeculatedReads();

struct MirrorWrite {
  u64 end = 0, seq = 0;
};

extern base::Map<u64, MirrorWrite> g_mirror_writes;
extern u64 g_mirror_seq;
const MirrorWrite* MirrorWritten(u64 base, u64 bytes);

void MirrorWriteBack(u64 base, u64 bytes);

u64 MirrorRevision(u64 base, u64 bytes);
bool MirrorHoldsNewest(u64 base, u64 bytes);

bool MirrorSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off);

void ReportGpuMemory();

}  // namespace gpu::render

namespace gpu::render::cs {

bool CsBatchOpenOutOfFrame();

}  // namespace gpu::render::cs
