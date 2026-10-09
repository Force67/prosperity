/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Compute dispatches. Each guest range a CS touches gets a persistent
// host-visible storage buffer keyed by its base address; dispatches are
// recorded into one batched command buffer, and their writes land back in guest
// memory lazily, only when something needs guest memory to be current.

#include "gpu/render/compute_internal.h"

#include <sys/mman.h>
#include <cstdio>
#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/threading/lock_guard.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gpu_check.h"
#include "gpu/guest_memory.h"
#include "gpu/guest_page_table.h"
#include "gpu/write_tracker.h"
#include "gpu/render/frame.h"
#include "gpu/render/hash.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"

namespace gpu::render::cs {

DELTA_OPTION(bool, kCsList, "DELTA_GPU_CSLIST", false);
DELTA_OPTION(int, kCsHist, "DELTA_GPU_CSHIST", 0);
DELTA_OPTION(bool, kCsRtTrace, "DELTA_GPU_CSRT", false);
DELTA_OPTION(bool, kCsRename, "DELTA_GPU_CSRENAME", false);
// Re-bridge a render target into its compute staging buffer only when
// something has rendered into it since the last copy. =0 restores the
// every-frame copy if a title ever writes a target by a path last_frame does
// not see.
DELTA_OPTION(bool, kCsRtCache, "DELTA_GPU_CS_RT_CACHE", true);
// Reuse a staged plain buffer when a later dispatch binds a smaller window of
// it instead of restaging the range. =0 restores the exact-size match.
DELTA_OPTION(bool, kCsSubset, "DELTA_GPU_CS_SUBSET", true);
// Skip staging in a compute range the shader never reads (see the use in
// compute_dispatch.cc).
DELTA_OPTION(bool, kCsSkipUpload, "DELTA_GPU_CS_SKIP_UPLOAD", false);
u64 g_cs_skip_n = 0;
// Put compute range buffers in VRAM rather than system RAM, with a host-cached
// mirror for the staging edges. The GPU reads and writes these every dispatch,
// and doing that across PCIe was 282 ms of a 457 ms SotC frame. No-op on a UMA
// part, where one buffer is already both (see SplitVram).
DELTA_OPTION(bool, kCsVram, "DELTA_GPU_CSVRAM", true);
// Bind the guest pages of a tiled surface straight into the tiling shader
// instead of copying them through a staging buffer both ways. =0 restores the
// copies for A/B.
// A draw sampling a surface a dispatch wrote copies it VRAM->image on the
// queue instead of writing it back to guest memory, retiling, hashing and
// re-uploading it. =0 restores the guest round trip.
DELTA_OPTION(bool, kCsTexBridge, "DELTA_GPU_CS_TEX_BRIDGE", true);
// Dispatches are recorded into the frame's own command list, in walk order
// with the draws, instead of into batches of their own.
DELTA_OPTION(bool, kCsInFrame, "DELTA_GPU_CS_IN_FRAME", true);
bool g_cs_in_frame = false;  // RecordComputeInFrame, where the option allows it
// An image range a draw took straight from VRAM this frame (CsSupplyTexture)
// stays there at frame end instead of being retiled into guest memory; a
// guest reader that does turn up (CP DMA, a reshaped dispatch, a vertex
// fetch) still gets the writeback on demand. Ranges with any other consumer
// are written back at frame end as before: one batch wait then covers them
// all, where deferring them costs a wait per reader next frame (measured 24 ->
// 32 syncs a frame in Demon's Souls). The risk is a guest CPU read through
// none of those hooks.
DELTA_OPTION(bool, kCsImageLazyWb, "DELTA_GPU_CS_IMAGE_LAZY_WB", true);
// A range holding a gfx10 image footprint keeps the guest bytes as they are
// (tiled). An image binding reads a detiled scratch the tiling shader fills
// inside the batch, and what it writes is retiled back the same way; a plain
// buffer view of the same memory binds the same bytes. No CPU ever tiles, a
// writeback is a copy, and a descriptor's shape stops mattering. =0 restores
// per-shape linear staging.
DELTA_OPTION(bool, kCsTruth, "DELTA_GPU_CS_TRUTH", true);
DELTA_OPTION(bool, kCsMergeOverlap, "DELTA_GPU_CS_MERGE_OVERLAP", true);
DELTA_OPTION(bool, kCsKeepOverlap, "DELTA_GPU_CS_KEEP_OVERLAP", true);
// Dispatches per batch submit. Small batches are what makes the asynchronous
// ring pay: a reader then waits for a few dispatches' worth of GPU work,
// usually already done, instead of a frame's worth.
DELTA_OPTION(u32, kCsBatchCap, "DELTA_GPU_CS_BATCH_CAP", 8);
// Why a sampled surface a dispatch wrote still takes the guest round trip.
DELTA_OPTION(bool, kCsTexBridgeTrace, "DELTA_GPU_CS_TEXBRIDGE_TRACE", false);
// A compute bridge to or from a target drawn into this frame submits the
// frame's open chunk first (SubmitFrameChunk), so it executes after those
// draws instead of a frame early. =0 restores the old one-frame latency.
DELTA_OPTION(bool, kFrameChunks, "DELTA_GPU_FRAME_CHUNKS", true);
DELTA_OPTION(u64, kCsFlushTrace, "DELTA_GPU_CSFLUSHTRACE", 0);
// DELTA_GPU_CSOVERLAP=<frame>: from that frame on, name the binding and the
// dirty range behind the next "overlap" writebacks (a dispatch input
// straddling another's unflushed output).
DELTA_OPTION(int, kCsOverlapTrace, "DELTA_GPU_CSOVERLAP", 0);
DELTA_OPTION(bool, kCsSyncReport, "DELTA_GPU_CSSYNC", false);
DELTA_OPTION(int, kCsSyncFrom, "DELTA_GPU_CSSYNC_FROM", 0);
// Every guest read that writes dispatch output back, and its cost, from this
// frame on (a few frames' worth).
DELTA_OPTION(int, kCsFlushFrom, "DELTA_GPU_CSFLUSH_FROM", 0);
// Off by default: with the CP DMA flush narrowed, ranges a dispatch rewrites
// every frame stay dirty indefinitely, and GTA:SA then shows ghosted
// post-processing (DELTA_GPU_CS_AGED lists those ranges). Some reader of them
// still misses its flush; the global DMA flush hides it.
DELTA_OPTION(bool, kCsTrack, "DELTA_GPU_CS_TRACK", false);
DELTA_OPTION(bool, kCsAged, "DELTA_GPU_CS_AGED", false);
bool CsSyncReporting() {
  return kCsSyncReport && gpu::render::g_frame.num >= kCsSyncFrom;
}

// Writeback-half accounting (DELTA_GPU_CSSYNC), counting what the aliased-image
// bridge (compute_alias.cc) does among others.
u64 g_tex_bridge_n = 0, g_tex_bridge_bytes = 0, g_cs_chunk_splits = 0;
u64 g_buf_bridge_n = 0, g_buf_bridge_bytes = 0, g_buf_bridge_declined = 0;
u64 g_vtx_bridge_n = 0;
u64 g_guest_override_n = 0, g_guest_mirror_tiles = 0, g_cs_rt_variants_made = 0;
u64 g_gds_imported_n = 0, g_guest_misses_n = 0, g_cs_indirect_n = 0;
u64 g_stage_ro_bytes = 0, g_stage_rw_bytes = 0, g_stage_img_bytes = 0;
u64 g_stage_guest_detile_bytes = 0, g_stage_guest_detile_n = 0;
u64 g_mirror_wb_bytes = 0, g_mirror_wb_n = 0, g_mirror_own_bytes = 0;
base::Atomic<u64> g_mirror_resolves{0};
base::HashSet<u64> g_mirror_cached;
base::Map<u64, u64> g_mirror_uncached;  // block -> frame the CPU last wrote it
base::Map<u64, u64> g_mirror_wb_union;
DELTA_OPTION(bool, kMirror, "DELTA_GPU_MIRROR", true);
DELTA_OPTION(bool, kCsListProf, "DELTA_GPU_LISTPROF", false);
// Staging-in halves: hashing guest memory to decide validity, CPU detiling,
// the render-target bridge (its own submit+wait), and the plain copy.
u64 g_in_hash_ns = 0, g_in_detile_ns = 0, g_in_rt_ns = 0, g_in_copy_ns = 0;
u64 g_in_hash_n = 0, g_in_rt_n = 0, g_in_wt_skips = 0;
u64 g_in_hash_bytes = 0, g_wb_hash_bytes = 0, g_wb_hash_ns = 0;
// The rest of a stage-in: writebacks forced by an overlapping dirty range or a
// reshape (each a GPU wait plus a retile), buffer allocation, the plain guest
// memcpy and its shadow copy. Without these `in=` is mostly unattributed.
u64 g_in_overlap_ns = 0, g_in_reshape_ns = 0, g_in_alloc_ns = 0;
u64 g_in_membuf_ns = 0;
u64 g_in_overlap_n = 0, g_in_reshape_n = 0, g_in_alloc_n = 0;
u64 g_truth_stage_bytes = 0, g_truth_import_n = 0, g_truth_detile_n = 0,
    g_truth_retile_n = 0, g_truth_parent_n = 0, g_truth_fold_n = 0,
    g_truth_rt_n = 0;
// Who forced each writeback: the reader tag of FlushCsWritesRange, or one of
// the internal sites.
base::Map<base::String, u64> g_wb_why;
u64 g_wb_why_bytes = 0;
u64 g_lazy_kept_n = 0, g_lazy_kept_bytes = 0;
u64 g_out_rt_submits = 0;

void ParallelCopy(void* dst, const void* src, u64 bytes) {
  if (bytes < kParallelCopyMin) {
    std::memcpy(dst, src, bytes);
    return;
  }
  constexpr u64 kChunk = 256 * 1024;
  const u32 chunks = static_cast<u32>((bytes + kChunk - 1) / kChunk);
  auto* d = static_cast<u8*>(dst);
  const auto* s = static_cast<const u8*>(src);
  gcn::DetileParallelWork(chunks, bytes, [&](u32 c0, u32 c1) {
    const u64 off = u64(c0) * kChunk;
    std::memcpy(d + off, s + off,
                base::Min<u64>(u64(c1 - c0) * kChunk, bytes - off));
  });
}

base::HashMap<u64, StageStat> g_stage_stats;

static_assert(ComputeInfo::kMaxResources == gcn::kMaxCsResources);

GdsBuffer g_gds;

base::HashMap<u64, CsPipe> g_cs_pipes;

// A buffer the GPU alone touches goes to VRAM, with a separate host mirror,
// only when the device heap is NOT already cheap for the CPU to read. On a UMA
// part one host-cached buffer serves both sides.
bool SplitVram() {
  const bool split = !Device().caps().unified_memory;
  static bool logged = false;
  if (!logged) {
    logged = true;
    BASE_LOGI("gpuvk", "cs buffers: {}",
              split ? "VRAM with a host-cached mirror"
                    : "one host-cached buffer (unified memory)");
  }
  return split;
}

// kReadback is the host-cached memory the CPU side of compute wants: reading
// back from a write-combined heap is ~100 MB/s, which is invisible in the code
// and obvious in the numbers.
rhi::Buffer* CreateCsBuffer(u64 size, u32 usage, rhi::MemoryKind memory) {
  rhi::BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = memory;
  return Device().CreateBuffer(desc);
}

// The set-0 layout every compute pipeline over `num_res` guest ranges uses.
rhi::BindGroupLayout* CsSetLayout(u32 num_res,
                                  int gds_binding,
                                  int guest_memory_binding,
                                  const base::Vector<int>& textures) {
  rhi::BindGroupLayoutDesc desc;
  for (int binding : textures)
    if (binding >= 0)
      desc.bindings.push_back({static_cast<u32>(binding),
                               rhi::BindingType::kSampledTexture,
                               rhi::kStageCompute});
  for (u32 i = 0; i < num_res; i++)
    desc.bindings.push_back(
        {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
  // The GDS scratchpad sits past the resources; it is ours, not a guest range.
  if (gds_binding >= 0)
    desc.bindings.push_back({static_cast<u32>(gds_binding),
                             rhi::BindingType::kStorageBuffer,
                             rhi::kStageCompute});
  if (guest_memory_binding >= 0)
    desc.bindings.push_back({static_cast<u32>(guest_memory_binding),
                             rhi::BindingType::kStorageBuffer,
                             rhi::kStageCompute});
  return Device().CreateBindGroupLayout(desc);
}

// 16 user-data dwords, then one bound (in dwords) per SSBO binding. The bound
// is what the emitted SSBO accesses clamp to; the SPIR-V block for a compute
// stage declares the same 16 + 48 shape, 256 B total, the driver max this
// backend runs against.
rhi::PipelineLayout* CsPipelineLayout(rhi::BindGroupLayout* set_layout) {
  rhi::PipelineLayoutDesc desc;
  desc.groups = {set_layout};
  desc.push_constant_bytes = 64 * 4;
  desc.push_constant_stages = rhi::kStageCompute;
  return Device().CreatePipelineLayout(desc);
}

rhi::Pipeline* CreateCsPipeline(rhi::PipelineLayout* layout,
                                const base::Vector<u32>& spirv) {
  rhi::ComputePipelineDesc desc;
  desc.layout = layout;
  desc.code = rhi::Code(spirv);
  desc.dispatch_base = true;
  return Device().CreateComputePipeline(desc);
}

base::Mutex g_prebuilt_mutex;
base::HashMap<u64, base::SharedPointer<PrebuiltSlot>> g_prebuilt;

u64 PrebuiltKey(const base::Vector<u32>& spirv,
                u32 num_res,
                int guest_memory_binding) {
  u64 h = 1469598103934665603ull;
  for (u32 w : spirv)
    h = (h ^ w) * 1099511628211ull;
  return HashWord(HashWord(h, num_res), static_cast<u32>(guest_memory_binding));
}

CsPipe* GetCsPipe(const ComputeInfo& ci) {
  const u64 key = reinterpret_cast<uintptr_t>(ci.recomp);
  auto it = g_cs_pipes.find(key);
  if (it != g_cs_pipes.end())
    return it->second.num_res == ci.num_res ? &it->second : nullptr;
  CsPipe cp;
  cp.num_res = ci.num_res;
  cp.gds_binding = ci.gds_binding;
  if (ci.gds_binding < 0 && ci.recomp->textures.empty()) {
    base::SharedPointer<PrebuiltSlot> slot;
    {
      base::LockGuard<base::Mutex> lock(g_prebuilt_mutex);
      const auto found = g_prebuilt.find(PrebuiltKey(
          ci.recomp->spirv, ci.num_res, ci.recomp->guest_memory_binding));
      if (found != g_prebuilt.end()) {
        slot = found->second;
        g_prebuilt.erase(found);
      }
    }
    PrebuiltCs prebuilt;
    if (slot) {
      base::UniqueLock<base::Mutex> lock(slot->mutex);
      slot->built.Wait(lock, [&] { return slot->ready; });
      prebuilt = slot->value;
    }
    if (prebuilt.pipe) {
      const u64 started = NowNs();
      cp.set_layout = prebuilt.set_layout;
      cp.layout = prebuilt.layout;
      cp.pipe = prebuilt.pipe;
      g_ns_pipe_build += NowNs() - started;
      g_pipe_build_n++;
      g_cs_pipes[key] = cp;
      return &g_cs_pipes[key];
    }
  }
  cp.set_layout = CsSetLayout(ci.num_res, ci.gds_binding,
                              ci.recomp->guest_memory_binding,
                              ci.recomp->textures);
  if (!cp.set_layout)
    return nullptr;
  cp.layout = CsPipelineLayout(cp.set_layout);
  if (!cp.layout) {
    Device().Destroy(cp.set_layout);
    return nullptr;
  }
  if (CsSyncReporting())
    BASE_LOGI("cspipe", "compile cs={:#x} words={}", ci.cs_addr,
              ci.recomp->spirv.size());
  const u64 started = NowNs();
  cp.pipe = CreateCsPipeline(cp.layout, ci.recomp->spirv);
  g_ns_pipe_build += NowNs() - started;
  if (CsSyncReporting())
    BASE_LOGI("cspipe", "compiled cs={:#x} ms={:.3f}", ci.cs_addr,
              double(NowNs() - started) / 1e6);

  g_pipe_build_n++;
  if (!cp.pipe) {
    BASE_LOGI("gpuvk", "compute pipeline failed");
    Device().Destroy(cp.layout);
    Device().Destroy(cp.set_layout);
    return nullptr;
  }
  if (Device().caps().debug_labels) {
    char name[32];
    std::snprintf(name, sizeof(name), "cs %#llx",
                  (unsigned long long)ci.cs_addr);
    Device().SetName(cp.pipe, name);
  }
  g_cs_pipes[key] = cp;
  return &g_cs_pipes[key];
}

CsStage g_cs_stage[ComputeInfo::kMaxResources];
// The open batch's command list.
rhi::CommandList* g_cs_list = nullptr;

// `packed`: narrow texels stay packed on the linear side (a texture upload)
// instead of being expanded to dwords (a shader view).
// A scratch that already holds revision `seq` of the bytes at `base`+`off`
// converted with `table`, or a fresh one (`*fresh` says which).
// What a scratch holds once the conversion recorded into it has run.
// Whether the open batch records into a list of its own (between frames).
// The linear side of a bridge copy for a truth range (its own buffer holds
// tiled bytes), used only inside the bridge's synchronous submit.
CsRange g_truth_bridge_scratch;
u64 g_cs_scratch_owner = 0;  // the dispatch or bridge being recorded
u64 g_cs_scratch_bytes = 0;

// Buffers a frame command list reads: destroyed two BeginFrames after
// retirement, like ReleaseRetiredTextures, once every command list that
// could reference them has retired.
base::Vector<rhi::Buffer*> g_cs_frame_retired;

bool CsFrameReferenced(const CsRange& e) {
  return e.frame_ref >= 0 && e.frame_ref + 2 > g_frame.num;
}

// Evicted range buffers kept by capacity for the next range of that size.
// The guest hands compute a fresh address for its per-frame data every
// frame, so without this every frame was ~50 buffer+memory pairs allocated
// and 20-60 ms of driver time.
struct CsFreeBuffer {
  rhi::Buffer* buf = nullptr;
  rhi::Buffer* host_buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
  bool device_local = false;
};
base::Vector<CsFreeBuffer> g_cs_free;
u64 g_cs_free_bytes = 0, g_cs_recycled_n = 0;

// Move a split range between its host mirror and its VRAM buffer, bracketed by
// barriers against the dispatches on either side. Recorded rather than
// submitted, so the caller decides which command list carries it.
void RecordStagingCopy(rhi::CommandList* c,
                       CsRange& e,
                       u64 bytes,
                       bool to_device) {
  if (!e.device_local || !e.host_buf || !e.buf || !bytes)
    return;
  if (bytes > e.cap)
    bytes = e.cap;
  // TRANSFER on both sides, not just compute and host. A batch stages the same
  // range more than once (GTA:SA restages some buffers ~170 times in one cs
  // batch), and a barrier that names only COMPUTE|HOST as its source leaves
  // copy-after-copy on the same buffer unordered. The later copy may then land
  // first and the dispatch reads the OLDER guest snapshot.
  c->Barrier(kAccessComputeRW | rhi::kAccessHostWrite | kAccessCopyRW,
             to_device ? rhi::kAccessCopyWrite : rhi::kAccessCopyRead);
  if (to_device)
    c->CopyBuffer(e.buf, 0, e.host_buf, 0, bytes);
  else
    c->CopyBuffer(e.host_buf, 0, e.buf, 0, bytes);
  // Global, so it also covers host_buf, which the readback direction writes.
  c->Barrier(rhi::kAccessCopyWrite,
             kAccessComputeRW | rhi::kAccessHostRead | kAccessCopyRW);
}

// The GDS scratchpad, created once. Zeroed at creation; after that it is the
// title's own counter store, so nothing here resets it.
bool EnsureGdsBuffer() {
  if (g_gds.buf)
    return true;
  g_gds.buf = CreateCsBuffer(
      GdsBuffer::kBytes,
      rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc,
      rhi::MemoryKind::kReadback);
  if (!g_gds.buf)
    return false;
  g_gds.map = g_gds.buf->mapped();
  if (!g_gds.map) {
    Device().Destroy(g_gds.buf);
    g_gds = {};
    return false;
  }
  std::memset(g_gds.map, 0, GdsBuffer::kBytes);
  return true;
}

void CsRangeDestroy(CsRange& e);

base::HashMap<u64, CsRange>::iterator EraseCsRange(
    base::HashMap<u64, CsRange>::iterator it) {
  if (it == g_cs_evict_next)
    ++g_cs_evict_next;
  UnindexCsRange(it->first, it->second.guest_bytes);
  CsRangeDestroy(it->second);
  return g_cs_ranges.erase(it);
}

// Device memory ran out: give back the recycled buffers and every range no
// dispatch has touched for a few frames, instead of dropping the dispatch.
// `keep` is the range being allocated for. True when anything was released.
bool CsReleaseMemory(const CsRange& keep) {
  const auto drop_free_list = [] {
    const bool any = !g_cs_free.empty();
    for (const CsFreeBuffer& f : g_cs_free) {
      Device().Destroy(f.buf);
      Device().Destroy(f.host_buf);
    }
    g_cs_free.clear();
    g_cs_free_bytes = 0;
    return any;
  };
  const bool released = drop_free_list();
  u64 evicted = 0;
  for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
    CsRange& r = it->second;
    if (&r == &keep || r.gpu_dirty || r.pending_batch || CsFrameReferenced(r) ||
        r.last_used_frame + 2 >= g_frame.num) {
      ++it;
      continue;
    }
    evicted += r.cap;
    it = EraseCsRange(it);
  }
  // CsRangeDestroy parks what it frees for reuse; that is the opposite of
  // what is wanted here.
  drop_free_list();
  static int logged = 0;
  if (logged++ < 16)
    BASE_LOGI("csgpu", "device memory full: evicted {} MB of idle ranges",
              evicted >> 20);
  return released || evicted;
}

bool CsRangeEnsureBuffer(CsRange& e, u64 size) {
  if (e.buf && e.cap >= size)
    return true;
  e.map = nullptr;
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.push_back(e.buf);
    g_cs_frame_retired.push_back(e.host_buf);
    e.frame_ref = -1;
    e.pending_batch = false;
  } else {
    Device().Destroy(e.buf);
    Device().Destroy(e.host_buf);
  }
  e.buf = nullptr;
  e.host_buf = nullptr;
  e.device_local = false;
  e.imported = false;
  if (!e.imported)
    g_cs_range_bytes -= e.cap;
  e.cap = 0;
  u64 cap = (size + 0xFFFF) & ~u64(0xFFFF);
  for (size_t i = g_cs_free.size(); i-- > 0;) {
    const CsFreeBuffer& f = g_cs_free[i];
    if (f.cap != cap || f.device_local != static_cast<bool>(kCsVram))
      continue;
    e.buf = f.buf;
    e.host_buf = f.host_buf;
    e.map = f.map;
    e.device_local = f.device_local;
    e.cap = cap;
    g_cs_free_bytes -= cap;
    g_cs_free.erase(g_cs_free.begin() + i);
    g_cs_range_bytes += cap;
    g_cs_recycled_n++;
    return true;
  }
  // CopyDst: RT-backed inputs are staged by an image->buffer copy on the
  // queue (StageCsRangeFromRt) instead of a CPU memcpy from guest memory.
  // Vertex: a draw may fetch a dispatch's output in place (CsVertexBuffer).
  u32 usage = rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferVertex |
              rhi::kBufferIndirect;
  // A pointer-chasing dispatch may read the range in place (GuestOverrides).
  if (Device().caps().buffer_address)
    usage |= rhi::kBufferAddress;
  if (kCsVram)
    usage |= rhi::kBufferCopySrc;  // readback of results
  const bool split = kCsVram && SplitVram();
  const rhi::MemoryKind memory =
      split ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback;
  e.buf = CreateCsBuffer(cap, usage, memory);
  if (!e.buf &&
      (!CsReleaseMemory(e) || !(e.buf = CreateCsBuffer(cap, usage, memory))))
    return false;
  // VRAM cannot be mapped usefully (uncached reads are ~100 MB/s), so the CPU
  // side gets its own host-cached buffer and the two are joined by DMA.
  if (split) {
    e.host_buf = CreateCsBuffer(cap, rhi::kBufferCopySrc | rhi::kBufferCopyDst,
                                rhi::MemoryKind::kReadback);
    if (!e.host_buf)
      return false;
    e.map = e.host_buf->mapped();
    e.device_local = true;
  } else {
    e.map = e.buf->mapped();
  }
  e.cap = cap;
  g_cs_range_bytes += cap;
  return true;
}

// DELTA_GPU_CSRENAME: a CPU write into a range the open batch already reads has
// to flush that batch and wait on the GPU. SotC's streaming does that 37 times
// a frame, and staging then costs 272 ms of a 2 fps frame. Giving the range a
// FRESH buffer and retiring the old one until the batch completes takes that to
// 24 ms: the recorded descriptors keep reading the contents they were built
// against, and this dispatch binds the new one. Only safe while nothing is
// waiting to read results back OUT of the old buffer, hence the caller's
// gpu_dirty/written check. Default OFF: it moves cost into the writeback rather
// than removing it (SotC gains ~7% overall), and no title that currently
// renders through the compute path could be used to gate the change.
bool CsRangeRename(CsRange& e, u64 size) {
  e.map = nullptr;
  if (e.buf)
    g_cs_frame_retired.push_back(e.buf);
  if (e.host_buf)
    g_cs_frame_retired.push_back(e.host_buf);
  e.frame_ref = -1;
  e.pending_batch = false;
  g_cs_range_bytes -= e.cap;
  e.buf = nullptr;
  e.host_buf = nullptr;
  e.device_local = false;
  e.cap = 0;
  return CsRangeEnsureBuffer(e, size);
}

void CsRangeDestroy(CsRange& e) {
  base::EraseIf(g_cs_pending, [&e](CsRange* r) { return r == &e; });
  if (e.mirror)
    g_cs_frame_retired.push_back(e.mirror);
  if (!e.imported && !CsFrameReferenced(e) && !e.pending_batch && e.buf &&
      e.cap && g_cs_free_bytes + e.cap <= (256ull << 20)) {
    g_cs_free.push_back({e.buf, e.host_buf, e.map, e.cap, e.device_local});
    g_cs_free_bytes += e.cap;
    g_cs_range_bytes -= e.cap;
    e = CsRange{};
    return;
  }
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.push_back(e.buf);
    g_cs_frame_retired.push_back(e.host_buf);
  } else {
    Device().Destroy(e.buf);
    Device().Destroy(e.host_buf);
  }
  g_cs_range_bytes -= e.cap;
  e = CsRange{};
}

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool CsEnsureStage(u32 i, u64 size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs_stage[i];
  if (s.buf && s.cap >= size)
    return true;
  s.map = nullptr;
  // A batch in flight may still bind the old scratch: it outlives the frame.
  if (s.buf)
    g_cs_frame_retired.push_back(s.buf);
  s.buf = nullptr;
  const u64 cap = (size + 0xFFFFF) & ~u64(0xFFFFF);  // 1 MiB granularity
  // Scratch is zeroed by FillBuffer and only ever touched by shaders, so it
  // wants VRAM and no mapping at all.
  const bool vram = kCsVram && SplitVram();
  s.buf = CreateCsBuffer(
      cap, rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc,
      vram ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback);
  if (!s.buf)
    return false;
  if (!vram)
    s.map = s.buf->mapped();
  s.cap = cap;
  return true;
}

}  // namespace gpu::render::cs

namespace gpu::render {

void ReleaseRetiredCsBuffers() {
  static base::Vector<rhi::Buffer*> aged;
  static const guest::SessionReset reset_aged(
      [] { guest::ResetResource(aged); });
  for (rhi::Buffer* buf : aged)
    Device().Destroy(buf);
  aged = base::move(g_cs_frame_retired);
  g_cs_frame_retired.clear();
}

}  // namespace gpu::render

namespace gpu::render::cs {

const guest::SessionReset g_session_reset([] {
  host_memory::SetWriteFaultHandler(nullptr);
  host_memory::SetHostWriteHook(nullptr);
  if (g_own)
    munmap(g_own, kOwnBlocks * sizeof(OwnBlock));
  g_own = nullptr;
  g_cs_writeback_gen = g_cs_batch_next_id = 1;
  guest::ResetResource(g_mirror_cached);
  guest::ResetResource(g_mirror_uncached);
  guest::ResetResource(g_mirror_wb_union);
  guest::ResetResource(g_wb_why);
  guest::ResetResource(g_stage_stats);
  guest::ResetResource(g_gds);
  guest::ResetResource(g_cs_pipes);
  guest::ResetResource(g_prebuilt);
  guest::ResetResource(g_cs_stage);
  guest::ResetResource(g_cs_list);
  guest::ResetResource(g_truth_bridge_scratch);
  guest::ResetResource(g_cs_scratch_owner);
  guest::ResetResource(g_cs_scratch_bytes);
  guest::ResetResource(g_cs_frame_retired);
  guest::ResetResource(g_cs_free);
  guest::ResetResource(g_cs_free_bytes);
  guest::ResetResource(g_bridge);
  guest::ResetResource(g_cs_failed);
  guest::ResetResource(g_bridge_float_images);
  guest::ResetResource(g_rt_unbridgeable);
  guest::ResetResource(g_cs_ranges);
  guest::ResetResource(g_cs_evict_next);
  guest::ResetResource(g_cs_dirty_stamp);
  guest::ResetResource(g_cs_pending);
  guest::ResetResource(g_cs_range_bytes);
  guest::ResetResource(g_cs_dirty_pages);
  guest::ResetResource(g_cs_dirty_bases);
  guest::ResetResource(g_cs_dirty_lo);
  guest::ResetResource(g_cs_range_blocks);
  guest::ResetResource(g_tile_tables);
  guest::ResetResource(g_cs_scratch);
  guest::ResetResource(g_cs_batch_open);
  guest::ResetResource(g_cs_batch_count);
  guest::ResetResource(g_cs_batch_id);
  guest::ResetResource(g_cs_batch_done);
  guest::ResetResource(g_pending_guest_writes);
  guest::ResetResource(g_publish_batch);
  guest::ResetResource(g_cs_batch_cur);
  guest::ResetResource(g_cs_batch_log);
  guest::ResetResource(g_cs_timestamps);
  guest::ResetResource(g_cs_gpu_times);
  guest::ResetResource(g_cs_heavy);
  guest::ResetResource(g_cs_batch_access);
  guest::ResetResource(g_cs_batches);
  guest::ResetResource(g_cs_chunk_needs_batch);
  guest::ResetResource(g_cs_reader_waits);
  guest::ResetResource(g_cs_reader_wbs);
  guest::ResetResource(g_submitted);
  guest::ResetResource(g_last_submitted_id);
  guest::ResetResource(g_guest_page_imports);
  guest::ResetResource(g_labels_deferrable);
  guest::ResetResource(g_mirror);
  guest::ResetResource(g_mirror_tried);
  guest::ResetResource(g_mirror_imports);
  guest::ResetResource(g_mirror_clean);
  guest::ResetResource(g_mirror_self);
  guest::ResetResource(g_own_unsubmitted);
  guest::ResetResource(g_mirror_writes);
  guest::ResetResource(g_mirror_seq);
});

}  // namespace gpu::render::cs
