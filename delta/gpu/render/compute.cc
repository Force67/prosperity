/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

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
#include "gpu/render/frame.h"
#include "gpu/render/hash.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"

namespace gpu::render::cs {

DELTA_OPTION(bool, kCsList, "DELTA_GPU_CSLIST", false);
DELTA_OPTION(int, kCsHist, "DELTA_GPU_CSHIST", 0);
DELTA_OPTION(bool, kCsRtTrace, "DELTA_GPU_CSRT", false);
DELTA_OPTION(bool, kCsRename, "DELTA_GPU_CSRENAME", false);
DELTA_OPTION(bool, kCsRtCache, "DELTA_GPU_CS_RT_CACHE", true);
DELTA_OPTION(bool, kCsSubset, "DELTA_GPU_CS_SUBSET", true);
DELTA_OPTION(bool, kCsSkipUpload, "DELTA_GPU_CS_SKIP_UPLOAD", false);
u64 g_cs_skip_n = 0;
DELTA_OPTION(bool, kCsVram, "DELTA_GPU_CSVRAM", true);
DELTA_OPTION(bool, kCsTexBridge, "DELTA_GPU_CS_TEX_BRIDGE", true);
DELTA_OPTION(bool, kCsInFrame, "DELTA_GPU_CS_IN_FRAME", true);
bool g_cs_in_frame = false;
DELTA_OPTION(bool, kCsImageLazyWb, "DELTA_GPU_CS_IMAGE_LAZY_WB", true);
DELTA_OPTION(bool, kCsTruth, "DELTA_GPU_CS_TRUTH", true);
DELTA_OPTION(bool, kCsMergeOverlap, "DELTA_GPU_CS_MERGE_OVERLAP", true);
DELTA_OPTION(bool, kCsKeepOverlap, "DELTA_GPU_CS_KEEP_OVERLAP", true);
DELTA_OPTION(u32, kCsBatchCap, "DELTA_GPU_CS_BATCH_CAP", 8);
DELTA_OPTION(bool, kCsTexBridgeTrace, "DELTA_GPU_CS_TEXBRIDGE_TRACE", false);
DELTA_OPTION(u64, kCsFlushTrace, "DELTA_GPU_CSFLUSHTRACE", 0);
DELTA_OPTION(int, kCsOverlapTrace, "DELTA_GPU_CSOVERLAP", 0);
DELTA_OPTION(bool, kCsSyncReport, "DELTA_GPU_CSSYNC", false);
DELTA_OPTION(int, kCsSyncFrom, "DELTA_GPU_CSSYNC_FROM", 0);
DELTA_OPTION(int, kCsFlushFrom, "DELTA_GPU_CSFLUSH_FROM", 0);
DELTA_OPTION(bool, kCsTrack, "DELTA_GPU_CS_TRACK", false);
DELTA_OPTION(bool, kCsAged, "DELTA_GPU_CS_AGED", false);
bool CsSyncReporting() {
  return kCsSyncReport && gpu::render::g_frame.num >= kCsSyncFrom;
}

u64 g_tex_bridge_n = 0, g_tex_bridge_bytes = 0, g_cs_chunk_splits = 0;
u64 g_buf_bridge_n = 0, g_buf_bridge_bytes = 0, g_buf_bridge_declined = 0;
u64 g_vtx_bridge_n = 0;
u64 g_guest_override_n = 0, g_guest_mirror_tiles = 0, g_cs_rt_variants_made = 0;
u64 g_gds_imported_n = 0, g_guest_misses_n = 0, g_cs_indirect_n = 0;
u64 g_indirect_staged_n = 0;
rhi::Buffer* g_indirect_ring = nullptr;
u64 g_indirect_head = 0, g_indirect_frame_n = 0;
int g_indirect_frame = -1;
rhi::Buffer* g_predicate = nullptr;
u64 g_predicate_offset = 0;
u64 g_stage_ro_bytes = 0, g_stage_rw_bytes = 0, g_stage_img_bytes = 0;
u64 g_stage_guest_detile_bytes = 0, g_stage_guest_detile_n = 0;
u64 g_mirror_wb_bytes = 0, g_mirror_wb_n = 0, g_mirror_own_bytes = 0;
base::Atomic<u64> g_mirror_resolves{0};
base::HashSet<u64> g_mirror_cached;
base::Vector<base::Pair<u64, u64>> g_mirror_stale;
base::HashSet<u64> g_mirror_hot;
base::HashSet<u64> g_mirror_was_owned;
base::Map<u64, u64> g_mirror_wb_union;
DELTA_OPTION(bool, kMirror, "DELTA_GPU_MIRROR", true);
DELTA_OPTION(bool, kCsListProf, "DELTA_GPU_LISTPROF", false);
u64 g_in_hash_ns = 0, g_in_detile_ns = 0, g_in_rt_ns = 0, g_in_copy_ns = 0;
u64 g_in_hash_n = 0, g_in_rt_n = 0, g_in_wt_skips = 0;
u64 g_in_hash_bytes = 0, g_wb_hash_bytes = 0, g_wb_hash_ns = 0;
u64 g_in_overlap_ns = 0, g_in_reshape_ns = 0, g_in_alloc_ns = 0;
u64 g_in_membuf_ns = 0;
u64 g_in_overlap_n = 0, g_in_reshape_n = 0, g_in_alloc_n = 0;
u64 g_truth_stage_bytes = 0, g_truth_import_n = 0, g_truth_detile_n = 0,
    g_truth_retile_n = 0, g_truth_parent_n = 0, g_truth_fold_n = 0,
    g_truth_rt_n = 0;
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

rhi::Buffer* CreateCsBuffer(u64 size, u32 usage, rhi::MemoryKind memory) {
  rhi::BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = memory;
  return Device().CreateBuffer(desc);
}

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
  cp.set_layout =
      CsSetLayout(ci.num_res, ci.gds_binding, ci.recomp->guest_memory_binding,
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
rhi::CommandList* g_cs_list = nullptr;

u64 g_cs_scratch_owner = 0;
u64 g_cs_scratch_bytes = 0;

base::Vector<rhi::Buffer*> g_cs_frame_retired;

bool CsFrameReferenced(const CsRange& e) {
  return e.frame_ref >= 0 && e.frame_ref + 2 > g_frame.num;
}

struct CsFreeBuffer {
  rhi::Buffer* buf = nullptr;
  u64 cap = 0;
  bool device_local = false;
};
base::Vector<CsFreeBuffer> g_cs_free;
u64 g_cs_free_bytes = 0, g_cs_recycled_n = 0;

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

bool CsReleaseMemory(const CsRange& keep) {
  const auto drop_free_list = [] {
    const bool any = !g_cs_free.empty();
    for (const CsFreeBuffer& f : g_cs_free)
      Device().Destroy(f.buf);
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
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.push_back(e.buf);
    e.frame_ref = -1;
    e.pending_batch = false;
  } else {
    Device().Destroy(e.buf);
  }
  e.buf = nullptr;
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
    e.device_local = f.device_local;
    e.cap = cap;
    g_cs_free_bytes -= cap;
    g_cs_free.erase(g_cs_free.begin() + i);
    g_cs_range_bytes += cap;
    g_cs_recycled_n++;
    return true;
  }
  u32 usage = rhi::kBufferStorage | rhi::kBufferUniform | rhi::kBufferCopyDst |
              rhi::kBufferVertex | rhi::kBufferIndirect | rhi::kBufferIndex;
  if (Device().caps().buffer_address)
    usage |= rhi::kBufferAddress;
  usage |= rhi::kBufferCopySrc;
  const bool split = kCsVram && SplitVram();
  const rhi::MemoryKind memory =
      split ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback;
  e.buf = CreateCsBuffer(cap, usage, memory);
  if (!e.buf &&
      (!CsReleaseMemory(e) || !(e.buf = CreateCsBuffer(cap, usage, memory))))
    return false;
  e.device_local = split;
  e.cap = cap;
  g_cs_range_bytes += cap;
  return true;
}

bool CsRangeRename(CsRange& e, u64 size) {
  if (e.buf)
    g_cs_frame_retired.push_back(e.buf);
  e.frame_ref = -1;
  e.pending_batch = false;
  g_cs_range_bytes -= e.cap;
  e.buf = nullptr;
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
    g_cs_free.push_back({e.buf, e.cap, e.device_local});
    g_cs_free_bytes += e.cap;
    g_cs_range_bytes -= e.cap;
    e = CsRange{};
    return;
  }
  if (CsFrameReferenced(e) || e.pending_batch)
    g_cs_frame_retired.push_back(e.buf);
  else
    Device().Destroy(e.buf);
  g_cs_range_bytes -= e.cap;
  e = CsRange{};
}

bool CsEnsureStage(u32 i, u64 size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs_stage[i];
  if (s.buf && s.cap >= size)
    return true;
  s.map = nullptr;
  if (s.buf)
    g_cs_frame_retired.push_back(s.buf);
  s.buf = nullptr;
  const u64 cap = (size + 0xFFFFF) & ~u64(0xFFFFF);
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
  guest::ResetResource(g_mirror_stale);
  guest::ResetResource(g_mirror_hot);
  guest::ResetResource(g_mirror_was_owned);
  guest::ResetResource(g_mirror_wb_union);
  guest::ResetResource(g_wb_why);
  guest::ResetResource(g_stage_stats);
  guest::ResetResource(g_gds);
  guest::ResetResource(g_cs_pipes);
  guest::ResetResource(g_prebuilt);
  guest::ResetResource(g_cs_stage);
  guest::ResetResource(g_cs_list);
  guest::ResetResource(g_cs_scratch_owner);
  guest::ResetResource(g_cs_scratch_bytes);
  guest::ResetResource(g_cs_frame_retired);
  guest::ResetResource(g_cs_free);
  guest::ResetResource(g_cs_free_bytes);
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
  guest::ResetResource(g_indirect_ring);
  guest::ResetResource(g_predicate);
  guest::ResetResource(g_indirect_head);
  guest::ResetResource(g_indirect_frame);
  guest::ResetResource(g_indirect_frame_n);
});

}  // namespace gpu::render::cs
