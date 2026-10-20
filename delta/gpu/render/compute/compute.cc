/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute/internal.h"

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

u64 g_predicate_offset = 0;
base::Atomic<u64> g_mirror_resolves{0};
DELTA_OPTION(bool, kMirror, "DELTA_GPU_MIRROR", true);
DELTA_OPTION(bool, kCsListProf, "DELTA_GPU_LISTPROF", false);

static_assert(ComputeInfo::kMaxResources == gcn::kMaxCsResources);

CsStats g_cs_stats;
ComputeState g_cs;

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
  auto it = g_cs.pipes.find(key);
  if (it != g_cs.pipes.end())
    return it->second.num_res == ci.num_res ? &it->second : nullptr;
  CsPipe cp;
  cp.num_res = ci.num_res;
  cp.gds_binding = ci.gds_binding;
  if (ci.gds_binding < 0) {
    base::SharedPointer<PrebuiltSlot> slot;
    {
      base::LockGuard<base::Mutex> lock(g_prebuilt_mutex);
      const auto found = g_cs.prebuilt.find(PrebuiltKey(
          ci.recomp->spirv, ci.num_res, ci.recomp->guest_memory_binding));
      if (found != g_cs.prebuilt.end()) {
        slot = found->second;
        g_cs.prebuilt.erase(found);
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
      g_stats.ns_pipe_build += NowNs() - started;
      g_stats.pipe_build_n++;
      g_cs.pipes[key] = cp;
      return &g_cs.pipes[key];
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
  g_stats.ns_pipe_build += NowNs() - started;
  if (CsSyncReporting())
    BASE_LOGI("cspipe", "compiled cs={:#x} ms={:.3f}", ci.cs_addr,
              double(NowNs() - started) / 1e6);

  g_stats.pipe_build_n++;
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
  g_cs.pipes[key] = cp;
  return &g_cs.pipes[key];
}




bool CsFrameReferenced(const CsRange& e) {
  return e.frame_ref >= 0 && e.frame_ref + 2 > g_frame.num;
}


bool EnsureGdsBuffer() {
  if (g_cs.gds.buf)
    return true;
  g_cs.gds.buf = CreateCsBuffer(
      GdsBuffer::kBytes,
      rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc,
      rhi::MemoryKind::kReadback);
  if (!g_cs.gds.buf)
    return false;
  g_cs.gds.map = g_cs.gds.buf->mapped();
  if (!g_cs.gds.map) {
    Device().Destroy(g_cs.gds.buf);
    g_cs.gds = {};
    return false;
  }
  std::memset(g_cs.gds.map, 0, GdsBuffer::kBytes);
  return true;
}

void CsRangeDestroy(CsRange& e);

base::HashMap<u64, CsRange>::iterator EraseCsRange(
    base::HashMap<u64, CsRange>::iterator it) {
  if (it == g_cs.evict_next)
    ++g_cs.evict_next;
  UnindexCsRange(it->first, it->second.guest_bytes);
  CsRangeDestroy(it->second);
  return g_cs.ranges.erase(it);
}

bool CsReleaseMemory(const CsRange& keep) {
  const auto drop_free_list = [] {
    const bool any = !g_cs.free_list.empty();
    for (const CsFreeBuffer& f : g_cs.free_list)
      Device().Destroy(f.buf);
    g_cs.free_list.clear();
    g_cs.free_bytes = 0;
    return any;
  };
  const bool released = drop_free_list();
  u64 evicted = 0;
  for (auto it = g_cs.ranges.begin(); it != g_cs.ranges.end();) {
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
    g_cs.frame_retired.push_back(e.buf);
    e.frame_ref = -1;
    e.pending_batch = false;
  } else {
    Device().Destroy(e.buf);
  }
  e.buf = nullptr;
  e.device_local = false;
  e.imported = false;
  if (!e.imported)
    g_cs.range_bytes -= e.cap;
  e.cap = 0;
  u64 cap = (size + 0xFFFF) & ~u64(0xFFFF);
  for (size_t i = g_cs.free_list.size(); i-- > 0;) {
    const CsFreeBuffer& f = g_cs.free_list[i];
    if (f.cap != cap || f.device_local != static_cast<bool>(kCsVram))
      continue;
    e.buf = f.buf;
    e.device_local = f.device_local;
    e.cap = cap;
    g_cs.free_bytes -= cap;
    g_cs.free_list.erase(g_cs.free_list.begin() + i);
    g_cs.range_bytes += cap;
    g_cs_stats.recycled_n++;
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
  g_cs.range_bytes += cap;
  return true;
}

bool CsRangeRename(CsRange& e, u64 size) {
  if (e.buf)
    g_cs.frame_retired.push_back(e.buf);
  e.shadow_stamp = 0;
  e.frame_ref = -1;
  e.pending_batch = false;
  g_cs.range_bytes -= e.cap;
  e.buf = nullptr;
  e.device_local = false;
  e.cap = 0;
  return CsRangeEnsureBuffer(e, size);
}

void CsRangeDestroy(CsRange& e) {
  base::EraseIf(g_cs.pending, [&e](CsRange* r) { return r == &e; });
  if (e.mirror)
    g_cs.frame_retired.push_back(e.mirror);
  if (e.shadow)
    g_cs.frame_retired.push_back(e.shadow);
  if (!e.imported && !CsFrameReferenced(e) && !e.pending_batch && e.buf &&
      e.cap && g_cs.free_bytes + e.cap <= (256ull << 20)) {
    g_cs.free_list.push_back({e.buf, e.cap, e.device_local});
    g_cs.free_bytes += e.cap;
    g_cs.range_bytes -= e.cap;
    e = CsRange{};
    return;
  }
  if (CsFrameReferenced(e) || e.pending_batch)
    g_cs.frame_retired.push_back(e.buf);
  else
    Device().Destroy(e.buf);
  g_cs.range_bytes -= e.cap;
  e = CsRange{};
}

bool CsEnsureStage(u32 i, u64 size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs.stage[i];
  if (s.buf && s.cap >= size)
    return true;
  s.map = nullptr;
  if (s.buf)
    g_cs.frame_retired.push_back(s.buf);
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
  aged = base::move(g_cs.frame_retired);
  g_cs.frame_retired.clear();
}

}  // namespace gpu::render

namespace gpu::render::cs {

const guest::SessionReset g_session_reset([] {
  host_memory::SetWriteFaultHandler(nullptr);
  host_memory::SetHostWriteHook(nullptr);
  if (g_cs.own)
    munmap(g_cs.own, kOwnBlocks * sizeof(OwnBlock));
  guest::ResetResource(g_cs);
  guest::ResetResource(g_cs_stats);
});

}  // namespace gpu::render::cs
