/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/draw/index_convert.h"

#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/render/compute/internal.h"
#include "gpu/render/frame.h"
#include "gpu/render/draw/index_upload.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/memory/upload_ring.h"

namespace gpu::render {

bool ConvertGuestIndices(Renderer& renderer,
                         u64 base,
                         u32 count,
                         u32 index_type,
                         bool quads,
                         u64 offset,
                         rhi::Buffer** indirect_buffer,
                         u64* indirect_offset) {
  const bool indirect = indirect_buffer && *indirect_buffer;
  const u32 output_count = quads ? count / 4 * 6 : count;
  const bool strided =
      indirect && output_count > gcn::kIndexConversionGroups * 64;
  const auto mode = strided    ? gcn::IndexConversionMode::kIndirectStrided
                    : indirect ? gcn::IndexConversionMode::kIndirect
                               : gcn::IndexConversionMode::kDirect;
  struct Pipe {
    rhi::BindGroupLayout* set = nullptr;
    rhi::PipelineLayout* layout = nullptr;
    rhi::Pipeline* pipe = nullptr;
    bool tried = false;
  };
  static Pipe pipes[3][2][3];
  if (index_type > 2 || (indirect && (!quads || !indirect_offset)))
    return false;
  Pipe& p = pipes[static_cast<u32>(mode)][quads][index_type];
  if (!p.tried) {
    p.tried = true;
#ifdef DELTA_HAVE_SPIRV_BACKEND
    const auto code = gcn::BuildIndexConversionShader(index_type, quads, mode);
#else
    const base::Vector<u32> code;
#endif
    rhi::BindGroupLayoutDesc set;
    for (u32 i = 0; i < (indirect ? 3u : 2u); i++)
      set.bindings.push_back(
          {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
    set.bindings[0].read_only = true;
    if (indirect)
      set.bindings[2].read_only = true;
    set.push = true;
    p.set = code.empty() ? nullptr : Device().CreateBindGroupLayout(set);
    rhi::PipelineLayoutDesc layout;
    layout.groups = {p.set};
    layout.push_constant_bytes = indirect ? 20 : 12;
    layout.push_constant_stages = rhi::kStageCompute;
    p.layout = p.set ? Device().CreatePipelineLayout(layout) : nullptr;
    rhi::ComputePipelineDesc pipe;
    pipe.layout = p.layout;
    pipe.code = rhi::Code(code);
    p.pipe = p.layout ? Device().CreateComputePipeline(pipe) : nullptr;
  }
  if (!p.pipe)
    return false;
  if (!g_ring.converted_ib) {
    rhi::BufferDesc desc;
    desc.size = kIbRing;
    desc.usage = rhi::kBufferStorage | rhi::kBufferIndex |
                 rhi::kBufferIndirect | rhi::kBufferCopySrc;
    g_ring.converted_ib = Device().CreateBuffer(desc);
  }
  if (!g_ring.converted_ib)
    return false;
  if (!output_count && !indirect)
    return true;
  const u64 bytes = u64(count) * GuestIndexElementBytes(index_type);
  const u64 padded = (bytes + 3) & ~u64(3);
  rhi::Buffer* raw = nullptr;
  u64 source_offset = 0;
  const u64 source_base = base & ~u64(3);
  const u64 source_bytes = (bytes + (base - source_base) + 3) & ~u64(3);
  const bool resident = !(base & (GuestIndexElementBytes(index_type) - 1)) &&
                        CsResidentGeometryBuffer(source_base, source_bytes,
                                                 &raw, &source_offset) &&
                        !(source_offset & 3);
  EndRegion();
  if (resident) {
    source_offset += base - source_base;
  } else {
    source_offset = 0;
    if (!g_ring.index_convert_src ||
        g_ring.index_convert_src->desc().size < padded) {
      rhi::BufferDesc desc;
      desc.size = (padded + 0x3fffff) & ~u64(0x3fffff);
      desc.usage = rhi::kBufferStorage | rhi::kBufferCopyDst;
      auto* replacement = Device().CreateBuffer(desc);
      if (!replacement)
        return false;
      if (g_ring.index_convert_src)
        g_cs_frame_retired.push_back(g_ring.index_convert_src);
      g_ring.index_convert_src = replacement;
    }
    raw = g_ring.index_convert_src;
    g_frame.list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
    if (!CsSupplyBuffer(base, bytes, raw, 0)) {
      if (!FlushCsWritesForGpu(renderer, base, bytes, "index") ||
          !RecordGuestRead(g_frame.list, base, bytes, raw))
        return false;
    }
    g_stats.index_convert_copy_bytes += bytes;
  }
  auto* list = g_frame.list;
  list->Barrier(rhi::kAccessAllWrite, kAccessComputeRW);
  list->ProfileBegin(15);
  list->SetPipeline(p.pipe);
  const u64 alignment = base::Max(Device().caps().storage_offset_alignment, 1u);
  const u64 window = offset & ~(alignment - 1);
  const u64 source_window = source_offset & ~(alignment - 1);
  const u64 source_byte = source_offset - source_window;
  rhi::BindingWrite writes[3];
  writes[0].buffer = raw;
  writes[0].offset = source_window;
  writes[0].range = (source_byte + bytes + 3) & ~u64(3);
  writes[1].binding = 1;
  writes[1].buffer = g_ring.converted_ib;
  writes[1].offset = window;
  writes[1].range =
      offset + u64(output_count) * 4 - window + (indirect ? 20 : 0);
  u64 argument_word = 0;
  if (indirect) {
    const u64 argument_window = *indirect_offset & ~(alignment - 1);
    argument_word = (*indirect_offset - argument_window) / 4;
    writes[2].binding = 2;
    writes[2].buffer = *indirect_buffer;
    writes[2].offset = argument_window;
    writes[2].range = *indirect_offset - argument_window + 20;
  }
  list->PushBindGroup(0, p.set, writes, indirect ? 3 : 2);
  const u32 params[] = {
      indirect ? count : output_count, static_cast<u32>((offset - window) / 4),
      static_cast<u32>(source_byte), static_cast<u32>(argument_word),
      static_cast<u32>((offset - window) / 4 + output_count)};
  list->SetPushConstants(0, indirect ? sizeof(params) : 12, params);
  list->Dispatch(strided ? gcn::kIndexConversionGroups
                         : base::Max(1u, (output_count + 63) / 64),
                 1, 1);
  list->ProfileEnd();
  g_stats.index_convert_n++;
  g_stats.index_convert_bytes += u64(output_count) * 4;
  list->Barrier(rhi::kAccessComputeWrite,
                rhi::kAccessIndexRead | rhi::kAccessIndirectRead);
  if (indirect) {
    *indirect_buffer = g_ring.converted_ib;
    *indirect_offset = offset + u64(output_count) * 4;
  }
  return true;
}

}  // namespace gpu::render
