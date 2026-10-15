#include "gpu/render/draw/mesh_args.h"

#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/render_target.h"
#include "guest/session.h"

namespace gpu::render {
namespace {
constexpr u64 kSlotBytes = 1ull << 20;
constexpr u64 kEntryBytes = 32;
struct MeshArgs {
  rhi::BindGroupLayout* draw_layout = nullptr;
  rhi::BindGroup* draw_set = nullptr;
  rhi::Buffer* ring = nullptr;
  rhi::BindGroupLayout* compute_set = nullptr;
  rhi::PipelineLayout* compute_layout = nullptr;
  rhi::Pipeline* compute = nullptr;
  int frame = -1;
  u64 head = kEntryBytes;
} g_mesh;
const guest::SessionReset reset_mesh([] { g_mesh = {}; });
}  // namespace

u64 g_mesh_indirect_n = 0;

rhi::BindGroupLayout* MeshArgsLayout() {
  if (!g_mesh.draw_layout) {
    rhi::BindGroupLayoutDesc desc;
    desc.bindings.push_back(
        {0, rhi::BindingType::kStorageBuffer, rhi::kStageMesh});
    desc.bindings[0].read_only = true;
    g_mesh.draw_layout = Device().CreateBindGroupLayout(desc);
  }
  return g_mesh.draw_layout;
}

rhi::BindGroup* MeshArgsSet() {
  return g_mesh.draw_set;
}

bool PrepareMeshArgs(rhi::Buffer* source,
                     u64 source_offset,
                     u32 inputs_per_group,
                     rhi::Buffer** tasks,
                     u64* tasks_offset,
                     u32* argument_word) {
  auto* list = g_frame.list;
  if (!g_mesh.ring) {
    EndRegion();
    rhi::BufferDesc desc;
    desc.size = kSlotBytes * 2;
    desc.usage =
        rhi::kBufferStorage | rhi::kBufferIndirect | rhi::kBufferCopyDst;
    desc.name = "mesh arguments";
    g_mesh.ring = Device().CreateBuffer(desc);
    if (!g_mesh.ring || !MeshArgsLayout())
      return false;
    list->FillBuffer(g_mesh.ring, 0, desc.size, 0);
    list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessShaderRead);
    rhi::BindGroupDesc set;
    set.layout = g_mesh.draw_layout;
    rhi::BindingWrite w;
    w.buffer = g_mesh.ring;
    w.range = desc.size;
    set.writes.push_back(w);
    g_mesh.draw_set = Device().CreateBindGroup(set);
  }
  if (!g_mesh.draw_set)
    return false;
  if (!source)
    return true;
  if (!inputs_per_group || (source_offset & 3))
    return false;
  EndRegion();
  if (!g_mesh.compute) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
    const auto code = gcn::BuildMeshIndirectShader();
#else
    const base::Vector<u32> code;
#endif
    rhi::BindGroupLayoutDesc desc;
    desc.bindings = {{0, rhi::BindingType::kStorageBuffer, rhi::kStageCompute},
                     {1, rhi::BindingType::kStorageBuffer, rhi::kStageCompute}};
    desc.bindings[0].read_only = true;
    desc.push = true;
    g_mesh.compute_set = Device().CreateBindGroupLayout(desc);
    rhi::PipelineLayoutDesc layout;
    layout.groups = {g_mesh.compute_set};
    layout.push_constant_bytes = 24;
    layout.push_constant_stages = rhi::kStageCompute;
    g_mesh.compute_layout =
        g_mesh.compute_set ? Device().CreatePipelineLayout(layout) : nullptr;
    rhi::ComputePipelineDesc pipeline;
    pipeline.layout = g_mesh.compute_layout;
    pipeline.code = rhi::Code(code);
    g_mesh.compute = g_mesh.compute_layout && !code.empty()
                         ? Device().CreateComputePipeline(pipeline)
                         : nullptr;
  }
  if (!g_mesh.compute)
    return false;
  if (g_mesh.frame != int(g_frame.num)) {
    g_mesh.frame = g_frame.num;
    g_mesh.head = kEntryBytes;
  }
  if (g_mesh.head + kEntryBytes > kSlotBytes)
    return false;
  const u64 offset = g_frame.slot_idx * kSlotBytes + g_mesh.head;
  g_mesh.head += kEntryBytes;
  const u64 alignment = base::Max(Device().caps().storage_offset_alignment, 1u);
  const u64 window = source_offset & ~(alignment - 1);
  rhi::BindingWrite writes[2];
  writes[0].buffer = source;
  writes[0].offset = window;
  writes[0].range = source_offset - window + 16;
  writes[1].binding = 1;
  writes[1].buffer = g_mesh.ring;
  writes[1].range = g_mesh.ring->desc().size;
  const auto& limits = Device().caps().mesh;
  const u32 params[] = {u32((source_offset - window) / 4),
                        u32(offset / 4),
                        inputs_per_group,
                        limits.max_groups[0],
                        limits.max_groups[1],
                        limits.max_total_groups};
  list->Barrier(rhi::kAccessAllWrite,
                rhi::kAccessComputeRead | rhi::kAccessComputeWrite);
  list->SetPipeline(g_mesh.compute);
  list->PushBindGroup(0, g_mesh.compute_set, writes, 2);
  list->SetPushConstants(0, sizeof(params), params);
  list->Dispatch(1, 1, 1);
  list->Barrier(rhi::kAccessComputeWrite,
                rhi::kAccessShaderRead | rhi::kAccessIndirectRead);
  *tasks = g_mesh.ring;
  *tasks_offset = offset + 16;
  *argument_word = offset / 4;
  g_mesh_indirect_n++;
  return true;
}

}  // namespace gpu::render
