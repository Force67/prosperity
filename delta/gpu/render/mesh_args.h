#pragma once

#include "gpu/rhi/device.h"

namespace gpu::render {

rhi::BindGroupLayout* MeshArgsLayout();
rhi::BindGroup* MeshArgsSet();
bool PrepareMeshArgs(rhi::Buffer* source,
                     u64 source_offset,
                     u32 inputs_per_group,
                     rhi::Buffer** tasks,
                     u64* tasks_offset,
                     u32* argument_word);
extern u64 g_mesh_indirect_n;

}  // namespace gpu::render
