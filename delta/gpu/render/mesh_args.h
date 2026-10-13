#pragma once

#include "gpu/rhi/device.h"

namespace gpu::render {

rhi::BindGroupLayout* MeshArgsLayout();
rhi::BindGroup* MeshArgsSet();
// Snapshot draw metadata and derive task counts on the GPU before the pass.
// A null source prepares the shared binding for a direct mesh draw.
bool PrepareMeshArgs(rhi::Buffer* source,
                     u64 source_offset,
                     u32 inputs_per_group,
                     rhi::Buffer** tasks,
                     u64* tasks_offset,
                     u32* argument_word);
extern u64 g_mesh_indirect_n;

}  // namespace gpu::render
