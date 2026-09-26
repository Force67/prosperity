/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// SPIR-V -> GLSL 4.60 for the OpenGL backend. Pure CPU work: no GL calls, so
// it runs on whichever thread creates the pipeline.
//
// GL has one flat namespace per resource kind instead of Vulkan's
// (set, binding) pairs. Each program gets its own slot assignment covering
// only the resources its stages use, so a wide pipeline layout does not run
// into GL's small per-kind limits.

#include <string>
#include <utility>
#include <vector>

#include "base/arch.h"
#include "gpu/rhi/types.h"

namespace gpu::opengl {

// kPointer: a storage buffer past GL's per-stage limit, read through a GPU
// address (GL_NV_shader_buffer_load); its slot indexes the `delta_ptr`
// uniform, one uvec4 {address lo, address hi, bytes, 0} per buffer.
enum class SlotKind : u8 {
  kUniformBuffer,
  kStorageBuffer,
  kTexture,
  kImage,
  kPointer,
};
constexpr u32 kSlotKinds = 5;
constexpr char kPointerTable[] = "delta_ptr";
// The workgroup base of CommandList::DispatchBase, a uvec3 uniform added to
// the workgroup and global invocation ids.
constexpr char kDispatchBase[] = "delta_base";

// Push constants that cannot be flattened into a uniform array live in this
// uniform buffer binding; the groups' uniform buffers start after it.
constexpr u32 kPushUboSlot = 0;

struct ResourceSlot {
  u32 set = 0;
  u32 binding = 0;
  SlotKind kind = SlotKind::kUniformBuffer;
  u32 slot = 0;
};

// One stage's push constants as a flattened `uniform {u,i,}vec4 name[count]`.
struct PushUniform {
  u32 stage = 0;  // rhi::ShaderStage bit
  std::string name;
  u32 vec4_count = 0;
  char type = 'u';  // 'f', 'i' or 'u'
};

struct ProgramInterface {
  std::vector<ResourceSlot> slots;  // sorted by (set, binding)
  std::vector<PushUniform> push_uniforms;
  bool push_ubo = false;  // some stage reads push constants from kPushUboSlot
  u32 pointer_count = 0;  // entries of kPointerTable
  // Vertex inputs are packed from GL attribute 0: {shader location, GL
  // attribute} for each location the vertex stage reads.
  std::vector<std::pair<u32, u32>> vertex_locations;
};

struct GlslFeatures {
  // The context has GL_NV_fragment_shader_barycentric but not the EXT one.
  bool nv_barycentric_only = false;
  u32 max_uniform_buffers = 84;
  u32 max_storage_buffers = 96;
  u32 max_textures = 192;
  u32 max_images = 8;
  u32 max_stage_storage_buffers = 16;
  u32 max_vertex_attribs = 16;
  // GL_NV_shader_buffer_load and GL_NV_gpu_shader5: storage buffers past the
  // per-stage limit become pointers.
  bool buffer_pointers = false;
  bool khr_subgroup = false;  // GL_KHR_shader_subgroup
};

struct StageCode {
  u32 stage = 0;  // rhi::ShaderStage bit
  rhi::ShaderCode code;
  bool dispatch_base = false;  // compute: offset by kDispatchBase
  bool strip_order = false;    // fragment: drawn as a triangle strip
};

// Lowers the stages of one program against `groups` (indexed by set; null
// entries are empty sets). On success `glsl` holds one source per stage, in
// order. A resource missing from the layout, or bound as another type, fails.
bool LowerProgram(const StageCode* stages,
                  u32 count,
                  const std::vector<const rhi::BindGroupLayoutDesc*>& groups,
                  const GlslFeatures& features,
                  std::vector<std::string>* glsl,
                  ProgramInterface* program,
                  std::string* error);

// The resources a module uses, as a layout would declare them, for tools
// that have a module but no pipeline layout. Sets are dense from 0.
bool ReflectLayout(const rhi::ShaderCode& code,
                   u32* stage,
                   std::vector<rhi::BindGroupLayoutDesc>* groups,
                   u32* push_constant_bytes,
                   std::string* error);

}  // namespace gpu::opengl
