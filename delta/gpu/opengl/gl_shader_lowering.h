/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/rhi/types.h"

namespace gpu::opengl {

enum class SlotKind : u8 {
  kUniformBuffer,
  kStorageBuffer,
  kTexture,
  kImage,
  kPointer,
};
constexpr u32 kSlotKinds = 5;
constexpr char kPointerTable[] = "delta_ptr";
constexpr char kDispatchBase[] = "delta_base";

constexpr u32 kPushUboSlot = 0;

struct ResourceSlot {
  u32 set = 0;
  u32 binding = 0;
  SlotKind kind = SlotKind::kUniformBuffer;
  u32 slot = 0;
};

struct PushUniform {
  u32 stage = 0;
  base::String name;
  u32 vec4_count = 0;
  char type = 'u';
};

struct ProgramInterface {
  base::Vector<ResourceSlot> slots;
  base::Vector<PushUniform> push_uniforms;
  bool push_ubo = false;
  u32 pointer_count = 0;
  base::Vector<base::Pair<u32, u32>> vertex_locations;
};

struct GlslFeatures {
  bool nv_barycentric_only = false;
  u32 max_uniform_buffers = 84;
  u32 max_storage_buffers = 96;
  u32 max_textures = 192;
  u32 max_images = 8;
  u32 max_stage_storage_buffers = 16;
  u32 max_vertex_attribs = 16;
  bool buffer_pointers = false;
  bool khr_subgroup = false;
};

struct StageCode {
  u32 stage = 0;
  rhi::ShaderCode code;
  bool dispatch_base = false;
  bool strip_order = false;
};

inline constexpr size_t kMaxModuleBytes = 640 * 1024;

bool LowerProgram(const StageCode* stages,
                  u32 count,
                  const base::Vector<const rhi::BindGroupLayoutDesc*>& groups,
                  const GlslFeatures& features,
                  base::Vector<base::String>* glsl,
                  ProgramInterface* program,
                  base::String* error);

bool ReflectLayout(const rhi::ShaderCode& code,
                   u32* stage,
                   base::Vector<rhi::BindGroupLayoutDesc>* groups,
                   u32* push_constant_bytes,
                   base::String* error);

}  // namespace gpu::opengl
