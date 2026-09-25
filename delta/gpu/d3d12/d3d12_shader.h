#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * SPIR-V -> HLSL for the D3D12 backend, and HLSL -> DXIL through DXC.
 *
 * Resource registers follow the SPIR-V decorations: register = binding and
 * space = descriptor set, for every class (b/t/u/s). A combined image sampler
 * becomes a texture at t<binding> and a sampler at s<binding>. The backend's
 * own constants live in kInternalSpace; the root signature builder uses the
 * same numbers.
 */

#include <string>
#include <utility>
#include <vector>

#include "base/arch.h"
#include "gpu/rhi/types.h"

namespace gpu::d3d12 {

constexpr u32 kInternalSpace = 1000;
constexpr u32 kPushRegister = 0;      // push constants
constexpr u32 kRasterRegister = 1;    // float y_sign, for the viewport flip
constexpr u32 kDrawRegister = 2;      // int base_vertex, base_instance
constexpr u32 kDispatchRegister = 3;  // uint3 workgroup count
constexpr u32 kGroupBaseRegister = 4;  // uint3 first workgroup id

struct LowerOptions {
  rhi::ShaderStage stage = rhi::kStageVertex;
  u32 shader_model = 60;  // 60 = 6.0
  // The last stage before the rasteriser: gl_Position.y is multiplied by the
  // kRasterRegister constant.
  bool flip_y = false;
  // Vertex inputs fed from an integer format that the shader reads as float
  // (Uscaled/Sscaled attributes, which DXGI lacks): bit n = location n.
  u32 uint_inputs = 0;
  u32 sint_inputs = 0;
  // Vertex inputs whose red and blue arrive swapped (A2R10G10B10).
  u32 swap_rb_inputs = 0;
  // Compute: workgroup ids start at the kGroupBaseRegister constant.
  bool dispatch_base = false;
  // Storage buffers the layout binds as SRVs, as (set, binding).
  std::vector<std::pair<u32, u32>> read_only_storage;
  // The previous stage's outputs (LoweredShader::outputs). D3D12 links
  // stages by signature register, not by semantic, so the inputs are
  // rewritten to the producer's order and shapes.
  std::string producer_outputs;
};

struct LoweredShader {
  std::string hlsl;
  std::string outputs;  // the stage's output signature, for the next stage
  std::string profile;  // "vs_6_0", ...
  bool uses_draw_params = false;    // reads kDrawRegister
  bool uses_workgroup_count = false;  // reads kDispatchRegister
  bool uses_raster = false;           // reads kRasterRegister
  std::string error;
};

// False (with out->error) when SPIRV-Cross cannot express the module.
bool LowerToHlsl(const u32* words,
                 size_t count,
                 const LowerOptions& options,
                 LoweredShader* out);

// The loaded DXC library. Thread-safe; each thread gets its own compiler.
class Dxc {
 public:
  // False when no DXC library could be loaded.
  static bool Load();
  static bool Compile(const std::string& hlsl,
                      const std::string& profile,
                      std::vector<u8>* dxil,
                      std::string* error);
};

// SPIR-V hash for caches (FNV-1a, 64 bits).
u64 HashWords(const u32* words, size_t count, u64 seed = 0);

}  // namespace gpu::d3d12
