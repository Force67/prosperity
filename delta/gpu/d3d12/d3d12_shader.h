#pragma once

#include "base/arch.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/rhi/types.h"

namespace gpu::d3d12 {

constexpr u32 kInternalSpace = 1000;
constexpr u32 kPushRegister = 0;
constexpr u32 kRasterRegister = 1;
constexpr u32 kDrawRegister = 2;
constexpr u32 kDispatchRegister = 3;
constexpr u32 kGroupBaseRegister = 4;

struct LowerOptions {
  rhi::ShaderStage stage = rhi::kStageVertex;
  u32 shader_model = 60;
  bool flip_y = false;
  u32 uint_inputs = 0;
  u32 sint_inputs = 0;
  u32 swap_rb_inputs = 0;
  bool dispatch_base = false;
  bool emulate_barycentrics = false;
  base::Vector<base::Pair<u32, u32>> read_only_storage;
  base::String producer_outputs;
};

struct LoweredShader {
  base::String hlsl;
  base::String outputs;
  base::String profile;
  bool uses_draw_params = false;
  bool uses_workgroup_count = false;
  bool uses_raster = false;
  base::String error;
};

bool LowerToHlsl(const u32* words,
                 size_t count,
                 const LowerOptions& options,
                 LoweredShader* out);

class Dxc {
 public:
  static bool Load();
  static bool Compile(const base::String& hlsl,
                      const base::String& profile,
                      base::Vector<u8>* dxil,
                      base::String* error);
};

base::String BarycentricGeometryShader(const base::String& vertex_outputs,
                                       base::String* outputs);

base::Vector<u32> PatchSpirvForHlsl(const u32* words, size_t count);
bool DeclaresBuiltIn(const u32* words, size_t count, u32 builtin);

u64 HashWords(const u32* words, size_t count, u64 seed = 0);

}  // namespace gpu::d3d12
