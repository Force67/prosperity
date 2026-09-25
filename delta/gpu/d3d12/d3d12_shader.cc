/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/d3d12/d3d12_shader.h"

#include <spirv_hlsl.hpp>

#include <algorithm>
#include <sstream>

namespace gpu::d3d12 {

namespace {

namespace sc = spirv_cross;

const char* StagePrefix(rhi::ShaderStage stage) {
  switch (stage) {
    case rhi::kStageVertex:
      return "vs";
    case rhi::kStageGeometry:
      return "gs";
    case rhi::kStageFragment:
      return "ps";
    case rhi::kStageCompute:
      return "cs";
    case rhi::kStageMesh:
      return "ms";
    default:
      return "??";
  }
}

bool Replace(std::string& text, const std::string& from,
             const std::string& to) {
  bool any = false;
  for (size_t at = text.find(from); at != std::string::npos;
       at = text.find(from, at + to.size())) {
    text.replace(at, from.size(), to);
    any = true;
  }
  return any;
}

// Retypes members of the vertex input struct whose TEXCOORD<n> semantic is in
// one of the masks. The generated main copies each member into the shader's
// float variable, and HLSL converts the integer value there, which is what a
// scaled format means.
void RetypeInputs(std::string& hlsl, const LowerOptions& o) {
  const u32 all = o.uint_inputs | o.sint_inputs | o.swap_rb_inputs;
  if (!all)
    return;
  const size_t begin = hlsl.find("struct SPIRV_Cross_Input\n{");
  if (begin == std::string::npos)
    return;
  const size_t end = hlsl.find("};", begin);
  std::istringstream in(hlsl.substr(begin, end - begin));
  std::string out, line;
  std::vector<std::string> swapped;
  while (std::getline(in, line)) {
    const size_t sem = line.find(": TEXCOORD");
    if (sem != std::string::npos) {
      const u32 loc = static_cast<u32>(std::atoi(line.c_str() + sem + 10));
      const size_t type = line.find_first_not_of(' ');
      if (loc < 32 && type != std::string::npos &&
          line.compare(type, 5, "float") == 0) {
        if (o.uint_inputs & (1u << loc))
          line.replace(type, 5, "uint");
        else if (o.sint_inputs & (1u << loc))
          line.replace(type, 5, "int");
        if (o.swap_rb_inputs & (1u << loc)) {
          const size_t name = line.find(' ', type) + 1;
          swapped.push_back(line.substr(name, line.find(' ', name) - name));
        }
      }
    }
    out += line + "\n";
  }
  hlsl.replace(begin, end - begin, out);
  for (const std::string& name : swapped)
    Replace(hlsl, "= stage_input." + name + ";",
            "= stage_input." + name + ".zyxw;");
}

}  // namespace

u64 HashWords(const u32* words, size_t count, u64 seed) {
  u64 h = 0xcbf29ce484222325ull ^ seed;
  for (size_t i = 0; i < count; i++) {
    h ^= words[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

bool LowerToHlsl(const u32* words,
                 size_t count,
                 const LowerOptions& options,
                 LoweredShader* out) {
  *out = {};
  try {
    sc::CompilerHLSL compiler(words, count);
    const spv::ExecutionModel model = compiler.get_execution_model();

    sc::CompilerGLSL::Options common = compiler.get_common_options();
    common.vertex.flip_vert_y =
        options.flip_y && model == spv::ExecutionModelVertex;
    common.vertex.fixup_clipspace = false;
    compiler.set_common_options(common);

    sc::CompilerHLSL::Options hlsl_options;
    hlsl_options.shader_model = options.shader_model;
    hlsl_options.point_size_compat = true;
    hlsl_options.point_coord_compat = true;
    hlsl_options.support_nonzero_base_vertex_base_instance =
        model == spv::ExecutionModelVertex;
    compiler.set_hlsl_options(hlsl_options);

    sc::HLSLResourceBinding push;
    push.stage = model;
    push.desc_set = sc::ResourceBindingPushConstantDescriptorSet;
    push.binding = sc::ResourceBindingPushConstantBinding;
    push.cbv = {kInternalSpace, kPushRegister};
    compiler.add_hlsl_resource_binding(push);
    compiler.set_hlsl_aux_buffer_binding(
        sc::HLSL_AUX_BINDING_BASE_VERTEX_INSTANCE, kDrawRegister,
        kInternalSpace);

    if (model == spv::ExecutionModelGLCompute) {
      const sc::VariableID id = compiler.remap_num_workgroups_builtin();
      if (id) {
        compiler.set_decoration(id, spv::DecorationDescriptorSet,
                                kInternalSpace);
        compiler.set_decoration(id, spv::DecorationBinding,
                                kDispatchRegister);
        out->uses_workgroup_count = true;
      }
    }

    const sc::ShaderResources resources = compiler.get_shader_resources();
    for (const sc::Resource& r : resources.storage_buffers) {
      const u32 set =
          compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
      const u32 binding = compiler.get_decoration(r.id, spv::DecorationBinding);
      const bool srv =
          std::find(options.read_only_storage.begin(),
                    options.read_only_storage.end(),
                    std::make_pair(set, binding)) !=
          options.read_only_storage.end();
      if (srv)
        compiler.set_decoration(r.id, spv::DecorationNonWritable);
      else
        compiler.set_hlsl_force_storage_buffer_as_uav(set, binding);
    }

    std::string hlsl = compiler.compile();
    out->uses_draw_params = compiler.is_hlsl_aux_buffer_binding_used(
        sc::HLSL_AUX_BINDING_BASE_VERTEX_INSTANCE);

    if (options.flip_y) {
      const std::string sign = "delta_y_sign";
      if (model == spv::ExecutionModelVertex)
        out->uses_raster = Replace(hlsl, "gl_Position.y = -gl_Position.y;",
                                   "gl_Position.y *= " + sign + ";");
      else if (model == spv::ExecutionModelGeometry)
        out->uses_raster =
            Replace(hlsl, "stage_output.gl_Position = gl_Position;",
                    "stage_output.gl_Position = float4(gl_Position.x, "
                    "gl_Position.y * " +
                        sign + ", gl_Position.zw);");
      if (out->uses_raster)
        hlsl = "cbuffer DeltaRaster : register(b" +
               std::to_string(kRasterRegister) + ", space" +
               std::to_string(kInternalSpace) + ")\n{\n    float " + sign +
               ";\n};\n\n" + hlsl;
    }
    if (model == spv::ExecutionModelVertex)
      RetypeInputs(hlsl, options);

    out->hlsl = std::move(hlsl);
    out->profile = std::string(StagePrefix(options.stage)) + "_" +
                   std::to_string(options.shader_model / 10) + "_" +
                   std::to_string(options.shader_model % 10);
    return true;
  } catch (const std::exception& e) {
    out->error = e.what();
    return false;
  }
}

}  // namespace gpu::d3d12
