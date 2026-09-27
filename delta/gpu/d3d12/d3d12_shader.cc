/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/d3d12/d3d12_shader.h"

#include <spirv_hlsl.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>

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

// One member of a generated stage input/output struct.
struct Element {
  std::string qualifiers, type, name, semantic;
};

bool FindStruct(const std::string& hlsl,
                const char* name,
                size_t* begin,
                size_t* end) {
  const std::string head = std::string("struct ") + name + "\n{\n";
  *begin = hlsl.find(head);
  if (*begin == std::string::npos)
    return false;
  *begin += head.size();
  *end = hlsl.find("};", *begin);
  return *end != std::string::npos;
}

std::vector<Element> ParseElements(const std::string& body) {
  std::vector<Element> out;
  std::istringstream in(body);
  std::string line;
  while (std::getline(in, line)) {
    const size_t colon = line.find(" : ");
    const size_t semi = line.rfind(';');
    if (colon == std::string::npos || semi == std::string::npos)
      continue;
    Element e;
    e.semantic = line.substr(colon + 3, semi - colon - 3);
    std::istringstream decl(line.substr(0, colon));
    std::vector<std::string> tokens;
    for (std::string t; decl >> t;)
      tokens.push_back(t);
    if (tokens.size() < 2)
      continue;
    e.name = tokens.back();
    e.type = tokens[tokens.size() - 2];
    for (size_t i = 0; i + 2 < tokens.size(); i++)
      e.qualifiers += tokens[i] + " ";
    out.push_back(e);
  }
  return out;
}

// Component count of a scalar or vector type name; 0 for anything else.
u32 Components(const std::string& type) {
  if (type.empty() || type.find('x') != std::string::npos)
    return 0;
  const char last = type.back();
  return last >= '1' && last <= '4' ? static_cast<u32>(last - '0') : 1;
}

std::string BaseType(const std::string& type) {
  return std::isdigit(static_cast<unsigned char>(type.back()))
             ? type.substr(0, type.size() - 1)
             : type;
}

// Rewrites the input struct to the producer's element order and widths:
// D3D12 matches stage signatures by packed register, so an input the
// consumer skips must still take its row.
void MatchInputs(std::string& hlsl, const std::string& producer) {
  size_t begin, end;
  if (producer.empty() || !FindStruct(hlsl, "SPIRV_Cross_Input", &begin, &end))
    return;
  const std::vector<Element> consumer =
      ParseElements(hlsl.substr(begin, end - begin));
  std::vector<bool> used(consumer.size());
  std::string out;
  u32 unused = 0;
  for (const Element& p : ParseElements(producer)) {
    size_t i = 0;
    while (i < consumer.size() && consumer[i].semantic != p.semantic)
      i++;
    if (i == consumer.size()) {
      out += "    " + p.type + " delta_unused" + std::to_string(unused++) +
             " : " + p.semantic + ";\n";
      continue;
    }
    used[i] = true;
    Element c = consumer[i];
    const u32 pn = Components(p.type), cn = Components(c.type);
    if (pn > cn && cn)
      c.type = BaseType(c.type) + std::to_string(pn);
    out += "    " + c.qualifiers + c.type + " " + c.name + " : " + c.semantic +
           ";\n";
  }
  for (size_t i = 0; i < consumer.size(); i++)
    if (!used[i])
      out += "    " + consumer[i].qualifiers + consumer[i].type + " " +
             consumer[i].name + " : " + consumer[i].semantic + ";\n";
  hlsl.replace(begin, end - begin, out);
}

// GLSL's extended integer arithmetic, which SPIRV-Cross emits by name. Bit
// arithmetic only: DXC folds the obvious 64-bit and compare forms into LLVM
// overflow intrinsics that DXIL validation rejects.
std::string ExtendedArithmetic(const std::string& hlsl) {
  std::string out;
  for (const char* n : {"", "2", "3", "4"}) {
    const std::string u = std::string("uint") + n, i = std::string("int") + n;
    if (hlsl.find("uaddCarry(") != std::string::npos)
      out += u + " uaddCarry(" + u + " x, " + u + " y, out " + u +
             " carry) { " + u +
             " s = x + y; carry = ((x & y) | ((x | y) & ~s)) >> 31; "
             "return s; }\n";
    if (hlsl.find("usubBorrow(") != std::string::npos)
      out += u + " usubBorrow(" + u + " x, " + u + " y, out " + u +
             " borrow) { " + u +
             " d = x - y; borrow = ((~x & y) | (~(x ^ y) & d)) >> 31; "
             "return d; }\n";
    const std::string mul =
        "{ " + u + " xl = x & 0xffffu, xh = x >> 16, yl = y & 0xffffu, "
        "yh = y >> 16; " + u + " lh = xl * yh, hl = xh * yl; " + u +
        " mid = ((xl * yl) >> 16) + (lh & 0xffffu) + (hl & 0xffffu); "
        "msb = xh * yh + (lh >> 16) + (hl >> 16) + (mid >> 16); "
        "lsb = x * y; }\n";
    if (hlsl.find("umulExtended(") != std::string::npos ||
        hlsl.find("imulExtended(") != std::string::npos)
      out += "void umulExtended(" + u + " x, " + u + " y, out " + u +
             " msb, out " + u + " lsb) " + mul;
    if (hlsl.find("imulExtended(") != std::string::npos)
      out += "void imulExtended(" + i + " x, " + i + " y, out " + i +
             " msb, out " + i + " lsb) { " + u + " hi, lo; umulExtended(" +
             u + "(x), " + u + "(y), hi, lo); msb = " + i + "(hi - (x < 0 ? " +
             u + "(y) : 0u) - (y < 0 ? " + u + "(x) : 0u)); lsb = " + i +
             "(lo); }\n";
  }
  return out;
}
// SPIRV-Cross cannot express this block with cbuffer packing rules: returns
// the variable's id from its message.
bool PackingFailure(const std::string& message, u32* id) {
  const std::string key = "cbuffer ID ";
  const size_t at = message.find(key);
  if (at == std::string::npos ||
      message.find("cannot be expressed") == std::string::npos)
    return false;
  *id = static_cast<u32>(std::strtoul(message.c_str() + at + key.size(),
                                      nullptr, 10));
  return true;
}

// Integer textures cannot be sampled before SM 6.7, and sampling one is
// always nearest: a texel fetch with the sampler's addressing. 2D textures
// and arrays gather the 2x2 footprint at level 0 and pick the texel the
// coordinate falls in; other levels, and 3D textures, load the clamped texel.
void IntegerSampling(std::string& hlsl) {
  static const std::regex decl(
      R"((Texture(?:2D|2DArray|3D))<((?:u)?int)4> (\w+) : register)");
  std::string helpers;
  std::vector<std::string> done;
  std::vector<std::string> names;
  for (auto it = std::sregex_iterator(hlsl.begin(), hlsl.end(), decl);
       it != std::sregex_iterator(); ++it) {
    const std::string tex = (*it)[1], base = (*it)[2];
    names.push_back((*it)[3]);
    const std::string key = tex + base;
    if (std::find(done.begin(), done.end(), key) != done.end())
      continue;
    done.push_back(key);
    const std::string t = base + "4";
    const std::string sig = t + " spvDeltaSampleInt(" + tex + "<" + t +
                            "> t, SamplerState s, ";
    if (tex == "Texture3D") {
      const std::string load =
          "{ uint w, h, d, n; t.GetDimensions(uint(lod), w, h, d, n); "
          "int3 p = clamp(int3(floor(c * float3(w, h, d))), 0, "
          "int3(w, h, d) - 1); return t.Load(int4(p, int(lod))); }\n";
      helpers += sig + "float3 c, float lod) " + load;
      helpers += sig + "float3 c) { return spvDeltaSampleInt(t, s, c, 0.0); }\n";
      helpers += sig +
                 "float3 c, float3 dx, float3 dy) { return "
                 "spvDeltaSampleInt(t, s, c, 0.0); }\n";
      continue;
    }
    const bool array = tex == "Texture2DArray";
    const std::string coord = array ? "float3" : "float2";
    const std::string dims = array ? "uint w, h, e, n; "
                                     "t.GetDimensions(uint(lod), w, h, e, n);"
                                   : "uint w, h, n; "
                                     "t.GetDimensions(uint(lod), w, h, n);";
    const std::string texel =
        array ? "int4(clamp(int2(floor(c.xy * float2(w, h))), 0, "
                "int2(w, h) - 1), int(c.z + 0.5), int(lod))"
              : "int3(clamp(int2(floor(c * float2(w, h))), 0, "
                "int2(w, h) - 1), int(lod))";
    helpers += sig + coord + " c, float lod) { " + dims +
               " if (lod >= 0.5) return t.Load(" + texel +
               "); float2 f = frac(c.xy * float2(w, h) - 0.5); "
               "uint i = f.y >= 0.5 ? (f.x >= 0.5 ? 1 : 0) : "
               "(f.x >= 0.5 ? 2 : 3); " +
               t + " r = t.GatherRed(s, c), g = t.GatherGreen(s, c), "
               "b = t.GatherBlue(s, c), a = t.GatherAlpha(s, c); "
               "return " + t + "(r[i], g[i], b[i], a[i]); }\n";
    helpers += sig + coord +
               " c) { return spvDeltaSampleInt(t, s, c, 0.0); }\n";
    helpers += sig + coord +
               " c, float2 dx, float2 dy) { return "
               "spvDeltaSampleInt(t, s, c, 0.0); }\n";
  }
  for (const std::string& name : names) {
    const std::regex use("\\b" + name + R"(\.Sample(?:Level|Bias|Grad)?\()");
    hlsl = std::regex_replace(hlsl, use, "spvDeltaSampleInt(" + name + ", ");
  }
  hlsl = helpers + "\n" + hlsl;
}

// Constructs above the target shader model that are lowered with the newer
// model's syntax and then rewritten.
struct Emulation {
  bool integer_sampling = false;  // SM 6.7
  bool barycentrics = false;      // SM 6.1
};

// SV_Barycentrics inputs become the varyings BarycentricGeometryShader adds.
void EmulateBarycentrics(std::string& hlsl) {
  if (hlsl.find("GetAttributeAtVertex") != std::string::npos)
    throw std::runtime_error("per-vertex fragment inputs need SM 6.1");
  static const std::regex persp(R"(: SV_Barycentrics[01]?;)");
  std::string out;
  std::istringstream in(hlsl);
  for (std::string line; std::getline(in, line);) {
    if (line.find("SV_Barycentrics") != std::string::npos)
      line = std::regex_replace(
          line, persp,
          line.find("noperspective") != std::string::npos ? ": DELTABARYNP;"
                                                          : ": DELTABARY;");
    out += line + "\n";
  }
  hlsl = std::move(out);
}

// One lowering pass. Blocks in `flatten` become arrays of 16-byte rows,
// declared inside a cbuffer at the block's register.
std::string Lower(const u32* words,
                  size_t count,
                  const LowerOptions& options,
                  const std::vector<u32>& flatten,
                  const Emulation& emulate,
                  LoweredShader* out) {
  sc::CompilerHLSL compiler(words, count);
  const spv::ExecutionModel model = compiler.get_execution_model();

  sc::CompilerGLSL::Options common = compiler.get_common_options();
  common.vertex.flip_vert_y =
      options.flip_y && model == spv::ExecutionModelVertex;
  common.vertex.fixup_clipspace = false;
  compiler.set_common_options(common);

  sc::CompilerHLSL::Options hlsl_options;
  hlsl_options.shader_model = options.shader_model;
  if (emulate.integer_sampling)
    hlsl_options.shader_model = std::max(hlsl_options.shader_model, 67u);
  if (emulate.barycentrics)
    hlsl_options.shader_model = std::max(hlsl_options.shader_model, 61u);
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
      compiler.set_decoration(id, spv::DecorationBinding, kDispatchRegister);
      out->uses_workgroup_count = true;
    }
  }

  const sc::ShaderResources resources = compiler.get_shader_resources();
  for (const sc::Resource& r : resources.storage_buffers) {
    const u32 set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
    const u32 binding = compiler.get_decoration(r.id, spv::DecorationBinding);
    const bool srv =
        std::find(options.read_only_storage.begin(),
                  options.read_only_storage.end(),
                  base::Pair<u32, u32>{set, binding}) !=
        options.read_only_storage.end();
    if (srv)
      compiler.set_decoration(r.id, spv::DecorationNonWritable);
    else
      compiler.set_hlsl_force_storage_buffer_as_uav(set, binding);
  }
  for (u32 id : flatten)
    compiler.flatten_buffer_block(id);

  std::string hlsl = compiler.compile();
  if (emulate.barycentrics && options.shader_model < 61)
    EmulateBarycentrics(hlsl);
  if (options.shader_model < 61 &&
      hlsl.find("GetAttributeAtVertex") != std::string::npos)
    throw std::runtime_error("per-vertex fragment inputs need SM 6.1");
  out->uses_draw_params = compiler.is_hlsl_aux_buffer_binding_used(
      sc::HLSL_AUX_BINDING_BASE_VERTEX_INSTANCE);

  for (u32 id : flatten) {
    const u32 type = compiler.get_type_from_variable(id).self;
    std::string name = compiler.get_name(type);
    if (name.empty())
      name = compiler.get_fallback_name(type);
    const bool is_push =
        compiler.get_storage_class(id) == spv::StorageClassPushConstant;
    const u32 reg = is_push ? kPushRegister
                            : compiler.get_decoration(id, spv::DecorationBinding);
    const u32 space =
        is_push ? kInternalSpace
                : compiler.get_decoration(id, spv::DecorationDescriptorSet);
    static const std::regex decl(R"(uniform (\w+) (\w+)\[(\d+)\];)");
    std::smatch m;
    std::string::const_iterator from = hlsl.cbegin();
    while (std::regex_search(from, hlsl.cend(), m, decl)) {
      if (m[2] == name) {
        const std::string block = "cbuffer DeltaFlat_" + name +
                                  " : register(b" + std::to_string(reg) +
                                  ", space" + std::to_string(space) +
                                  ")\n{\n    " + m[1].str() + " " + name +
                                  "[" + m[3].str() + "];\n};";
        const size_t at = m.position(0) + (from - hlsl.cbegin());
        hlsl.replace(at, m.length(0), block);
        break;
      }
      from = m.suffix().first;
    }
  }

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
  if (model == spv::ExecutionModelGLCompute && options.dispatch_base) {
    // D3D12 has no dispatch base: offset the ids the shader reads.
    std::string size = "uint3(";
    for (u32 i = 0; i < 3; i++)
      size += std::to_string(compiler.get_execution_mode_argument(
                  spv::ExecutionModeLocalSize, i)) +
              (i < 2 ? ", " : ")");
    Replace(hlsl, "gl_WorkGroupID = stage_input.gl_WorkGroupID;",
            "gl_WorkGroupID = stage_input.gl_WorkGroupID + delta_group_base;");
    Replace(hlsl,
            "gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;",
            "gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID + "
            "delta_group_base * " +
                size + ";");
    hlsl = "cbuffer DeltaGroupBase : register(b" +
           std::to_string(kGroupBaseRegister) + ", space" +
           std::to_string(kInternalSpace) +
           ")\n{\n    uint3 delta_group_base;\n};\n\n" + hlsl;
  }
  if (model == spv::ExecutionModelGeometry) {
    // A geometry shader without inputs or outputs still names both structs.
    for (const char* name : {"SPIRV_Cross_Output", "SPIRV_Cross_Input"})
      if (hlsl.find(std::string("struct ") + name) == std::string::npos)
        hlsl = std::string("struct ") + name +
               "\n{\n    float4 delta_unused : TEXCOORD31;\n};\n\n" + hlsl;
    // SPIRV-Cross copies gl_in[].gl_Position into gl_PositionIn but still
    // reads the GLSL name.
    static const std::regex gl_in(R"(gl_in\[([^\]]+)\]\.gl_Position)");
    hlsl = std::regex_replace(hlsl, gl_in, "gl_PositionIn[$1]");
  }
  if (model == spv::ExecutionModelFragment ||
      model == spv::ExecutionModelGeometry)
    MatchInputs(hlsl, std::string(options.producer_outputs.c_str(),
                                  options.producer_outputs.size()));
  size_t begin, end;
  if ((model == spv::ExecutionModelVertex ||
       model == spv::ExecutionModelGeometry) &&
      FindStruct(hlsl, "SPIRV_Cross_Output", &begin, &end))
    out->outputs = base::String(hlsl.data() + begin, end - begin);
  if (emulate.integer_sampling && options.shader_model < 67)
    IntegerSampling(hlsl);
  return ExtendedArithmetic(hlsl) + hlsl;
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

base::String BarycentricGeometryShader(const base::String& vertex_outputs_in,
                                       base::String* outputs_out) {
  const std::string vertex_outputs(vertex_outputs_in.c_str(),
                                   vertex_outputs_in.size());
  std::string outputs_storage;
  std::string* outputs = &outputs_storage;
  *outputs = vertex_outputs +
             "    float3 delta_bary : DELTABARY;\n"
             "    noperspective float3 delta_bary_np : DELTABARYNP;\n";
  std::string copy;
  for (const Element& e : ParseElements(vertex_outputs))
    copy += "        o." + e.name + " = v[i]." + e.name + ";\n";
  const std::string gs = "struct DeltaIn\n{\n" + vertex_outputs +
         "};\n\nstruct DeltaOut\n{\n" + *outputs +
         "};\n\n[maxvertexcount(3)]\n"
         "void main(triangle DeltaIn v[3], "
         "inout TriangleStream<DeltaOut> s)\n{\n"
         "    for (uint i = 0; i < 3; i++)\n    {\n        DeltaOut o;\n" +
         copy +
         "        o.delta_bary = float3(i == 0, i == 1, i == 2);\n"
         "        o.delta_bary_np = o.delta_bary;\n"
         "        s.Append(o);\n    }\n}\n";
  *outputs_out = base::String(outputs->c_str(), outputs->size());
  return base::String(gs.c_str(), gs.size());
}

bool LowerToHlsl(const u32* words,
                 size_t count,
                 const LowerOptions& options,
                 LoweredShader* out) {
  const base::Vector<u32> patched = PatchSpirvForHlsl(words, count);
  std::vector<u32> flatten;
  Emulation emulate;
  for (;;) {
    *out = {};
    try {
      const std::string hlsl = Lower(patched.data(), patched.size(), options,
                                     flatten, emulate, out);
      out->hlsl = base::String(hlsl.c_str(), hlsl.size());
      break;
    } catch (const std::exception& e) {
      u32 id;
      if (flatten.size() < 16 && PackingFailure(e.what(), &id) &&
          std::find(flatten.begin(), flatten.end(), id) == flatten.end()) {
        flatten.push_back(id);
        continue;
      }
      if (!emulate.integer_sampling &&
          std::strstr(e.what(), "Sampling non-float textures")) {
        emulate.integer_sampling = true;
        continue;
      }
      if (!emulate.barycentrics && options.emulate_barycentrics &&
          std::strstr(e.what(), "SM 6.1") &&
          std::strstr(e.what(), "barycentrics")) {
        emulate.barycentrics = true;
        continue;
      }
      out->error = e.what();
      return false;
    }
  }
  const std::string profile = std::string(StagePrefix(options.stage)) + "_" +
                              std::to_string(options.shader_model / 10) + "_" +
                              std::to_string(options.shader_model % 10);
  out->profile = base::String(profile.c_str(), profile.size());
  return true;
}

}  // namespace gpu::d3d12
