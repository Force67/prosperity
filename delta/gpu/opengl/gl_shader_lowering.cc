/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/opengl/gl_shader_lowering.h"

#include <spirv_glsl.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_map>

namespace gpu::opengl {

namespace {

using spirv_cross::SPIRType;

class Lowerer : public spirv_cross::CompilerGLSL {
 public:
  using CompilerGLSL::CompilerGLSL;

  bool Std140(u32 type_id) {
    const SPIRType& t = get<SPIRType>(type_id);
    return buffer_is_packing_standard(t, spirv_cross::BufferPackingStd140) ||
           buffer_is_packing_standard(t, spirv_cross::BufferPackingStd140EnhancedLayout);
  }

  // 'f', 'i' or 'u' when every scalar in the block has that type.
  char CommonType(u32 type_id) {
    SPIRType::BaseType base;
    if (!get_common_basic_type(get<SPIRType>(type_id), base))
      return 0;
    switch (base) {
      case SPIRType::Float:
        return 'f';
      case SPIRType::Int:
        return 'i';
      case SPIRType::UInt:
        return 'u';
      default:
        return 0;
    }
  }

  // Consecutive locations an input of this type takes.
  u32 LocationCount(u32 type_id) {
    const SPIRType& t = get<SPIRType>(type_id);
    u32 n = t.columns;
    for (u32 i = 0; i < t.array.size(); i++)
      n *= t.array_size_literal[i] ? t.array[i] : 1;
    return n ? n : 1;
  }

  u32 DeclaredSize(u32 type_id) {
    return static_cast<u32>(get_declared_struct_size(get<SPIRType>(type_id)));
  }

  // The element type of a block that is one tightly packed runtime array,
  // which a pointer can stand in for; empty for any other block.
  std::string PointerElement(u32 type_id) {
    const SPIRType& t = get<SPIRType>(type_id);
    if (t.member_types.size() != 1)
      return {};
    const u32 member = t.member_types[0];
    const SPIRType& m = get<SPIRType>(member);
    if (m.array.size() != 1 || m.array[0] != 0 || m.columns != 1 ||
        m.width != 32 ||
        (m.basetype != SPIRType::UInt && m.basetype != SPIRType::Int &&
         m.basetype != SPIRType::Float) ||
        get_decoration(member, spv::DecorationArrayStride) != m.vecsize * 4)
      return {};
    return type_to_glsl(m);
  }

  // Reads and writes `var` through entry `index` of the pointer table.
  void Spill(u32 var, u32 type_id, u32 index) {
    const std::string name = "DELTA_BUF_" + std::to_string(index);
    set_name(var, name);
    const u32 member = get<SPIRType>(type_id).member_types[0];
    spilled_[var] = {index, get_decoration(member, spv::DecorationArrayStride)};
    add_header_line("#define " + name + " ((" + PointerElement(type_id) +
                    "*)packUint2x32(" + kPointerTable + "[" +
                    std::to_string(index) + "].xy))");
  }

  // GLSL for GL_KHR_shader_subgroup is the same as Vulkan's; SPIRV-Cross
  // only knows that it is, not that GL has it.
  void emit_subgroup_op(const spirv_cross::Instruction& instr) override {
    if (!khr_subgroup_ || is_supported_subgroup_op_in_opengl(
                              static_cast<spv::Op>(instr.op), stream(instr)))
      return CompilerGLSL::emit_subgroup_op(instr);
    options.vulkan_semantics = true;
    CompilerGLSL::emit_subgroup_op(instr);
    options.vulkan_semantics = false;
  }
  void set_khr_subgroup(bool on) { khr_subgroup_ = on; }

  void AddDispatchBase() {
    dispatch_base_ = true;
    add_header_line(std::string("uniform uvec3 ") + kDispatchBase + ";");
  }

  std::string builtin_to_glsl(spv::BuiltIn builtin,
                              spv::StorageClass storage) override {
    if (dispatch_base_ && builtin == spv::BuiltInWorkgroupId)
      return std::string("(gl_WorkGroupID + ") + kDispatchBase + ")";
    if (dispatch_base_ && builtin == spv::BuiltInGlobalInvocationId)
      return std::string("(gl_GlobalInvocationID + ") + kDispatchBase +
             " * gl_WorkGroupSize)";
    return CompilerGLSL::builtin_to_glsl(builtin, storage);
  }

  void emit_buffer_block(const spirv_cross::SPIRVariable& var) override {
    if (!spilled_.count(var.self))
      CompilerGLSL::emit_buffer_block(var);
  }

  // A spilled block is its array: `ptr[i]`, not `block._m0[i]`.
  std::string to_member_reference(u32 base,
                                  const SPIRType& type,
                                  u32 index,
                                  bool resolved) override {
    const spirv_cross::SPIRVariable* var = maybe_get_backing_variable(base);
    if (var && spilled_.count(var->self))
      return {};
    return CompilerGLSL::to_member_reference(base, type, index, resolved);
  }

  void emit_instruction(const spirv_cross::Instruction& instr) override {
    const auto op = static_cast<spv::Op>(instr.op);
    const u32* ops = stream(instr);
    if (op == spv::OpArrayLength && instr.length >= 4) {
      const spirv_cross::SPIRVariable* var = maybe_get_backing_variable(ops[2]);
      auto it = var ? spilled_.find(var->self) : spilled_.end();
      if (it != spilled_.end()) {
        set<spirv_cross::SPIRExpression>(
            ops[1],
            spirv_cross::join(type_to_glsl(get<SPIRType>(ops[0])), "(",
                              kPointerTable, "[", it->second.index, "].z / ",
                              it->second.stride, "u)"),
            ops[0], true);
        return;
      }
    }
    // GLSL's imulExtended/umulExtended take only int/uint operands, which
    // SPIR-V does not require of OpSMulExtended/OpUMulExtended.
    if ((op != spv::OpSMulExtended && op != spv::OpUMulExtended) ||
        instr.length < 4)
      return CompilerGLSL::emit_instruction(instr);
    const SPIRType::BaseType want =
        op == spv::OpSMulExtended ? SPIRType::Int : SPIRType::UInt;
    const SPIRType& type = get<SPIRType>(ops[0]);
    const SPIRType& member = get<SPIRType>(type.member_types[0]);
    if (member.basetype == want && expression_type(ops[2]).basetype == want &&
        expression_type(ops[3]).basetype == want)
      return CompilerGLSL::emit_instruction(instr);
    SPIRType cast = member;
    cast.basetype = want;
    const std::string t = type_to_glsl(cast);
    const std::string m = type_to_glsl(member);
    emit_uninitialized_temporary_expression(ops[0], ops[1]);
    const std::string result = to_expression(ops[1]);
    const std::string hi = result + "_hi", lo = result + "_lo";
    statement(t, " ", hi, ", ", lo, ";");
    statement(op == spv::OpSMulExtended ? "imulExtended(" : "umulExtended(",
              t, "(", to_unpacked_expression(ops[2]), "), ", t, "(",
              to_unpacked_expression(ops[3]), "), ", hi, ", ", lo, ");");
    statement(result, ".", to_member_name(type, 0), " = ", m, "(", lo, ");");
    statement(result, ".", to_member_name(type, 1), " = ", m, "(", hi, ");");
  }

  // A uniform block std140 cannot express becomes a read-only storage block:
  // GLSL has no std430 uniform blocks.
  void MakeStorageBlock(u32 type_id) {
    unset_decoration(type_id, spv::DecorationBlock);
    set_decoration(type_id, spv::DecorationBufferBlock);
    const SPIRType& t = get<SPIRType>(type_id);
    for (u32 i = 0; i < t.member_types.size(); i++)
      set_member_decoration(type_id, i, spv::DecorationNonWritable);
  }

 private:
  struct Spilled {
    u32 index;
    u32 stride;
  };
  std::unordered_map<u32, Spilled> spilled_;
  bool dispatch_base_ = false;
  bool khr_subgroup_ = false;
};

u32 StageOf(spv::ExecutionModel model) {
  switch (model) {
    case spv::ExecutionModelVertex:
      return rhi::kStageVertex;
    case spv::ExecutionModelGeometry:
      return rhi::kStageGeometry;
    case spv::ExecutionModelFragment:
      return rhi::kStageFragment;
    case spv::ExecutionModelGLCompute:
      return rhi::kStageCompute;
    case spv::ExecutionModelMeshEXT:
    case spv::ExecutionModelMeshNV:
      return rhi::kStageMesh;
    default:
      return 0;
  }
}

const char* StageSuffix(u32 stage) {
  switch (stage) {
    case rhi::kStageVertex:
      return "vs";
    case rhi::kStageGeometry:
      return "gs";
    case rhi::kStageFragment:
      return "fs";
    case rhi::kStageCompute:
      return "cs";
    default:
      return "ms";
  }
}

u64 Key(u32 set, u32 binding) {
  return (u64(set) << 32) | binding;
}

// Atomics take a pointer, not an lvalue behind one: atomicAdd(P[i], v)
// becomes atomicAdd((P + (i)), v) for the spilled buffers.
void RewritePointerAtomics(std::string& s) {
  static constexpr char kCall[] = "(DELTA_BUF_";
  for (size_t at = s.find(kCall); at != std::string::npos;
       at = s.find(kCall, at + 1)) {
    size_t word = at;
    while (word > 0 && (std::isalnum(static_cast<unsigned char>(s[word - 1])) ||
                        s[word - 1] == '_'))
      word--;
    if (s.compare(word, 6, "atomic") != 0)
      continue;
    const size_t open = s.find('[', at);
    if (open == std::string::npos)
      break;
    size_t close = open + 1;
    for (int depth = 1; close < s.size(); close++) {
      if (s[close] == '[')
        depth++;
      else if (s[close] == ']' && --depth == 0)
        break;
    }
    if (close >= s.size())
      break;
    const std::string name = s.substr(at + 1, open - at - 1);
    const std::string index = s.substr(open + 1, close - open - 1);
    s.replace(at + 1, close - at, "(" + name + " + (" + index + "))");
  }
}

void ReplaceAll(std::string& s, const char* from, const char* to) {
  const size_t n = std::strlen(from), m = std::strlen(to);
  for (size_t at = s.find(from); at != std::string::npos;
       at = s.find(from, at + m))
    s.replace(at, n, to);
}

bool Accepts(rhi::BindingType type, SlotKind kind) {
  switch (type) {
    case rhi::BindingType::kUniformBuffer:
    case rhi::BindingType::kUniformBufferDynamic:
      return kind == SlotKind::kUniformBuffer;
    case rhi::BindingType::kStorageBuffer:
    case rhi::BindingType::kStorageBufferDynamic:
      return kind == SlotKind::kStorageBuffer;
    case rhi::BindingType::kSampledTexture:
      return kind == SlotKind::kTexture;
    case rhi::BindingType::kStorageTexture:
      return kind == SlotKind::kImage;
  }
  return false;
}

struct Stage {
  u32 stage = 0;
  bool dispatch_base = false;
  std::unique_ptr<Lowerer> compiler;
  spirv_cross::ShaderResources resources;
};

struct Use {
  SlotKind kind;
  bool demoted = false;
  bool spillable = true;
  bool spilled = false;
  u32 slot = 0;
};

}  // namespace

bool LowerProgram(const StageCode* stages,
                  u32 count,
                  const std::vector<const rhi::BindGroupLayoutDesc*>& groups,
                  const GlslFeatures& features,
                  std::vector<std::string>* glsl,
                  ProgramInterface* program,
                  std::string* error) {
  *program = {};
  glsl->clear();
  try {
    std::vector<Stage> parsed(count);
    std::map<u64, Use> uses;
    for (u32 i = 0; i < count; i++) {
      Stage& s = parsed[i];
      s.compiler = std::make_unique<Lowerer>(stages[i].code.words,
                                             stages[i].code.count);
      s.stage = StageOf(s.compiler->get_execution_model());
      s.dispatch_base =
          stages[i].dispatch_base && s.stage == rhi::kStageCompute;
      if (!s.stage || s.stage != stages[i].stage) {
        *error = "unsupported or mismatched execution model";
        return false;
      }
      const auto active = s.compiler->get_active_interface_variables();
      s.resources = s.compiler->get_shader_resources(active);
      s.compiler->set_enabled_interface_variables(active);
      const auto& r = s.resources;
      if (!r.separate_images.empty() || !r.separate_samplers.empty()) {
        *error = "separate images/samplers";
        return false;
      }
      if (!r.acceleration_structures.empty() || !r.atomic_counters.empty() ||
          !r.subpass_inputs.empty()) {
        *error = "unsupported resource type";
        return false;
      }
      auto note = [&](const spirv_cross::SmallVector<spirv_cross::Resource>&
                          list,
                      SlotKind kind) {
        for (const auto& res : list) {
          const u32 set = s.compiler->get_decoration(
              res.id, spv::DecorationDescriptorSet);
          const u32 binding =
              s.compiler->get_decoration(res.id, spv::DecorationBinding);
          Use& use = uses.try_emplace(Key(set, binding), Use{kind}).first->second;
          if (use.kind != kind) {
            *error = "one binding used as two resource kinds";
            return false;
          }
          if (kind == SlotKind::kUniformBuffer &&
              !s.compiler->Std140(res.base_type_id))
            use.demoted = true;
          if (s.compiler->PointerElement(res.base_type_id).empty())
            use.spillable = false;
        }
        return true;
      };
      if (!note(r.uniform_buffers, SlotKind::kUniformBuffer) ||
          !note(r.storage_buffers, SlotKind::kStorageBuffer) ||
          !note(r.sampled_images, SlotKind::kTexture) ||
          !note(r.storage_images, SlotKind::kImage))
        return false;
    }

    // Storage buffers past the per-stage limit, highest bindings first,
    // become pointers in every stage.
    for (Stage& s : parsed) {
      std::vector<u64> keys;
      auto collect = [&](const spirv_cross::SmallVector<spirv_cross::Resource>&
                             list) {
        for (const auto& res : list) {
          const u64 key = Key(
              s.compiler->get_decoration(res.id, spv::DecorationDescriptorSet),
              s.compiler->get_decoration(res.id, spv::DecorationBinding));
          const Use& use = uses[key];
          if (use.kind == SlotKind::kStorageBuffer || use.demoted)
            keys.push_back(key);
        }
      };
      collect(s.resources.uniform_buffers);
      collect(s.resources.storage_buffers);
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      size_t kept = keys.size();
      for (u64 key : keys)
        if (uses[key].spilled)
          kept--;
      for (auto it = keys.rbegin();
           it != keys.rend() && kept > features.max_stage_storage_buffers;
           ++it) {
        Use& use = uses[*it];
        if (use.spilled || !use.spillable || !features.buffer_pointers)
          continue;
        use.spilled = true;
        kept--;
      }
      if (kept > features.max_stage_storage_buffers) {
        *error = "too many storage buffers in one stage for GL";
        return false;
      }
    }

    u32 next[kSlotKinds] = {kPushUboSlot + 1, 0, 0, 0, 0};
    const u32 limit[kSlotKinds] = {
        features.max_uniform_buffers, features.max_storage_buffers,
        features.max_textures, features.max_images, ~0u};
    for (auto& [key, use] : uses) {
      const u32 set = static_cast<u32>(key >> 32);
      const u32 binding = static_cast<u32>(key);
      const rhi::BindingLayout* layout = nullptr;
      if (set < groups.size() && groups[set])
        for (const auto& b : groups[set]->bindings)
          if (b.binding == binding)
            layout = &b;
      const SlotKind kind = use.spilled   ? SlotKind::kPointer
                            : use.demoted ? SlotKind::kStorageBuffer
                                          : use.kind;
      if (!layout || !Accepts(layout->type, use.kind)) {
        *error = "set " + std::to_string(set) + " binding " +
                 std::to_string(binding) + " missing from the layout";
        return false;
      }
      const u32 k = static_cast<u32>(kind);
      use.slot = next[k]++;
      if (use.slot >= limit[k]) {
        *error = "too many resources of one kind for GL";
        return false;
      }
      program->slots.push_back({set, binding, kind, use.slot});
    }
    program->pointer_count = next[static_cast<u32>(SlotKind::kPointer)];

    for (Stage& s : parsed) {
      Lowerer& c = *s.compiler;
      u32 spills = 0;
      auto remap = [&](const spirv_cross::SmallVector<spirv_cross::Resource>&
                           list) {
        for (const auto& res : list) {
          const Use& use = uses[Key(
              c.get_decoration(res.id, spv::DecorationDescriptorSet),
              c.get_decoration(res.id, spv::DecorationBinding))];
          if (use.spilled) {
            if (!spills++) {
              c.require_extension("GL_NV_shader_buffer_load");
              c.require_extension("GL_NV_gpu_shader5");
              c.add_header_line(std::string("uniform uvec4 ") + kPointerTable +
                                "[" + std::to_string(program->pointer_count) +
                                "];");
            }
            c.Spill(res.id, res.base_type_id, use.slot);
            continue;
          }
          if (use.demoted)
            c.MakeStorageBlock(res.base_type_id);
          c.unset_decoration(res.id, spv::DecorationDescriptorSet);
          c.set_decoration(res.id, spv::DecorationBinding, use.slot);
        }
      };
      remap(s.resources.uniform_buffers);
      remap(s.resources.storage_buffers);
      remap(s.resources.sampled_images);
      remap(s.resources.storage_images);

      spirv_cross::CompilerGLSL::Options options;
      options.version = 460;
      options.es = false;
      options.vulkan_semantics = false;
      options.enable_420pack_extension = true;
      options.vertex.support_nonzero_base_instance = true;
      for (const auto& pc : s.resources.push_constant_buffers) {
        const char t = c.CommonType(pc.base_type_id);
        const std::string name =
            std::string("delta_push_") + StageSuffix(s.stage);
        c.set_name(pc.base_type_id, name);
        if (t) {
          // A default-block uniform array: GL's nearest thing to push
          // constants, and free of std140's 16-byte array strides.
          c.flatten_buffer_block(pc.id);
          program->push_uniforms.push_back(
              {s.stage, name, (c.DeclaredSize(pc.base_type_id) + 15) / 16, t});
        } else {
          options.emit_push_constant_as_uniform_buffer = true;
          c.set_decoration(pc.id, spv::DecorationBinding, kPushUboSlot);
          program->push_ubo = true;
        }
      }
      if (s.dispatch_base)
        c.AddDispatchBase();
      if (s.stage == rhi::kStageVertex) {
        // Vulkan allows more attribute locations than GL does (16 on some
        // drivers), and the recompiler leaves gaps: pack them, keeping their
        // order so that multi-location inputs and components sharing a
        // location stay together.
        std::map<u32, u32> packed;
        for (const auto& res : s.resources.stage_inputs) {
          const u32 location =
              c.get_decoration(res.id, spv::DecorationLocation);
          for (u32 k = 0; k < c.LocationCount(res.type_id); k++)
            packed[location + k] = 0;
        }
        u32 next = 0;
        for (auto& [location, attribute] : packed)
          attribute = next++;
        for (const auto& res : s.resources.stage_inputs)
          c.set_decoration(
              res.id, spv::DecorationLocation,
              packed[c.get_decoration(res.id, spv::DecorationLocation)]);
        program->vertex_locations.assign(packed.begin(), packed.end());
        if (next > features.max_vertex_attribs) {
          *error = "too many vertex attributes for GL";
          return false;
        }
      }
      c.set_common_options(options);
      c.set_khr_subgroup(features.khr_subgroup);
      std::string source = c.compile();
      if (spills)
        RewritePointerAtomics(source);
      // Per-vertex inputs take no interpolation qualifier.
      ReplaceAll(source, "flat pervertex", "pervertex");
      if (features.nv_barycentric_only &&
          source.find("GL_EXT_fragment_shader_barycentric") !=
              std::string::npos) {
        ReplaceAll(source, "GL_EXT_fragment_shader_barycentric",
                   "GL_NV_fragment_shader_barycentric");
        ReplaceAll(source, "gl_BaryCoordEXT", "gl_BaryCoordNV");
        ReplaceAll(source, "gl_BaryCoordNoPerspEXT", "gl_BaryCoordNoPerspNV");
        ReplaceAll(source, "pervertexEXT", "pervertexNV");
      }
      glsl->push_back(std::move(source));
    }
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
  return true;
}

bool ReflectLayout(const rhi::ShaderCode& code,
                   u32* stage,
                   std::vector<rhi::BindGroupLayoutDesc>* groups,
                   u32* push_constant_bytes,
                   std::string* error) {
  groups->clear();
  *push_constant_bytes = 0;
  try {
    Lowerer c(code.words, code.count);
    *stage = StageOf(c.get_execution_model());
    if (!*stage) {
      *error = "unsupported execution model";
      return false;
    }
    const auto r = c.get_shader_resources(c.get_active_interface_variables());
    auto add = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& list,
                   rhi::BindingType type) {
      for (const auto& res : list) {
        const u32 set = c.get_decoration(res.id, spv::DecorationDescriptorSet);
        if (groups->size() <= set)
          groups->resize(set + 1);
        rhi::BindingLayout b;
        b.binding = c.get_decoration(res.id, spv::DecorationBinding);
        b.type = type;
        b.stages = *stage;
        (*groups)[set].bindings.push_back(b);
      }
    };
    add(r.uniform_buffers, rhi::BindingType::kUniformBuffer);
    add(r.storage_buffers, rhi::BindingType::kStorageBuffer);
    add(r.sampled_images, rhi::BindingType::kSampledTexture);
    add(r.storage_images, rhi::BindingType::kStorageTexture);
    for (const auto& pc : r.push_constant_buffers)
      *push_constant_bytes = c.DeclaredSize(pc.base_type_id);
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
  return true;
}

}  // namespace gpu::opengl
