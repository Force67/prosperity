/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/opengl/gl_shader_lowering.h"

#include <spirv_glsl.hpp>

#include <cstring>
#include <map>
#include <memory>

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

  u32 DeclaredSize(u32 type_id) {
    return static_cast<u32>(get_declared_struct_size(get<SPIRType>(type_id)));
  }

  // GLSL's imulExtended/umulExtended take only int/uint operands, which
  // SPIR-V does not require of OpSMulExtended/OpUMulExtended.
  void emit_instruction(const spirv_cross::Instruction& instr) override {
    const auto op = static_cast<spv::Op>(instr.op);
    if ((op != spv::OpSMulExtended && op != spv::OpUMulExtended) ||
        instr.length < 4)
      return CompilerGLSL::emit_instruction(instr);
    const u32* ops = stream(instr);
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
  std::unique_ptr<Lowerer> compiler;
  spirv_cross::ShaderResources resources;
};

struct Use {
  SlotKind kind;
  bool demoted = false;
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
        }
        return true;
      };
      if (!note(r.uniform_buffers, SlotKind::kUniformBuffer) ||
          !note(r.storage_buffers, SlotKind::kStorageBuffer) ||
          !note(r.sampled_images, SlotKind::kTexture) ||
          !note(r.storage_images, SlotKind::kImage))
        return false;
    }

    u32 next[4] = {kPushUboSlot + 1, 0, 0, 0};
    const u32 limit[4] = {features.max_uniform_buffers,
                          features.max_storage_buffers, features.max_textures,
                          features.max_images};
    for (auto& [key, use] : uses) {
      const u32 set = static_cast<u32>(key >> 32);
      const u32 binding = static_cast<u32>(key);
      const rhi::BindingLayout* layout = nullptr;
      if (set < groups.size() && groups[set])
        for (const auto& b : groups[set]->bindings)
          if (b.binding == binding)
            layout = &b;
      const SlotKind kind =
          use.demoted ? SlotKind::kStorageBuffer : use.kind;
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

    for (Stage& s : parsed) {
      Lowerer& c = *s.compiler;
      auto remap = [&](const spirv_cross::SmallVector<spirv_cross::Resource>&
                           list) {
        for (const auto& res : list) {
          const Use& use = uses[Key(
              c.get_decoration(res.id, spv::DecorationDescriptorSet),
              c.get_decoration(res.id, spv::DecorationBinding))];
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
      c.set_common_options(options);
      std::string source = c.compile();
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
