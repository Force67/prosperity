/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/rhi/null_device.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace gpu::rhi {
namespace {

std::string Fmt(const char* fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  const int n = vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf, n > 0 ? static_cast<size_t>(n) : 0);
}

const char* StateName(ResourceState s) {
  switch (s) {
    case ResourceState::Undefined: return "undefined";
    case ResourceState::Common: return "common";
    case ResourceState::RenderTarget: return "rendertarget";
    case ResourceState::DepthWrite: return "depthwrite";
    case ResourceState::DepthRead: return "depthread";
    case ResourceState::ShaderRead: return "shaderread";
    case ResourceState::UnorderedAccess: return "uav";
    case ResourceState::CopySrc: return "copysrc";
    case ResourceState::CopyDst: return "copydst";
    case ResourceState::VertexInput: return "vertexinput";
    case ResourceState::IndexInput: return "indexinput";
    case ResourceState::IndirectArgument: return "indirect";
  }
  return "?";
}

const char* LoadName(LoadOp op) {
  switch (op) {
    case LoadOp::Load: return "load";
    case LoadOp::Clear: return "clear";
    case LoadOp::Discard: return "discard";
  }
  return "?";
}

class NullBuffer : public Buffer {
 public:
  explicit NullBuffer(const BufferDesc& desc) : storage_(desc.size) {
    desc_ = desc;
  }
  void* Map() override { return storage_.data(); }

 private:
  std::vector<u8> storage_;
};

class NullTexture : public Texture {
 public:
  explicit NullTexture(const TextureDesc& desc) { desc_ = desc; }
};

class NullSampler : public Sampler {};
class NullShader : public ShaderModule {};

class NullBindGroupLayout : public BindGroupLayout {
 public:
  explicit NullBindGroupLayout(const BindGroupLayoutDesc& desc) {
    desc_ = desc;
  }
};

class NullBindGroup : public BindGroup {
 public:
  explicit NullBindGroup(const BindGroupDesc& desc) : desc_(desc) {}
  const BindGroupDesc& desc() const { return desc_; }  // NOLINT: accessor

 private:
  BindGroupDesc desc_;
};

class NullPipeline : public Pipeline {};

class NullCommandList : public CommandList {
 public:
  explicit NullCommandList(NullDevice& device) : device_(device) {}

  void Begin() override { device_.Record("begin"); }
  void End() override { device_.Record("end"); }

  void BeginRenderPass(const RenderPassDesc& pass) override {
    // A pass owns its attachments' state, exactly as the real backends do.
    for (u32 i = 0; i < pass.num_color; i++)
      Transition(pass.color[i].texture, ResourceState::RenderTarget);
    if (pass.has_depth) {
      Transition(pass.depth.texture, pass.depth.read_only
                                         ? ResourceState::DepthRead
                                         : ResourceState::DepthWrite);
    }

    std::string line = Fmt("pass %ux%u", pass.width, pass.height);
    for (u32 i = 0; i < pass.num_color; i++) {
      const ColorAttachment& c = pass.color[i];
      line += Fmt(" color%u=%s:%s", i, device_.Label(c.texture).c_str(),
                  LoadName(c.load));
      if (c.load == LoadOp::Clear) {
        line += Fmt("(%.3f,%.3f,%.3f,%.3f)", c.clear[0], c.clear[1],
                    c.clear[2], c.clear[3]);
      }
    }
    if (pass.has_depth) {
      line += Fmt(" depth=%s:%s", device_.Label(pass.depth.texture).c_str(),
                  LoadName(pass.depth.load));
      if (pass.depth.read_only)
        line += "(readonly)";
    }
    device_.Record(std::move(line));
  }

  void EndRenderPass() override { device_.Record("endpass"); }

  void SetPipeline(Pipeline* pipeline) override {
    device_.Record(Fmt("pipeline %s", device_.Label(pipeline).c_str()));
  }

  void SetBindGroup(u32 index,
                    BindGroup* group,
                    const u32* offsets,
                    u32 num_offsets) override {
    std::string line =
        Fmt("bindgroup %u %s", index, device_.Label(group).c_str());
    for (u32 i = 0; i < num_offsets; i++)
      line += Fmt(" +%u", offsets[i]);
    device_.Record(std::move(line));
  }

  void SetPushConstants(u32 offset, u32 bytes, const void* data) override {
    u32 first = 0;
    if (bytes >= sizeof(u32) && data)
      memcpy(&first, data, sizeof(first));
    device_.Record(
        Fmt("push %u+%u first=%#x", offset, bytes, first));
  }

  void SetVertexBuffer(u32 slot, Buffer* buffer, u64 offset) override {
    device_.Record(Fmt("vbuf %u %s+%llu", slot, device_.Label(buffer).c_str(),
                       (unsigned long long)offset));
  }

  void SetIndexBuffer(Buffer* buffer, u64 offset, IndexType type) override {
    device_.Record(Fmt("ibuf %s+%llu %s", device_.Label(buffer).c_str(),
                       (unsigned long long)offset,
                       type == IndexType::U16 ? "u16" : "u32"));
  }

  void SetViewport(float x,
                   float y,
                   float width,
                   float height,
                   float min_depth,
                   float max_depth) override {
    device_.Record(Fmt("viewport %.1f,%.1f %.1fx%.1f z=%.3f..%.3f", x, y,
                       width, height, min_depth, max_depth));
  }

  void SetScissor(i32 x, i32 y, u32 width, u32 height) override {
    device_.Record(Fmt("scissor %d,%d %ux%u", x, y, width, height));
  }

  void SetBlendConstants(const float rgba[4]) override {
    device_.Record(Fmt("blendconst %.3f,%.3f,%.3f,%.3f", rgba[0], rgba[1],
                       rgba[2], rgba[3]));
  }

  void Draw(u32 vertex_count,
            u32 instance_count,
            u32 first_vertex,
            u32 first_instance) override {
    device_.Record(Fmt("draw v=%u i=%u first=%u,%u", vertex_count,
                       instance_count, first_vertex, first_instance));
  }

  void DrawIndexed(u32 index_count,
                   u32 instance_count,
                   u32 first_index,
                   i32 vertex_offset,
                   u32 first_instance) override {
    device_.Record(Fmt("drawindexed n=%u i=%u first=%u vofs=%d inst=%u",
                       index_count, instance_count, first_index, vertex_offset,
                       first_instance));
  }

  void Dispatch(u32 groups_x, u32 groups_y, u32 groups_z) override {
    device_.Record(
        Fmt("dispatch %u,%u,%u", groups_x, groups_y, groups_z));
  }

  void CopyBuffer(Buffer* dst,
                  u64 dst_offset,
                  Buffer* src,
                  u64 src_offset,
                  u64 bytes) override {
    device_.Record(Fmt("copybuf %s+%llu <- %s+%llu %llu",
                       device_.Label(dst).c_str(),
                       (unsigned long long)dst_offset,
                       device_.Label(src).c_str(),
                       (unsigned long long)src_offset,
                       (unsigned long long)bytes));
  }

  void CopyBufferToTexture(Texture* dst,
                           const TextureRegion& region,
                           Buffer* src,
                           u64 src_offset,
                           u32 src_row_pitch) override {
    device_.Record(Fmt("upload %s mip%u layer%u %ux%u <- %s+%llu pitch=%u",
                       device_.Label(dst).c_str(), region.mip, region.layer,
                       region.width, region.height,
                       device_.Label(src).c_str(),
                       (unsigned long long)src_offset, src_row_pitch));
  }

  void CopyTextureToBuffer(Buffer* dst,
                           u64 dst_offset,
                           u32 dst_row_pitch,
                           Texture* src,
                           const TextureRegion& region) override {
    device_.Record(Fmt("readback %s+%llu pitch=%u <- %s mip%u %ux%u",
                       device_.Label(dst).c_str(),
                       (unsigned long long)dst_offset, dst_row_pitch,
                       device_.Label(src).c_str(), region.mip, region.width,
                       region.height));
  }

  void FillBuffer(Buffer* buffer,
                  u64 offset,
                  u64 bytes,
                  u32 value) override {
    device_.Record(Fmt("fill %s+%llu %llu = %#x",
                       device_.Label(buffer).c_str(),
                       (unsigned long long)offset, (unsigned long long)bytes,
                       value));
  }

  void Transition(Buffer* buffer, ResourceState state) override {
    if (!buffer)
      return;
    if (buffer->state() == state) {
      if (state == ResourceState::UnorderedAccess)
        device_.Record(Fmt("uavbarrier %s", device_.Label(buffer).c_str()));
      return;
    }
    device_.Record(Fmt("barrier %s %s->%s", device_.Label(buffer).c_str(),
                       StateName(buffer->state()), StateName(state)));
    buffer->set_state(state);
  }

  void Transition(Texture* texture, ResourceState state) override {
    if (!texture)
      return;
    if (texture->state() == state) {
      if (state == ResourceState::UnorderedAccess)
        device_.Record(Fmt("uavbarrier %s", device_.Label(texture).c_str()));
      return;
    }
    device_.Record(Fmt("barrier %s %s->%s", device_.Label(texture).c_str(),
                       StateName(texture->state()), StateName(state)));
    texture->set_state(state);
  }

  void PushDebugLabel(const char* label) override {
    device_.Record(Fmt("[%s", label ? label : ""));
  }

  void PopDebugLabel() override { device_.Record("]"); }

 private:
  NullDevice& device_;
};

}  // namespace

NullDevice::NullDevice() {
  caps_.adapter_name = "null";
  caps_.max_texture_2d = 16384;
  caps_.max_bound_textures = 64;
  caps_.independent_blend = true;
  caps_.anisotropic_filtering = true;
}

NullDevice::~NullDevice() = default;

const char* NullDevice::backend_name() const {
  return "null";
}

const DeviceCaps& NullDevice::caps() const {
  return caps_;
}

void NullDevice::Record(std::string line) {
  log_.push_back(std::move(line));
}

void NullDevice::Name(const void* object, const char* name, const char* kind) {
  names_.emplace_back(object, name ? std::string(name)
                                   : Fmt("%s%u", kind, next_id_++));
}

std::string NullDevice::Label(const void* object) const {
  if (!object)
    return "none";
  for (const auto& [ptr, name] : names_) {
    if (ptr == object)
      return name;
  }
  return "unknown";
}

std::string NullDevice::LogText() const {
  std::string out;
  for (const std::string& line : log_) {
    out += line;
    out += '\n';
  }
  return out;
}

void NullDevice::ClearLog() {
  log_.clear();
}

std::vector<std::string> NullDevice::LogMatching(const char* needle) const {
  std::vector<std::string> out;
  for (const std::string& line : log_) {
    if (line.find(needle) != std::string::npos)
      out.push_back(line);
  }
  return out;
}

std::unique_ptr<Buffer> NullDevice::CreateBuffer(const BufferDesc& desc) {
  auto buffer = std::make_unique<NullBuffer>(desc);
  Name(buffer.get(), desc.name, "buf");
  return buffer;
}

std::unique_ptr<Texture> NullDevice::CreateTexture(const TextureDesc& desc) {
  auto texture = std::make_unique<NullTexture>(desc);
  Name(texture.get(), desc.name, "tex");
  return texture;
}

std::unique_ptr<Sampler> NullDevice::CreateSampler(const SamplerDesc&) {
  auto sampler = std::make_unique<NullSampler>();
  Name(sampler.get(), nullptr, "smp");
  return sampler;
}

std::unique_ptr<ShaderModule> NullDevice::CreateShader(
    const ShaderDesc& desc) {
  auto shader = std::make_unique<NullShader>();
  Name(shader.get(), desc.name, "shader");
  return shader;
}

std::unique_ptr<BindGroupLayout> NullDevice::CreateBindGroupLayout(
    const BindGroupLayoutDesc& desc) {
  auto layout = std::make_unique<NullBindGroupLayout>(desc);
  Name(layout.get(), desc.name, "layout");
  return layout;
}

std::unique_ptr<BindGroup> NullDevice::CreateBindGroup(
    const BindGroupDesc& desc) {
  auto group = std::make_unique<NullBindGroup>(desc);
  Name(group.get(), desc.name, "group");
  return group;
}

std::unique_ptr<Pipeline> NullDevice::CreateGraphicsPipeline(
    const GraphicsPipelineDesc& desc) {
  auto pipeline = std::make_unique<NullPipeline>();
  Name(pipeline.get(), desc.name, "pipe");
  return pipeline;
}

std::unique_ptr<Pipeline> NullDevice::CreateComputePipeline(
    const ComputePipelineDesc& desc) {
  auto pipeline = std::make_unique<NullPipeline>();
  Name(pipeline.get(), desc.name, "pipe");
  return pipeline;
}

std::unique_ptr<CommandList> NullDevice::CreateCommandList() {
  return std::make_unique<NullCommandList>(*this);
}

u64 NullDevice::Submit(CommandList*) {
  const u64 id = next_submission_++;
  Record(Fmt("submit %llu", (unsigned long long)id));
  return id;
}

void NullDevice::Wait(u64 submission) {
  Record(Fmt("wait %llu", (unsigned long long)submission));
}

void NullDevice::WaitIdle() {
  Record("waitidle");
}

}  // namespace gpu::rhi
