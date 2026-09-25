/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The GL objects behind the rhi classes. Private to gl_rhi_*.cc.
//
// Threads: one render thread owns the context that replays command lists,
// and the objects GL does not share between contexts (framebuffers, vertex
// arrays, queries). A resource thread creates textures, buffers and samplers
// on a shared context, compile threads build programs, and a waiter thread
// retires fences. A command list records into a CPU-side stream, with every
// binding already resolved to GL names and slots; the render thread replays
// it at Submit.

#include <epoxy/gl.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/arch.h"
#include "gpu/opengl/gl_context.h"
#include "gpu/opengl/gl_rhi.h"
#include "gpu/opengl/gl_shader_lowering.h"
#include "gpu/rhi/device.h"

namespace gpu::opengl {

constexpr u32 kMaxGroups = 8;
constexpr u32 kMaxDynamicOffsets = 16;
constexpr u32 kMaxVertexBuffers = 16;
constexpr u32 kMaxPushBytes = 256;
constexpr u32 kMaxPointers = 128;
constexpr u8 kNotDynamic = 0xff;

struct GlFormat {
  GLenum internal = 0;  // texture storage; 0 when not a texture format
  GLenum format = 0;    // client format of pixel transfers
  GLenum type = 0;
  GLint vertex_size = 0;  // 0 when not a vertex format
  GLenum vertex_type = 0;
  bool vertex_normalized = false;
};
const GlFormat& ToGl(rhi::Format format);

class GlDevice;
class GlView;

class GlBuffer final : public rhi::Buffer {
 public:
  explicit GlBuffer(const rhi::BufferDesc& desc) { desc_ = desc; }
  void set_mapped(u8* p) { mapped_ = p; }
  GLuint name = 0;
  u64 address = 0;  // GPU address of a storage buffer, for pointer slots
};

class GlTexture final : public rhi::Texture {
 public:
  explicit GlTexture(const rhi::TextureDesc& desc) { desc_ = desc; }
  GLuint name = 0;
  GLenum target = 0;
  GLenum internal = 0;
  // Array layers are the y (1D) or z (2D) coordinate of transfers.
  bool arrayed = false;
};

struct FboKey {
  const GlView* colors[8] = {};
  const GlView* depth = nullptr;
  u8 depth_planes = 0;  // rhi::Aspect bits attached from `depth`
  bool operator==(const FboKey& o) const;
};
struct FboKeyHash {
  size_t operator()(const FboKey& key) const;
};

class GlView final : public rhi::TextureView {
 public:
  GlView(rhi::Texture* texture, const rhi::TextureViewDesc& desc) {
    texture_ = texture;
    desc_ = desc;
  }
  // A glTextureView, or the texture itself where GL cannot view it (a 3D
  // slice as 2D): then `level` and `layer` select within it.
  GLuint name = 0;
  bool owns_name = true;
  GLint level = 0;
  GLint layer = -1;  // -1: all layers (a layered attachment)
  GLenum internal = 0;
  rhi::Format format = rhi::Format::kUndefined;
  // Framebuffers this view is attached to; render thread only.
  mutable std::vector<FboKey> fbos;
};

class GlSampler final : public rhi::Sampler {
 public:
  GLuint name = 0;
  // The same sampler without filtering, for integer textures: GL treats a
  // filtered integer texture as incomplete.
  GLuint nearest = 0;
};

class GlBindGroupLayout final : public rhi::BindGroupLayout {
 public:
  explicit GlBindGroupLayout(const rhi::BindGroupLayoutDesc& desc) {
    desc_ = desc;
  }
  // Per binding: its index among the dynamic offsets, or kNotDynamic.
  std::vector<u8> dynamic_index;
  i32 Position(u32 binding) const;
};

// One binding with its GL names resolved.
struct BoundResource {
  GLuint name = 0;     // buffer or texture
  GLuint sampler = 0;  // sampled textures
  u8 dynamic = kNotDynamic;
  u64 offset = 0;
  u64 size = 0;         // bytes from offset
  u64 buffer_size = 0;  // clamps dynamic windows
  u64 address = 0;
  GLint level = 0;      // storage images
  GLint layer = 0;
  GLboolean layered = GL_TRUE;
  GLenum format = 0;
};

class GlBindGroup final : public rhi::BindGroup {
 public:
  GlBindGroupLayout* layout = nullptr;
  std::vector<BoundResource> entries;  // parallel to layout bindings
};

class GlPipelineLayout final : public rhi::PipelineLayout {
 public:
  explicit GlPipelineLayout(const rhi::PipelineLayoutDesc& desc) {
    desc_ = desc;
  }
};

// Which bindings of each group a program uses, and their GL slots. Interned:
// pipelines with equal maps share one, so switching between them keeps the
// bound groups.
struct SlotMap {
  struct Slot {
    u32 binding = 0;
    i32 position = -1;  // in the layout the map was built against
    SlotKind kind = SlotKind::kUniformBuffer;
    u32 slot = 0;
  };
  struct Group {
    const GlBindGroupLayout* layout = nullptr;
    std::vector<Slot> slots;
  };
  std::vector<Group> groups;
};

// Vertex attribute formats and per-binding divisors, i.e. one vertex array
// object. Interned; the VAO itself is made on the render thread.
struct VertexInput {
  struct Attribute {
    u32 location = 0;
    u32 binding = 0;
    GLint size = 0;
    GLenum type = 0;
    u8 kind = 0;  // 0 float, 1 normalized, 2 integer
    u32 offset = 0;
  };
  std::vector<Attribute> attributes;
  std::vector<u8> per_instance;
  GLuint vao = 0;
};

struct BlendState {
  bool enable = false;
  GLenum src_rgb = GL_ONE, dst_rgb = GL_ZERO, op_rgb = GL_FUNC_ADD;
  GLenum src_alpha = GL_ONE, dst_alpha = GL_ZERO, op_alpha = GL_FUNC_ADD;
  u8 mask = 0xF;
  bool operator==(const BlendState& o) const;
};

struct StencilState {
  GLenum func = GL_ALWAYS;
  GLint ref = 0;
  GLuint compare_mask = 0xFF, write_mask = 0xFF;
  GLenum fail = GL_KEEP, depth_fail = GL_KEEP, pass = GL_KEEP;
  bool operator==(const StencilState& o) const;
};

// Fixed-function state of a graphics pipeline.
struct RasterState {
  bool primitive_restart = false;
  bool cull = false;
  GLenum cull_face = GL_BACK;
  bool front_ccw = true;
  bool depth_clamp = false;
  bool depth_test = false;
  bool depth_write = false;
  GLenum depth_func = GL_ALWAYS;
  bool stencil_test = false;
  StencilState front, back;
  BlendState blend[8];
};

class GlPipeline final : public rhi::Pipeline {
 public:
  GLuint program = 0;
  bool compute = false;
  const SlotMap* slots = nullptr;
  VertexInput* vertex = nullptr;
  u32 strides[kMaxVertexBuffers] = {};
  GLenum mode = GL_TRIANGLES;
  RasterState raster;
  struct Push {
    GLint location = -1;
    u32 vec4_count = 0;
    char type = 'u';
  };
  Push push[3];
  u32 push_count = 0;
  bool push_ubo = false;
  u32 push_bytes = 0;
  GLint pointer_location = -1;
  u32 pointer_count = 0;
};

class GlTimestampPool final : public rhi::TimestampPool {
 public:
  u32 count = 0;
  GLuint buffer = 0;  // one u64 result per query; 0 = not written
  const u64* results = nullptr;
  std::vector<GLuint> queries;  // render thread
};

// ---- the recorded stream ---------------------------------------------------

enum class Op : u8 {
  kBeginPass,
  kEndPass,
  kPipeline,
  kBindBuffer,
  kBindTexture,
  kBindImage,
  kPush,
  kPointers,
  kVertexBuffers,
  kIndexBuffer,
  kViewport,
  kScissor,
  kBlendConstants,
  kDraw,
  kDrawIndexed,
  kDispatch,
  kClearAttachment,
  kCopyBuffer,
  kBufferToTexture,
  kTextureToBuffer,
  kCopyTexture,
  kBlit,
  kClearTexture,
  kClearDepthStencil,
  kFillBuffer,
  kUpdateBuffer,
  kBarrier,
  kResetTimestamps,
  kWriteTimestamp,
  kPushLabel,
  kPopLabel,
  kInsertLabel,
};

struct Cmd {
  Op op;
  u32 size;  // of the whole command, a multiple of 8
};

// How a clear value is read for one attachment.
enum class ClearKind : u8 { kFloat, kUint, kInt };

struct CmdBeginPass {
  static constexpr Op kOp = Op::kBeginPass;
  Cmd h;
  FboKey key;
  u8 color_count;
  u8 clear_mask;    // colours with LoadOp::kClear
  u8 discard_mask;  // colours with LoadOp::kDontCare
  u8 depth_clear;   // rhi::Aspect bits
  u8 depth_discard;
  ClearKind kinds[8];
  rhi::ClearColor clear[8];
  float clear_depth;
  u8 clear_stencil;
  i32 x, y;
  u32 width, height;
};

struct CmdEndPass {
  static constexpr Op kOp = Op::kEndPass;
  Cmd h;
  u8 discard_mask;   // colours with StoreOp::kDontCare
  i32 x, y;
  u32 width, height;
};

struct CmdPipeline {
  static constexpr Op kOp = Op::kPipeline;
  Cmd h;
  GlPipeline* pipeline;
};

struct CmdBindBuffer {
  static constexpr Op kOp = Op::kBindBuffer;
  Cmd h;
  GLenum target;
  u32 slot;
  GLuint name;
  u64 offset, size;
};

struct CmdBindTexture {
  static constexpr Op kOp = Op::kBindTexture;
  Cmd h;
  u32 unit;
  GLuint name, sampler;
};

struct CmdBindImage {
  static constexpr Op kOp = Op::kBindImage;
  Cmd h;
  u32 unit;
  GLuint name;
  GLint level, layer;
  GLboolean layered;
  GLenum format;
};

struct CmdPush {  // followed by `bytes` of data
  static constexpr Op kOp = Op::kPush;
  Cmd h;
  u32 bytes;
};

struct CmdPointers {  // followed by `count` uvec4 entries
  static constexpr Op kOp = Op::kPointers;
  Cmd h;
  u32 count;
};

struct CmdVertexBuffers {
  static constexpr Op kOp = Op::kVertexBuffers;
  Cmd h;
  VertexInput* input;
  u32 first, count;
  GLuint names[kMaxVertexBuffers];
  GLintptr offsets[kMaxVertexBuffers];
  GLsizei strides[kMaxVertexBuffers];
};

struct CmdIndexBuffer {
  static constexpr Op kOp = Op::kIndexBuffer;
  Cmd h;
  VertexInput* input;
  GLuint name;
};

struct CmdViewport {
  static constexpr Op kOp = Op::kViewport;
  Cmd h;
  float x, y, width, height, min_depth, max_depth;
};

struct CmdScissor {
  static constexpr Op kOp = Op::kScissor;
  Cmd h;
  i32 x, y;
  u32 width, height;
};

struct CmdBlendConstants {
  static constexpr Op kOp = Op::kBlendConstants;
  Cmd h;
  float rgba[4];
};

struct CmdDraw {
  static constexpr Op kOp = Op::kDraw;
  Cmd h;
  u32 vertex_count, instance_count, first_vertex, first_instance;
};

struct CmdDrawIndexed {
  static constexpr Op kOp = Op::kDrawIndexed;
  Cmd h;
  u32 index_count, instance_count;
  i32 vertex_offset;
  u32 first_instance;
  GLenum type;
  u64 offset;  // bytes into the index buffer
};

struct CmdDispatch {
  static constexpr Op kOp = Op::kDispatch;
  Cmd h;
  u32 x, y, z;
};

struct CmdClearAttachment {
  static constexpr Op kOp = Op::kClearAttachment;
  Cmd h;
  u32 attachment;  // ~0u: depth/stencil
  ClearKind kind;
  u8 aspect;
  rhi::ClearColor color;
  float depth;
  u8 stencil;
  i32 x, y;
  u32 width, height;
};

struct CmdCopyBuffer {
  static constexpr Op kOp = Op::kCopyBuffer;
  Cmd h;
  GLuint dst, src;
  u64 dst_offset, src_offset, bytes;
};

struct CmdBufferTexture {  // followed by `count` rhi::BufferTextureCopy
  Cmd h;
  GlTexture* texture;
  GlBuffer* buffer;
  u32 count;
};
struct CmdBufferToTexture : CmdBufferTexture {
  static constexpr Op kOp = Op::kBufferToTexture;
};
struct CmdTextureToBuffer : CmdBufferTexture {
  static constexpr Op kOp = Op::kTextureToBuffer;
};

struct CmdCopyTexture {
  static constexpr Op kOp = Op::kCopyTexture;
  Cmd h;
  GlTexture* dst;
  GlTexture* src;
  rhi::TextureRegion dst_region, src_region;
};

struct CmdBlit {
  static constexpr Op kOp = Op::kBlit;
  Cmd h;
  GlTexture* dst;
  GlTexture* src;
  rhi::TextureRegion dst_region, src_region;
  GLenum filter;
};

struct CmdClearTexture {
  static constexpr Op kOp = Op::kClearTexture;
  Cmd h;
  GlTexture* texture;
  rhi::TextureRange range;
  rhi::ClearColor color;
};

struct CmdClearDepthStencil {
  static constexpr Op kOp = Op::kClearDepthStencil;
  Cmd h;
  GlTexture* texture;
  rhi::TextureRange range;
  float depth;
  u8 stencil;
};

struct CmdFillBuffer {
  static constexpr Op kOp = Op::kFillBuffer;
  Cmd h;
  GLuint name;
  u64 offset, bytes;
  u32 value;
};

struct CmdUpdateBuffer {  // followed by `bytes` of data
  static constexpr Op kOp = Op::kUpdateBuffer;
  Cmd h;
  GLuint name;
  u64 offset, bytes;
};

struct CmdBarrier {
  static constexpr Op kOp = Op::kBarrier;
  Cmd h;
  GLbitfield bits;
};

struct CmdTimestamps {
  Cmd h;
  GlTimestampPool* pool;
  u32 first, count;
};
struct CmdResetTimestamps : CmdTimestamps {
  static constexpr Op kOp = Op::kResetTimestamps;
};
struct CmdWriteTimestamp : CmdTimestamps {
  static constexpr Op kOp = Op::kWriteTimestamp;
};

struct CmdLabel {  // followed by the NUL-terminated text
  Cmd h;
};
struct CmdPushLabel : CmdLabel {
  static constexpr Op kOp = Op::kPushLabel;
};
struct CmdPopLabel : CmdLabel {
  static constexpr Op kOp = Op::kPopLabel;
};
struct CmdInsertLabel : CmdLabel {
  static constexpr Op kOp = Op::kInsertLabel;
};

class CommandStream {
 public:
  void Clear() { bytes_.clear(); }
  const u8* begin() const { return bytes_.data(); }
  const u8* end() const { return bytes_.data() + bytes_.size(); }

  // Appends a zeroed T plus `extra` payload bytes. The pointer is valid
  // until the next Add.
  template <typename T>
  T* Add(size_t extra = 0) {
    const size_t size = (sizeof(T) + extra + 7) & ~size_t(7);
    const size_t at = bytes_.size();
    bytes_.resize(at + size);
    T* cmd = new (bytes_.data() + at) T();
    cmd->h.op = T::kOp;
    cmd->h.size = static_cast<u32>(size);
    return cmd;
  }

 private:
  std::vector<u8> bytes_;
};

// ---- recording -------------------------------------------------------------

class GlCommandList final : public rhi::CommandList {
 public:
  explicit GlCommandList(GlDevice& device);

  const CommandStream& stream() const { return stream_; }

  void Begin() override;
  void End() override {}
  void BeginRenderPass(const rhi::RenderPassDesc& pass) override;
  void EndRenderPass() override;
  void SetPipeline(rhi::Pipeline* pipeline) override;
  void SetBindGroup(u32 index,
                    rhi::BindGroup* group,
                    const u32* dynamic_offsets,
                    u32 num_offsets) override;
  void PushBindGroup(u32 index,
                     rhi::BindGroupLayout* layout,
                     const rhi::BindingWrite* writes,
                     u32 num_writes) override;
  void SetPushConstants(u32 offset, u32 bytes, const void* data) override;
  void SetVertexBuffers(u32 first,
                        u32 count,
                        rhi::Buffer* const* buffers,
                        const u64* offsets) override;
  void SetIndexBuffer(rhi::Buffer* buffer,
                      u64 offset,
                      rhi::IndexType type) override;
  void SetViewport(float x,
                   float y,
                   float width,
                   float height,
                   float min_depth,
                   float max_depth) override;
  void SetScissor(i32 x, i32 y, u32 width, u32 height) override;
  void SetBlendConstants(const float rgba[4]) override;
  void Draw(u32 vertex_count,
            u32 instance_count,
            u32 first_vertex,
            u32 first_instance) override;
  void DrawIndexed(u32 index_count,
                   u32 instance_count,
                   u32 first_index,
                   i32 vertex_offset,
                   u32 first_instance) override;
  void DrawMeshTasks(u32 x, u32 y, u32 z) override;
  void Dispatch(u32 x, u32 y, u32 z) override;
  void ClearAttachment(u32 attachment,
                       const rhi::ClearColor& color,
                       float depth,
                       u8 stencil,
                       u8 aspect,
                       i32 x,
                       i32 y,
                       u32 width,
                       u32 height) override;
  void CopyBuffer(rhi::Buffer* dst,
                  u64 dst_offset,
                  rhi::Buffer* src,
                  u64 src_offset,
                  u64 bytes) override;
  void CopyBufferToTexture(rhi::Texture* dst,
                           rhi::Buffer* src,
                           const rhi::BufferTextureCopy* regions,
                           u32 count) override;
  void CopyTextureToBuffer(rhi::Buffer* dst,
                           rhi::Texture* src,
                           const rhi::BufferTextureCopy* regions,
                           u32 count) override;
  void CopyTexture(rhi::Texture* dst,
                   const rhi::TextureRegion& dst_region,
                   rhi::Texture* src,
                   const rhi::TextureRegion& src_region) override;
  void BlitTexture(rhi::Texture* dst,
                   const rhi::TextureRegion& dst_region,
                   rhi::Texture* src,
                   const rhi::TextureRegion& src_region,
                   rhi::Filter filter) override;
  void ClearTexture(rhi::Texture* texture,
                    rhi::TextureState state,
                    const rhi::TextureRange& range,
                    const rhi::ClearColor& color) override;
  void ClearDepthStencil(rhi::Texture* texture,
                         rhi::TextureState state,
                         const rhi::TextureRange& range,
                         float depth,
                         u8 stencil) override;
  void FillBuffer(rhi::Buffer* buffer,
                  u64 offset,
                  u64 bytes,
                  u32 value) override;
  void UpdateBuffer(rhi::Buffer* buffer,
                    u64 offset,
                    u64 bytes,
                    const void* data) override;
  void Barrier(u32 src_access,
               u32 dst_access,
               const rhi::TextureBarrier* textures,
               u32 num_textures) override;
  void ResetTimestamps(rhi::TimestampPool* pool,
                       u32 first,
                       u32 count) override;
  void WriteTimestamp(rhi::TimestampPool* pool, u32 index) override;
  void PushLabel(const char* label) override;
  void PopLabel() override;
  void InsertLabel(const char* label) override;

 private:
  struct BoundGroup {
    const GlBindGroupLayout* layout = nullptr;
    const BoundResource* entries = nullptr;
    u32 offsets[kMaxDynamicOffsets] = {};
    bool dirty = false;
  };
  struct SlotState {
    GLuint name = ~0u;
    GLuint sampler = 0;
    u64 offset = 0, size = 0;
  };

  void FlushBindings();
  void FlushGroup(u32 index, const SlotMap::Group& map);
  void BindBuffer(SlotKind kind, u32 slot, const BoundResource& r, u64 offset);
  void FlushVertexInput(bool indexed);
  void Label(Op op, const char* text);

  GlDevice& device_;
  CommandStream stream_;
  GlPipeline* pipeline_ = nullptr;
  const SlotMap* flushed_slots_ = nullptr;
  const GlPipeline* flushed_push_ = nullptr;
  BoundGroup groups_[kMaxGroups];
  // PushBindGroup storage; reset at Begin, once the last replay retired.
  std::vector<std::vector<BoundResource>> push_groups_;
  u32 push_groups_used_ = 0;
  u8 push_data_[kMaxPushBytes] = {};
  bool push_dirty_ = true;
  GLuint vertex_buffers_[kMaxVertexBuffers] = {};
  u64 vertex_offsets_[kMaxVertexBuffers] = {};
  u32 vertex_dirty_ = 0;
  const VertexInput* flushed_input_ = nullptr;
  u32 flushed_strides_[kMaxVertexBuffers] = {};
  GLuint index_buffer_ = 0;
  u64 index_offset_ = 0;
  GLenum index_type_ = GL_UNSIGNED_SHORT;
  bool index_dirty_ = true;
  // What this list already bound, per GL slot, to drop repeats.
  std::vector<SlotState> slot_state_[kSlotKinds];
  u32 pointers_[kMaxPointers][4] = {};
  bool pointers_dirty_ = true;
  const GlPipeline* flushed_pointers_ = nullptr;
  const GlView* pass_colors_[8] = {};
  CmdEndPass pass_end_{};
};

// ---- replay (render thread) -------------------------------------------------

class Replayer {
 public:
  explicit Replayer(GlDevice& device) : device_(device) {}
  void Init();
  void Execute(const GlCommandList& list);
  // Drops cached GL objects that name `object`, before it is deleted.
  void Forget(rhi::Object* object);

 private:
  void BeginPass(const CmdBeginPass& c);
  void EndPass(const CmdEndPass& c);
  GLuint Framebuffer(const FboKey& key);
  void ApplyGraphics();
  void ApplyBlend(u32 i, const BlendState& b);
  void ApplyStencil(GLenum face, const StencilState& s, StencilState& have);
  void UseProgram(GLuint program);
  void BindVao(VertexInput* input);
  GLuint Vao(VertexInput* input);
  // Every colour, depth and stencil write enabled, as clears and blits need.
  void ResetMasks();
  // Scissor and masks set for a clear of a rectangle.
  void PrepareClear(i32 x, i32 y, u32 width, u32 height);
  void SetScissor(i32 x, i32 y, u32 width, u32 height);
  void SetScissorTest(bool enable);
  void ClearFramebuffer(GLuint fbo,
                        u32 index,
                        ClearKind kind,
                        const rhi::ClearColor& color);
  void ClearDepthStencilFbo(GLuint fbo, u8 aspect, float depth, u8 stencil);
  void Push(const CmdPush& c);
  void BufferToTexture(const CmdBufferTexture& c, bool upload);
  void CopyTexture(const CmdCopyTexture& c);
  void Blit(const CmdBlit& c);
  void ClearTexture(const CmdClearTexture& c);
  void ClearDepthStencil(const CmdClearDepthStencil& c);
  void AttachScratch(GLuint fbo,
                     GLenum attachment,
                     const GlTexture* t,
                     u32 mip,
                     u32 layer);
  void DetachScratch(GLuint fbo);

  GlDevice& device_;
  std::unordered_map<FboKey, GLuint, FboKeyHash> fbos_;
  GLuint scratch_read_ = 0, scratch_draw_ = 0;
  GLuint push_ubo_ = 0;
  GLuint pixel_pack_ = 0, pixel_unpack_ = 0;

  // The pipeline the next draw uses, and what GL currently has.
  GlPipeline* pipeline_ = nullptr;
  GLuint program_ = ~0u;
  GLuint vao_ = ~0u;
  GLuint fbo_ = 0;
  bool upper_left_ = false;
  GLenum front_face_ = 0;
  bool have_raster_ = false;
  RasterState raster_;
  i32 scissor_[4] = {};  // wanted by the recording
  i32 gl_scissor_[4] = {-1, -1, -1, -1};
  bool scissor_test_ = true;
};

// ---- device ------------------------------------------------------------------

class GlDevice final : public rhi::Device {
 public:
  ~GlDevice() override;
  bool Init(const OpenGLOptions& options);

  const rhi::Caps& caps() const override { return caps_; }
  rhi::Buffer* CreateBuffer(const rhi::BufferDesc& desc) override;
  rhi::Texture* CreateTexture(const rhi::TextureDesc& desc) override;
  rhi::TextureView* CreateView(rhi::Texture* texture,
                               const rhi::TextureViewDesc& desc) override;
  rhi::Sampler* CreateSampler(const rhi::SamplerDesc& desc) override;
  rhi::BindGroupLayout* CreateBindGroupLayout(
      const rhi::BindGroupLayoutDesc& desc) override;
  rhi::BindGroup* CreateBindGroup(const rhi::BindGroupDesc& desc) override;
  void UpdateBindGroup(rhi::BindGroup* group,
                       const rhi::BindingWrite* writes,
                       u32 count) override;
  rhi::PipelineLayout* CreatePipelineLayout(
      const rhi::PipelineLayoutDesc& desc) override;
  rhi::Pipeline* CreateGraphicsPipeline(
      const rhi::GraphicsPipelineDesc& desc) override;
  rhi::Pipeline* CreateComputePipeline(
      const rhi::ComputePipelineDesc& desc) override;
  rhi::TimestampPool* CreateTimestampPool(u32 count) override;
  rhi::CommandList* CreateCommandList() override;
  void Destroy(rhi::Object* object) override;
  void SetName(rhi::Object* object, const char* name) override;
  bool SupportsFormat(rhi::Format format, u32 usage) const override;
  u64 Submit(rhi::CommandList* const* lists, u32 count) override;
  bool IsComplete(u64 submission) override;
  bool Wait(u64 submission, u64 timeout_ns) override;
  void WaitIdle() override;
  u64 LastSubmission() const override { return submitted_; }
  bool ReadTimestamps(rhi::TimestampPool* pool,
                      u32 first,
                      u32 count,
                      u64* out) override;

  // Resolves one write into `out`, which is at the binding's position.
  void Resolve(const GlBindGroupLayout& layout,
               const rhi::BindingWrite& write,
               BoundResource* out) const;
  // Slot counts per SlotKind, for the lists' shadow state.
  u32 slot_limit(SlotKind kind) const {
    return slot_limits_[static_cast<u32>(kind)];
  }

 private:
  friend class Replayer;

  bool InitRenderThread();
  rhi::Pipeline* BuildPipeline(const rhi::PipelineLayoutDesc& layout,
                               const StageCode* stages,
                               u32 count,
                               GlPipeline* pipeline);
  const SlotMap* InternSlots(const rhi::PipelineLayoutDesc& layout,
                             const ProgramInterface& program);
  VertexInput* InternVertexInput(const rhi::GraphicsPipelineDesc& desc);
  void WaitLoop();

  bool debug_ = false;
  bool buffer_pointers_ = false;
  EglDevice egl_;
  GlWorker render_;
  GlWorker resource_;
  GlWorker compile_;
  std::unique_ptr<Replayer> replayer_;

  EGLContext waiter_context_ = EGL_NO_CONTEXT;
  std::thread waiter_;
  std::mutex fence_mutex_;
  std::condition_variable fence_cv_;
  std::deque<std::pair<u64, GLsync>> fences_;
  bool stop_waiter_ = false;
  std::mutex completed_mutex_;
  std::condition_variable completed_cv_;
  std::atomic<u64> completed_{0};
  std::atomic<bool> lost_{false};

  std::mutex submit_mutex_;
  std::atomic<u64> submitted_{0};

  rhi::Caps caps_;
  std::string device_name_;
  GlslFeatures glsl_;
  u32 slot_limits_[kSlotKinds] = {};
  u32 format_usage_[static_cast<size_t>(rhi::Format::kCount)] = {};

  std::mutex intern_mutex_;
  std::map<std::string, std::unique_ptr<SlotMap>> slot_maps_;
  std::map<std::string, std::unique_ptr<VertexInput>> vertex_inputs_;
};

}  // namespace gpu::opengl
