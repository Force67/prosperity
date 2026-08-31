#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The neutral vocabulary the device abstraction is spelled in: formats,
 * resource states, and the descriptions a resource or pipeline is created
 * from. Nothing here names a graphics API type, and every value in it can be
 * served by every backend -- a format or state that only one API has does not
 * belong in this file.
 *
 * This is deliberately narrower than either API. Guest encodings (GCN dfmt /
 * nfmt, GNM blend words, T# tiling) are decoded ABOVE this seam and arrive
 * here already neutral; see rhi/device.h for where that line runs.
 */

#include "base/arch.h"

namespace gpu::rhi {

// Colour targets a pipeline or render pass may bind, matching CB_COLOR0..7.
inline constexpr u32 kMaxColorTargets = 8;
// Vertex buffer bindings and attributes a pipeline may declare.
inline constexpr u32 kMaxVertexBuffers = 8;
inline constexpr u32 kMaxVertexAttributes = 16;
// Bind groups a pipeline may declare (the recompiler plans four: textures,
// constant buffers, raw buffers, scratch).
inline constexpr u32 kMaxBindGroups = 4;
// Bindings within one bind group.
inline constexpr u32 kMaxBindGroupEntries = 32;
// The HLSL register space push constants live in on D3D12, kept clear of the
// bind group spaces. A shader declares them as
// `cbuffer Push : register(b0, space8)`, or with [[vk::push_constant]].
inline constexpr u32 kPushConstantSpace = 8;

// Texel formats. Both backends must serve every entry: a format one API lacks
// is expanded or rejected above this seam, not added here.
enum class Format : u8 {
  Unknown,
  R8Unorm,
  R8Uint,
  RG8Unorm,
  RGBA8Unorm,
  RGBA8Srgb,
  BGRA8Unorm,
  BGRA8Srgb,
  R16Uint,
  R16Float,
  RG16Float,
  RGBA16Float,
  RGBA16Unorm,
  R32Uint,
  R32Float,
  RG32Float,
  // Vertex attributes only: neither API accepts a three-channel 32-bit format
  // as a render target, and D3D12 does not sample one.
  RGB32Float,
  RGBA32Float,
  RGB10A2Unorm,
  RG11B10Float,
  D16Unorm,
  D32Float,
  D32FloatS8Uint,
  BC1Unorm,
  BC2Unorm,
  BC3Unorm,
  BC4Unorm,
  BC5Unorm,
  BC7Unorm,
  kCount,
};

// Bytes one texel occupies (one 4x4 block, for a compressed format).
u32 FormatTexelBytes(Format f);
// Texels along one edge of an addressable block: 1 uncompressed, 4 for BC.
u32 FormatBlockDim(Format f);
bool FormatIsDepth(Format f);
bool FormatHasStencil(Format f);
bool FormatIsCompressed(Format f);
// Static name, for logs and test failures. Never null.
const char* FormatName(Format f);
// Bytes one row of `width` texels occupies, rounded up to whole blocks.
u64 FormatRowBytes(Format f, u32 width);

// What a resource is being used for. The backend turns this into an image
// layout plus access mask, or a D3D12 resource state.
//
// Undefined means "the contents may be discarded": it is the state a texture
// is created in, and the only state a transition may not target.
enum class ResourceState : u8 {
  Undefined,
  Common,
  RenderTarget,
  DepthWrite,
  DepthRead,
  ShaderRead,
  UnorderedAccess,
  CopySrc,
  CopyDst,
  VertexInput,
  IndexInput,
  IndirectArgument,
};

enum class LoadOp : u8 { Load, Clear, Discard };
enum class StoreOp : u8 { Store, Discard };

enum class CompareOp : u8 {
  Never,
  Less,
  Equal,
  LessEqual,
  Greater,
  NotEqual,
  GreaterEqual,
  Always,
};

enum class BlendFactor : u8 {
  Zero,
  One,
  SrcColor,
  OneMinusSrcColor,
  DstColor,
  OneMinusDstColor,
  SrcAlpha,
  OneMinusSrcAlpha,
  DstAlpha,
  OneMinusDstAlpha,
  ConstantColor,
  OneMinusConstantColor,
  ConstantAlpha,
  OneMinusConstantAlpha,
  SrcAlphaSaturate,
};

enum class BlendOp : u8 { Add, Subtract, ReverseSubtract, Min, Max };

enum class CullMode : u8 { None, Front, Back };

enum class PrimitiveTopology : u8 {
  PointList,
  LineList,
  LineStrip,
  TriangleList,
  TriangleStrip,
};

enum class IndexType : u8 { U16, U32 };

enum class FilterMode : u8 { Nearest, Linear };
enum class AddressMode : u8 {
  Repeat,
  MirrorRepeat,
  ClampToEdge,
  ClampToBorder,
};
enum class BorderColor : u8 { TransparentBlack, OpaqueBlack, OpaqueWhite };

// Where a buffer's memory lives. Upload is host-writable and GPU-readable;
// Readback is host-readable after the GPU has written it. Neither is fast for
// the other direction.
enum class MemoryKind : u8 { Device, Upload, Readback };

// Bit flags. Plain enums so the | of two of them stays a u32.
enum BufferUsage : u32 {
  kBufferVertex = 1u << 0,
  kBufferIndex = 1u << 1,
  kBufferConstant = 1u << 2,
  kBufferStorage = 1u << 3,
  kBufferCopySrc = 1u << 4,
  kBufferCopyDst = 1u << 5,
  kBufferIndirect = 1u << 6,
};

enum TextureUsage : u32 {
  kTextureSampled = 1u << 0,
  kTextureRenderTarget = 1u << 1,
  kTextureDepthStencil = 1u << 2,
  kTextureStorage = 1u << 3,
  kTextureCopySrc = 1u << 4,
  kTextureCopyDst = 1u << 5,
};

enum ShaderStage : u32 {
  kStageVertex = 1u << 0,
  kStagePixel = 1u << 1,
  kStageCompute = 1u << 2,
};

enum ColorWrite : u8 {
  kWriteRed = 1u << 0,
  kWriteGreen = 1u << 1,
  kWriteBlue = 1u << 2,
  kWriteAlpha = 1u << 3,
  kWriteAll = 0xF,
};

struct BufferDesc {
  u64 size = 0;
  u32 usage = 0;  // BufferUsage bits
  MemoryKind memory = MemoryKind::Device;
  const char* name = nullptr;  // static string, for debug labels
};

struct TextureDesc {
  u32 width = 1;
  u32 height = 1;
  u32 depth = 1;  // > 1 makes it a 3D texture
  u32 mip_levels = 1;
  u32 array_layers = 1;
  Format format = Format::Unknown;
  u32 usage = 0;  // TextureUsage bits
  const char* name = nullptr;
};

struct SamplerDesc {
  FilterMode min_filter = FilterMode::Linear;
  FilterMode mag_filter = FilterMode::Linear;
  FilterMode mip_filter = FilterMode::Linear;
  AddressMode address_u = AddressMode::Repeat;
  AddressMode address_v = AddressMode::Repeat;
  AddressMode address_w = AddressMode::Repeat;
  float min_lod = 0.0f;
  float max_lod = 1000.0f;
  float lod_bias = 0.0f;
  u32 max_anisotropy = 1;
  bool compare_enable = false;
  CompareOp compare = CompareOp::Never;
  BorderColor border = BorderColor::TransparentBlack;
};

// Shader bytecode. The RHI does not compile shaders: the caller hands each
// backend the form it consumes, because the two are produced by different
// parts of the emulator (the recompiler emits SPIR-V; the D3D12 path takes
// HLSL, or DXBC that was compiled from it earlier).
struct ShaderDesc {
  u32 stage = 0;  // exactly one ShaderStage bit
  // Vulkan input.
  const u32* spirv = nullptr;
  u64 spirv_bytes = 0;
  // D3D12 input: precompiled DXBC, else HLSL source compiled on creation.
  const void* dxbc = nullptr;
  u64 dxbc_bytes = 0;
  const char* hlsl = nullptr;
  const char* entry = "main";
  // Shader model to compile `hlsl` against, e.g. "vs_5_1". Ignored when dxbc
  // is supplied.
  const char* hlsl_target = nullptr;
  const char* name = nullptr;
};

enum class BindingType : u8 {
  UniformBuffer,
  StorageBuffer,
  SampledTexture,
  StorageTexture,
  Sampler,
};

// One binding in a bind group.
//
// `binding` is unique within the group ACROSS types. That is what lets one
// number serve as both a Vulkan binding and an HLSL register slot: a group
// holding a constant buffer, a texture and a sampler numbers them 0, 1, 2 and
// the HLSL declares b0, t1, s2 in the group's register space.
struct BindGroupEntry {
  u32 binding = 0;
  BindingType type = BindingType::UniformBuffer;
  u32 stages = 0;  // ShaderStage bits
  // The offset of a Uniform/StorageBuffer binding is supplied per draw rather
  // than baked into the group. Its size stays the one the group was made with.
  bool dynamic = false;
};

struct BindGroupLayoutDesc {
  BindGroupEntry entries[kMaxBindGroupEntries];
  u32 num_entries = 0;
  // HLSL register space this group's bindings live in on D3D12, and the
  // descriptor set index on Vulkan. Groups are bound by this index.
  u32 index = 0;
  const char* name = nullptr;
};

// One binding's contents. Only the fields its BindingType uses are read.
struct BindGroupBinding {
  u32 binding = 0;
  // Uniform/StorageBuffer. size 0 means "to the end of the buffer". For a
  // dynamic binding, `offset` is the base the per-draw offset adds to.
  struct Buffer* buffer = nullptr;
  u64 offset = 0;
  u64 size = 0;
  // Sampled/StorageTexture. Unknown view_format means the texture's own.
  struct Texture* texture = nullptr;
  Format view_format = Format::Unknown;
  u32 base_mip = 0;
  u32 mips = 1;
  u32 base_layer = 0;
  u32 layers = 1;
  struct Sampler* sampler = nullptr;
};

struct BindGroupDesc {
  const struct BindGroupLayout* layout = nullptr;
  BindGroupBinding bindings[kMaxBindGroupEntries];
  u32 num_bindings = 0;
  const char* name = nullptr;
};

struct VertexAttribute {
  u32 location = 0;
  u32 buffer = 0;  // index into GraphicsPipelineDesc::vertex_buffers
  u32 offset = 0;  // bytes into the vertex record
  Format format = Format::Unknown;
};

struct VertexBufferLayout {
  u32 stride = 0;
  bool per_instance = false;
};

struct BlendState {
  bool enable = false;
  BlendFactor src_color = BlendFactor::One;
  BlendFactor dst_color = BlendFactor::Zero;
  BlendOp color_op = BlendOp::Add;
  BlendFactor src_alpha = BlendFactor::One;
  BlendFactor dst_alpha = BlendFactor::Zero;
  BlendOp alpha_op = BlendOp::Add;
  u8 write_mask = kWriteAll;
};

struct DepthState {
  bool test_enable = false;
  bool write_enable = false;
  CompareOp compare = CompareOp::Always;
};

struct RasterState {
  CullMode cull = CullMode::None;
  bool front_ccw = true;
  bool depth_clip = true;
};

struct GraphicsPipelineDesc {
  const struct ShaderModule* vertex = nullptr;
  const struct ShaderModule* pixel = nullptr;

  VertexBufferLayout vertex_buffers[kMaxVertexBuffers];
  u32 num_vertex_buffers = 0;
  VertexAttribute attributes[kMaxVertexAttributes];
  u32 num_attributes = 0;

  PrimitiveTopology topology = PrimitiveTopology::TriangleList;
  RasterState raster;
  DepthState depth;

  BlendState blend[kMaxColorTargets];
  Format color_formats[kMaxColorTargets] = {};
  u32 num_color_targets = 0;
  Format depth_format = Format::Unknown;

  // Bind group layouts, in group index order. A null entry leaves that index
  // unbound (a pipeline may declare group 2 without declaring group 1).
  const struct BindGroupLayout* bind_groups[kMaxBindGroups] = {};
  u32 num_bind_groups = 0;

  // Bytes of push constants (Vulkan) / root constants (D3D12) the pipeline
  // takes. Visible to every stage.
  u32 push_constant_bytes = 0;

  const char* name = nullptr;
};

struct ComputePipelineDesc {
  const struct ShaderModule* compute = nullptr;
  const struct BindGroupLayout* bind_groups[kMaxBindGroups] = {};
  u32 num_bind_groups = 0;
  u32 push_constant_bytes = 0;
  const char* name = nullptr;
};

struct ColorAttachment {
  struct Texture* texture = nullptr;
  u32 mip = 0;
  u32 layer = 0;
  LoadOp load = LoadOp::Load;
  StoreOp store = StoreOp::Store;
  float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct DepthAttachment {
  struct Texture* texture = nullptr;
  LoadOp load = LoadOp::Load;
  StoreOp store = StoreOp::Store;
  float clear_depth = 1.0f;
  u8 clear_stencil = 0;
  // Bound for reading only, so the same image may be sampled in the pass.
  bool read_only = false;
};

struct RenderPassDesc {
  ColorAttachment color[kMaxColorTargets];
  u32 num_color = 0;
  DepthAttachment depth;
  bool has_depth = false;
  // The region the pass renders into. Attachments may be larger.
  u32 width = 0;
  u32 height = 0;
  const char* name = nullptr;
};

// A rectangle of one texture subresource, for copies.
struct TextureRegion {
  u32 mip = 0;
  u32 layer = 0;
  u32 x = 0, y = 0, z = 0;
  u32 width = 0, height = 0, depth = 1;
};

// What the device can do. Read it instead of assuming: the two backends do not
// agree on all of these, and the row-pitch alignment in particular is a D3D12
// hard requirement that Vulkan does not have.
struct DeviceCaps {
  const char* adapter_name = "";
  u32 max_texture_2d = 0;
  u32 max_bound_textures = 0;
  // Alignment a dynamic uniform buffer offset must satisfy.
  u32 constant_buffer_offset_alignment = 256;
  // Alignment a dynamic storage buffer offset must satisfy.
  u32 storage_buffer_offset_alignment = 16;
  // Alignment CopyBufferToTexture / CopyTextureToBuffer row pitches must
  // satisfy (256 on D3D12, 1 on Vulkan).
  u32 texture_row_pitch_alignment = 1;
  // Alignment a buffer offset in a texture copy must satisfy (512 on D3D12).
  u32 texture_copy_offset_alignment = 1;
  bool independent_blend = false;
  bool anisotropic_filtering = false;
  // Guest pages can be imported as buffer memory rather than copied.
  bool host_import = false;
  float timestamp_period_ns = 0.0f;
};

}  // namespace gpu::rhi
