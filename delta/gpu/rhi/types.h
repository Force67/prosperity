#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The vocabulary of the device abstraction: formats, resource states and the
 * descriptions a Device creates objects from. Nothing here names a graphics
 * API; each backend maps these onto its own types.
 */

#include "base/arch.h"

#include <vector>

namespace gpu::rhi {

enum class Backend : u8 { kVulkan, kOpenGL, kD3D12 };

const char* BackendName(Backend backend);

// Channel order is memory order from the lowest byte, as in Vulkan's names.
// Packed formats name their fields from the most significant bit.
enum class Format : u16 {
  kUndefined,
  // 8-bit channels
  kR8Unorm, kR8Snorm, kR8Uint, kR8Sint, kR8Uscaled, kR8Sscaled,
  kRG8Unorm, kRG8Snorm, kRG8Uint, kRG8Sint, kRG8Uscaled, kRG8Sscaled,
  kRGBA8Unorm, kRGBA8Snorm, kRGBA8Uint, kRGBA8Sint, kRGBA8Uscaled,
  kRGBA8Sscaled, kRGBA8Srgb,
  kBGRA8Unorm, kBGRA8Srgb,
  // 16-bit channels
  kR16Unorm, kR16Snorm, kR16Uint, kR16Sint, kR16Uscaled, kR16Sscaled,
  kR16Float,
  kRG16Unorm, kRG16Snorm, kRG16Uint, kRG16Sint, kRG16Uscaled, kRG16Sscaled,
  kRG16Float,
  kRGBA16Unorm, kRGBA16Snorm, kRGBA16Uint, kRGBA16Sint, kRGBA16Uscaled,
  kRGBA16Sscaled, kRGBA16Float,
  // 32-bit channels
  kR32Uint, kR32Sint, kR32Float,
  kRG32Uint, kRG32Sint, kRG32Float,
  kRGB32Float,
  kRGBA32Uint, kRGBA32Sint, kRGBA32Float,
  // Packed
  kB10G11R11Float,     // R in the low bits (DXGI R11G11B10_FLOAT)
  kA2B10G10R10Unorm,   // R in the low bits (DXGI R10G10B10A2_UNORM)
  kA2R10G10B10Unorm,   // B in the low bits
  // Block compressed
  kBC1Unorm, kBC1Srgb, kBC2Unorm, kBC2Srgb, kBC3Unorm, kBC3Srgb,
  kBC4Unorm, kBC4Snorm, kBC5Unorm, kBC5Snorm, kBC6HUfloat, kBC6HSfloat,
  kBC7Unorm, kBC7Srgb,
  // Depth / stencil
  kD32Float, kD32FloatS8Uint, kS8Uint,
  kCount,
};

struct FormatInfo {
  u8 bytes = 0;         // per texel, or per 4x4 block when compressed
  u8 channels = 0;
  bool compressed = false;
  bool is_integer = false;  // Uint/Sint: never filtered, never blended
  bool is_depth = false;
  bool is_stencil = false;
  bool is_srgb = false;
};

const FormatInfo& GetFormatInfo(Format format);
const char* FormatName(Format format);

enum ShaderStage : u32 {
  kStageVertex = 1u << 0,
  kStageGeometry = 1u << 1,
  kStageFragment = 1u << 2,
  kStageCompute = 1u << 3,
  kStageMesh = 1u << 4,
  kStageAllGraphics = kStageVertex | kStageGeometry | kStageFragment |
                      kStageMesh,
};

// Where the memory of a buffer lives.
enum class MemoryKind : u8 {
  kDevice,    // device-local, no host mapping
  kUpload,    // host-visible, write-combined; for CPU writes the GPU reads
  // As kUpload, but in device-local memory where the host can map it.
  kUploadDevice,
  kReadback,  // host-visible, cached; for GPU writes the CPU reads
};

enum BufferUsage : u32 {
  kBufferVertex = 1u << 0,
  kBufferIndex = 1u << 1,
  kBufferUniform = 1u << 2,
  kBufferStorage = 1u << 3,
  kBufferIndirect = 1u << 4,
  kBufferCopySrc = 1u << 5,
  kBufferCopyDst = 1u << 6,
  // Readable in a shader through a 64-bit address (Caps::buffer_address).
  kBufferAddress = 1u << 7,
};

struct BufferDesc {
  u64 size = 0;
  u32 usage = 0;  // BufferUsage bits
  MemoryKind memory = MemoryKind::kDevice;
  // Back the buffer with this host allocation instead of new memory
  // (Caps::host_import). Must be aligned to Caps::host_import_alignment in
  // both address and size, and outlive the buffer.
  void* host_pointer = nullptr;
  const char* name = nullptr;
};

enum class TextureDim : u8 { k1D, k2D, k3D };

enum TextureUsage : u32 {
  kTextureSampled = 1u << 0,
  kTextureStorage = 1u << 1,
  kTextureColorTarget = 1u << 2,
  kTextureDepthTarget = 1u << 3,
  kTextureCopySrc = 1u << 4,
  kTextureCopyDst = 1u << 5,
  // Views may use any format of the same texel size, not only this one.
  kTextureMutableFormat = 1u << 6,
  // 2D: array layers may be viewed as cube faces.
  kTextureCubeCompatible = 1u << 7,
  // 3D: slices may be viewed (and rendered) as a 2D array.
  kTextureArrayCompatible = 1u << 8,
};

struct TextureDesc {
  TextureDim dim = TextureDim::k2D;
  Format format = Format::kUndefined;
  u32 width = 1, height = 1;
  u32 depth = 1;   // 3D only
  u32 layers = 1;  // 1D/2D only
  u32 mips = 1;
  u32 usage = 0;  // TextureUsage bits
  const char* name = nullptr;
};

enum class ViewDim : u8 {
  k1D, k1DArray, k2D, k2DArray, kCube, kCubeArray, k3D,
};

enum Aspect : u8 {
  kAspectColor = 1u << 0,
  kAspectDepth = 1u << 1,
  kAspectStencil = 1u << 2,
};

enum class Swizzle : u8 { kIdentity, kZero, kOne, kR, kG, kB, kA };

struct TextureViewDesc {
  ViewDim dim = ViewDim::k2D;
  Format format = Format::kUndefined;  // kUndefined = the texture's
  // One bit for a sampled or storage view; depth|stencil for an attachment.
  u8 aspect = kAspectColor;
  u32 base_mip = 0, mips = 1;
  u32 base_layer = 0, layers = 1;
  Swizzle swizzle[4] = {Swizzle::kIdentity, Swizzle::kIdentity,
                        Swizzle::kIdentity, Swizzle::kIdentity};
};

enum class Filter : u8 { kNearest, kLinear };
enum class AddressMode : u8 {
  kRepeat, kMirroredRepeat, kClampToEdge, kClampToBorder, kMirrorClampToEdge,
};
enum class CompareOp : u8 {
  kNever, kLess, kEqual, kLessEqual, kGreater, kNotEqual, kGreaterEqual,
  kAlways,
};
enum class BorderColor : u8 {
  kTransparentBlack, kOpaqueBlack, kOpaqueWhite,
};

struct SamplerDesc {
  Filter mag = Filter::kLinear, min = Filter::kLinear;
  Filter mip = Filter::kNearest;
  AddressMode address_u = AddressMode::kClampToEdge;
  AddressMode address_v = AddressMode::kClampToEdge;
  AddressMode address_w = AddressMode::kClampToEdge;
  float lod_bias = 0.0f;
  float min_lod = 0.0f, max_lod = 1000.0f;
  float max_anisotropy = 1.0f;  // > 1 enables anisotropic filtering
  bool compare_enable = false;
  CompareOp compare = CompareOp::kNever;
  BorderColor border = BorderColor::kTransparentBlack;
};

// Resource states. A texture is in exactly one per subresource at any point
// of the recording; the caller names both sides of each transition.
enum class TextureState : u8 {
  kUndefined,     // contents may be discarded
  kGeneral,       // storage image reads/writes, or any mixed use
  kColorTarget,
  kDepthTarget,   // depth/stencil writable
  kDepthRead,     // depth/stencil bound read-only, and/or sampled
  kShaderRead,
  kCopySrc,
  kCopyDst,
};

// Memory access classes for buffer and global barriers.
enum Access : u32 {
  kAccessNone = 0,
  kAccessVertexRead = 1u << 0,
  kAccessIndexRead = 1u << 1,
  kAccessUniformRead = 1u << 2,
  kAccessIndirectRead = 1u << 3,
  kAccessShaderRead = 1u << 4,   // any stage
  kAccessShaderWrite = 1u << 5,  // any stage
  kAccessCopyRead = 1u << 6,
  kAccessCopyWrite = 1u << 7,
  kAccessHostRead = 1u << 8,
  kAccessHostWrite = 1u << 9,
  kAccessColorWrite = 1u << 10,
  kAccessDepthWrite = 1u << 11,
  // As kAccessShaderRead/Write, scoped to compute dispatches: a barrier on
  // these does not wait for (or hold back) graphics shading.
  kAccessComputeRead = 1u << 12,
  kAccessComputeWrite = 1u << 13,
  kAccessAllRead = kAccessVertexRead | kAccessIndexRead | kAccessUniformRead |
                   kAccessIndirectRead | kAccessShaderRead | kAccessCopyRead |
                   kAccessHostRead | kAccessComputeRead,
  kAccessAllWrite = kAccessShaderWrite | kAccessCopyWrite | kAccessHostWrite |
                    kAccessColorWrite | kAccessDepthWrite | kAccessComputeWrite,
};

// Descriptor bindings. Each backend turns a (set, binding) pair into its own
// slot; shaders are SPIR-V with the same set/binding decorations.
enum class BindingType : u8 {
  kSampledTexture,  // combined texture + sampler
  kStorageTexture,
  kUniformBuffer,
  kStorageBuffer,
  // The offset is supplied at SetBindGroup time rather than when the group is
  // written, so one group serves every window of a ring buffer.
  kUniformBufferDynamic,
  kStorageBufferDynamic,
};

struct BindingLayout {
  u32 binding = 0;
  BindingType type = BindingType::kSampledTexture;
  u32 stages = 0;  // ShaderStage bits
  // Storage buffers the shader never writes: lets D3D12 bind an SRV.
  bool read_only = false;
};

struct BindGroupLayoutDesc {
  std::vector<BindingLayout> bindings;
  // Bound only through CommandList::PushBindGroup, never as a BindGroup.
  bool push = false;
};

class Buffer;
class Texture;
class TextureView;
class Sampler;
class BindGroupLayout;
class PipelineLayout;

// One written binding of a group. Unused members stay null.
struct BindingWrite {
  u32 binding = 0;
  TextureView* view = nullptr;
  Sampler* sampler = nullptr;  // kSampledTexture
  TextureState view_state = TextureState::kShaderRead;
  Buffer* buffer = nullptr;
  u64 offset = 0;  // for dynamic bindings: added to the bind-time offset
  u64 range = 0;
};

struct BindGroupDesc {
  BindGroupLayout* layout = nullptr;
  std::vector<BindingWrite> writes;
};

struct PipelineLayoutDesc {
  // Group index = position; a null entry is an empty group.
  std::vector<BindGroupLayout*> groups;
  u32 push_constant_bytes = 0;
  u32 push_constant_stages = 0;
};

enum class Topology : u8 {
  kPointList, kLineList, kLineStrip, kTriangleList, kTriangleStrip,
  kTriangleFan, kLineListAdjacency, kTriangleListAdjacency,
};
enum class CullMode : u8 { kNone, kFront, kBack, kFrontAndBack };
enum class BlendFactor : u8 {
  kZero, kOne, kSrcColor, kOneMinusSrcColor, kDstColor, kOneMinusDstColor,
  kSrcAlpha, kOneMinusSrcAlpha, kDstAlpha, kOneMinusDstAlpha,
  kConstantColor, kOneMinusConstantColor, kConstantAlpha,
  kOneMinusConstantAlpha, kSrcAlphaSaturate, kSrc1Color, kOneMinusSrc1Color,
  kSrc1Alpha, kOneMinusSrc1Alpha,
};
enum class BlendOp : u8 { kAdd, kSubtract, kReverseSubtract, kMin, kMax };
enum class StencilOp : u8 {
  kKeep, kZero, kReplace, kIncrementClamp, kDecrementClamp, kInvert,
  kIncrementWrap, kDecrementWrap,
};

struct BlendAttachment {
  bool enable = false;
  BlendFactor src_color = BlendFactor::kOne, dst_color = BlendFactor::kZero;
  BlendOp color_op = BlendOp::kAdd;
  BlendFactor src_alpha = BlendFactor::kOne, dst_alpha = BlendFactor::kZero;
  BlendOp alpha_op = BlendOp::kAdd;
  u8 write_mask = 0xF;  // bit 0 = R
};

struct StencilFace {
  StencilOp fail = StencilOp::kKeep;
  StencilOp pass = StencilOp::kKeep;
  StencilOp depth_fail = StencilOp::kKeep;
  CompareOp compare = CompareOp::kAlways;
  // D3D12 and OpenGL take one compare/write mask for both faces: the front
  // face's is used there.
  u8 compare_mask = 0xFF, write_mask = 0xFF, reference = 0;
};

struct VertexBufferLayout {
  u32 stride = 0;
  bool per_instance = false;
};

struct VertexAttribute {
  u32 location = 0;
  u32 buffer = 0;  // index into GraphicsPipelineDesc::vertex_buffers
  Format format = Format::kUndefined;
  u32 offset = 0;
};

// Shaders are SPIR-V. A backend that does not consume SPIR-V lowers it when
// the pipeline is created, against the pipeline's layout.
struct ShaderCode {
  const u32* words = nullptr;
  size_t count = 0;  // in words
  bool empty() const { return count == 0; }  // NOLINT: accessor
};

inline ShaderCode Code(const std::vector<u32>& spirv) {
  return {spirv.data(), spirv.size()};
}

struct GraphicsPipelineDesc {
  PipelineLayout* layout = nullptr;
  ShaderCode vertex;  // or mesh
  ShaderCode geometry;
  ShaderCode fragment;
  bool mesh = false;  // `vertex` is a mesh shader; no vertex input
  std::vector<VertexBufferLayout> vertex_buffers;
  std::vector<VertexAttribute> vertex_attributes;
  Topology topology = Topology::kTriangleList;
  bool primitive_restart = false;
  CullMode cull = CullMode::kNone;
  bool front_ccw = true;
  bool depth_clamp = false;
  bool depth_test = false;
  bool depth_write = false;
  CompareOp depth_compare = CompareOp::kAlways;
  bool stencil_test = false;
  StencilFace stencil_front, stencil_back;
  u32 color_count = 0;
  Format color_formats[8] = {};
  BlendAttachment blend[8];
  Format depth_format = Format::kUndefined;
  Format stencil_format = Format::kUndefined;
  const char* name = nullptr;
};

struct ComputePipelineDesc {
  PipelineLayout* layout = nullptr;
  ShaderCode code;
  bool dispatch_base = false;  // dispatched with CommandList::DispatchBase
  const char* name = nullptr;
};

enum class LoadOp : u8 { kLoad, kClear, kDontCare };
enum class StoreOp : u8 { kStore, kDontCare };

union ClearColor {
  float f[4];
  u32 u[4];
  i32 i[4];
};

struct ColorAttachment {
  TextureView* view = nullptr;
  LoadOp load = LoadOp::kLoad;
  StoreOp store = StoreOp::kStore;
  ClearColor clear{};
};

struct DepthAttachment {
  TextureView* view = nullptr;  // null = no depth/stencil
  LoadOp depth_load = LoadOp::kLoad;
  LoadOp stencil_load = LoadOp::kLoad;
  float clear_depth = 1.0f;
  u8 clear_stencil = 0;
  bool depth = true;    // the view's depth plane is attached
  bool stencil = false;  // the view's stencil plane is attached
  bool read_only = false;          // depth plane
  bool stencil_read_only = false;
};

// Attachments must already be in kColorTarget / kDepthTarget (or kDepthRead
// when read_only); a pass does not transition them.
struct RenderPassDesc {
  u32 color_count = 0;
  ColorAttachment colors[8];
  DepthAttachment depth;
  i32 x = 0, y = 0;
  u32 width = 0, height = 0;
  u32 layers = 1;
};

enum class IndexType : u8 { kUint16, kUint32 };

struct TextureRegion {
  u8 aspect = kAspectColor;
  u32 mip = 0;
  u32 base_layer = 0, layers = 1;
  i32 x = 0, y = 0, z = 0;
  u32 width = 1, height = 1, depth = 1;
};

// A buffer <-> texture copy. Row length and image height are in texels (0 =
// tightly packed), as in Vulkan.
struct BufferTextureCopy {
  u64 buffer_offset = 0;
  u32 row_length = 0;
  u32 image_height = 0;
  TextureRegion region;
};

struct TextureRange {
  u8 aspect = kAspectColor;
  u32 base_mip = 0, mips = 1;
  u32 base_layer = 0, layers = 1;
};

struct TextureBarrier {
  Texture* texture = nullptr;
  TextureState before = TextureState::kUndefined;
  TextureState after = TextureState::kShaderRead;
  TextureRange range;
};

struct Caps {
  Backend backend = Backend::kVulkan;
  const char* device_name = "";
  bool geometry_shader = false;
  bool mesh_shader = false;
  struct MeshLimits {
    u32 max_threads = 0;  // invocations per workgroup
    u32 max_threads_x = 0;
    u32 max_shared_bytes = 0;
    u32 max_output_vertices = 0;
    u32 max_output_primitives = 0;
    u32 max_groups[2] = {};  // per dispatch dimension
    u32 max_total_groups = 0;
  } mesh;
  bool fragment_barycentric = false;
  bool independent_blend = false;
  bool sampler_anisotropy = false;
  bool sampler_mirror_clamp = false;
  bool storage_image_write_without_format = false;
  bool buffer_address = false;  // kBufferAddress + SPIR-V PhysicalStorageBuffer
  bool host_import = false;
  u64 host_import_alignment = 0;
  bool texture_blit = true;
  bool dispatch_base = false;
  // One memory type is both device-local and host-cached: a kReadback buffer
  // is as fast for the GPU as kDevice.
  bool unified_memory = false;
  bool debug_labels = false;  // labels and names reach a tool
  bool timestamps = false;
  double timestamp_period_ns = 1.0;
  u32 timestamp_bits = 64;  // valid bits: tick differences wrap at this width
  u32 subgroup_size = 32;
  u32 max_dynamic_uniform_buffers = 8;
  u32 max_dynamic_storage_buffers = 4;
  u32 max_storage_buffers_per_stage = 8;
  u32 max_push_constant_bytes = 128;
  u64 max_storage_buffer_range = 128ull << 20;
  u32 uniform_offset_alignment = 256;
  u32 storage_offset_alignment = 256;
  u32 max_texture_size = 16384;
  u32 max_texture_size_3d = 2048;
  u32 max_compute_resources = 64;
};

}  // namespace gpu::rhi
