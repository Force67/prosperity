/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#include <base/logging.h>

#include "gpu/opengl/gl_rhi_internal.h"

namespace gpu::opengl {

namespace {

void GLAPIENTRY OnDebugMessage(GLenum /*source*/,
                               GLenum type,
                               GLuint /*id*/,
                               GLenum severity,
                               GLsizei /*length*/,
                               const GLchar* message,
                               const void* /*user*/) {
  if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
    return;
  BASE_LOGI("gpugl", "{}: {}",
            type == GL_DEBUG_TYPE_ERROR ? "error" : "debug", message);
}

std::function<void()> ContextInit(bool debug) {
  if (!debug)
    return {};
  return [] {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(OnDebugMessage, nullptr);
  };
}

GLint GetInt(GLenum pname) {
  GLint v = 0;
  glGetIntegerv(pname, &v);
  return v;
}

GLenum ToGlCompare(rhi::CompareOp op) {
  static constexpr GLenum kOps[] = {GL_NEVER,   GL_LESS,     GL_EQUAL,
                                    GL_LEQUAL,  GL_GREATER,  GL_NOTEQUAL,
                                    GL_GEQUAL,  GL_ALWAYS};
  return kOps[static_cast<u32>(op) & 7];
}

GLenum ToGlStencilOp(rhi::StencilOp op) {
  static constexpr GLenum kOps[] = {GL_KEEP,   GL_ZERO,      GL_REPLACE,
                                    GL_INCR,   GL_DECR,      GL_INVERT,
                                    GL_INCR_WRAP, GL_DECR_WRAP};
  return kOps[static_cast<u32>(op) & 7];
}

GLenum ToGlBlendFactor(rhi::BlendFactor f) {
  static constexpr GLenum kFactors[] = {
      GL_ZERO,
      GL_ONE,
      GL_SRC_COLOR,
      GL_ONE_MINUS_SRC_COLOR,
      GL_DST_COLOR,
      GL_ONE_MINUS_DST_COLOR,
      GL_SRC_ALPHA,
      GL_ONE_MINUS_SRC_ALPHA,
      GL_DST_ALPHA,
      GL_ONE_MINUS_DST_ALPHA,
      GL_CONSTANT_COLOR,
      GL_ONE_MINUS_CONSTANT_COLOR,
      GL_CONSTANT_ALPHA,
      GL_ONE_MINUS_CONSTANT_ALPHA,
      GL_SRC_ALPHA_SATURATE,
      GL_SRC1_COLOR,
      GL_ONE_MINUS_SRC1_COLOR,
      GL_SRC1_ALPHA,
      GL_ONE_MINUS_SRC1_ALPHA,
  };
  const u32 i = static_cast<u32>(f);
  return i < sizeof(kFactors) / sizeof(kFactors[0]) ? kFactors[i] : GL_ONE;
}

GLenum ToGlBlendOp(rhi::BlendOp op) {
  static constexpr GLenum kOps[] = {GL_FUNC_ADD, GL_FUNC_SUBTRACT,
                                    GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX};
  const u32 i = static_cast<u32>(op);
  return i < 5 ? kOps[i] : GL_FUNC_ADD;
}

GLenum ToGlMode(rhi::Topology t) {
  switch (t) {
    case rhi::Topology::kPointList:
      return GL_POINTS;
    case rhi::Topology::kLineList:
      return GL_LINES;
    case rhi::Topology::kLineStrip:
      return GL_LINE_STRIP;
    case rhi::Topology::kTriangleList:
      return GL_TRIANGLES;
    case rhi::Topology::kTriangleStrip:
      return GL_TRIANGLE_STRIP;
    case rhi::Topology::kTriangleFan:
      return GL_TRIANGLE_FAN;
    case rhi::Topology::kLineListAdjacency:
      return GL_LINES_ADJACENCY;
    case rhi::Topology::kTriangleListAdjacency:
      return GL_TRIANGLES_ADJACENCY;
  }
  return GL_TRIANGLES;
}

GLenum ToGlAddress(rhi::AddressMode m) {
  switch (m) {
    case rhi::AddressMode::kRepeat:
      return GL_REPEAT;
    case rhi::AddressMode::kMirroredRepeat:
      return GL_MIRRORED_REPEAT;
    case rhi::AddressMode::kClampToEdge:
      return GL_CLAMP_TO_EDGE;
    case rhi::AddressMode::kClampToBorder:
      return GL_CLAMP_TO_BORDER;
    case rhi::AddressMode::kMirrorClampToEdge:
      return GL_MIRROR_CLAMP_TO_EDGE;
  }
  return GL_CLAMP_TO_EDGE;
}

GLint ToGlSwizzle(rhi::Swizzle s, GLint identity) {
  switch (s) {
    case rhi::Swizzle::kIdentity:
      return identity;
    case rhi::Swizzle::kZero:
      return GL_ZERO;
    case rhi::Swizzle::kOne:
      return GL_ONE;
    case rhi::Swizzle::kR:
      return GL_RED;
    case rhi::Swizzle::kG:
      return GL_GREEN;
    case rhi::Swizzle::kB:
      return GL_BLUE;
    case rhi::Swizzle::kA:
      return GL_ALPHA;
  }
  return identity;
}

GLenum ViewTarget(rhi::ViewDim dim) {
  switch (dim) {
    case rhi::ViewDim::k1D:
      return GL_TEXTURE_1D;
    case rhi::ViewDim::k1DArray:
      return GL_TEXTURE_1D_ARRAY;
    case rhi::ViewDim::k2D:
      return GL_TEXTURE_2D;
    case rhi::ViewDim::k2DArray:
      return GL_TEXTURE_2D_ARRAY;
    case rhi::ViewDim::kCube:
      return GL_TEXTURE_CUBE_MAP;
    case rhi::ViewDim::kCubeArray:
      return GL_TEXTURE_CUBE_MAP_ARRAY;
    case rhi::ViewDim::k3D:
      return GL_TEXTURE_3D;
  }
  return GL_TEXTURE_2D;
}

StencilState ToGlStencil(const rhi::StencilFace& f) {
  StencilState s;
  s.func = ToGlCompare(f.compare);
  s.ref = f.reference;
  s.compare_mask = f.compare_mask;
  s.write_mask = f.write_mask;
  s.fail = ToGlStencilOp(f.fail);
  s.depth_fail = ToGlStencilOp(f.depth_fail);
  s.pass = ToGlStencilOp(f.pass);
  return s;
}

std::string InfoLog(GLuint object, bool program) {
  GLint n = 0;
  if (program)
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &n);
  else
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &n);
  std::string log(n > 1 ? n : 1, '\0');
  if (program)
    glGetProgramInfoLog(object, n, nullptr, log.data());
  else
    glGetShaderInfoLog(object, n, nullptr, log.data());
  log.resize(std::strlen(log.c_str()));
  return log;
}

template <typename T>
void AppendBytes(std::string& key, const T& value) {
  key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

}  // namespace

bool FboKey::operator==(const FboKey& o) const {
  return !std::memcmp(colors, o.colors, sizeof(colors)) && depth == o.depth &&
         depth_planes == o.depth_planes;
}

size_t FboKeyHash::operator()(const FboKey& key) const {
  size_t h = reinterpret_cast<size_t>(key.depth) ^ key.depth_planes;
  for (const GlView* v : key.colors)
    h = h * 1099511628211ull ^ reinterpret_cast<size_t>(v);
  return h;
}

bool BlendState::operator==(const BlendState& o) const {
  return enable == o.enable && src_rgb == o.src_rgb && dst_rgb == o.dst_rgb &&
         op_rgb == o.op_rgb && src_alpha == o.src_alpha &&
         dst_alpha == o.dst_alpha && op_alpha == o.op_alpha && mask == o.mask;
}

bool StencilState::operator==(const StencilState& o) const {
  return func == o.func && ref == o.ref && compare_mask == o.compare_mask &&
         write_mask == o.write_mask && fail == o.fail &&
         depth_fail == o.depth_fail && pass == o.pass;
}

i32 GlBindGroupLayout::Position(u32 binding) const {
  const auto& b = desc_.bindings;
  for (size_t i = 0; i < b.size(); i++)
    if (b[i].binding == binding)
      return static_cast<i32>(i);
  return -1;
}

GlDevice::~GlDevice() {
  if (!replayer_)
    return;
  render_.Run([] { glFinish(); });
  {
    std::lock_guard<std::mutex> lock(fence_mutex_);
    stop_waiter_ = true;
  }
  fence_cv_.notify_all();
  if (waiter_.joinable())
    waiter_.join();
  if (waiter_context_ != EGL_NO_CONTEXT)
    eglDestroyContext(egl_.display, waiter_context_);
  compile_.Stop();
  resource_.Stop();
  render_.Stop();
}

bool GlDevice::Init(const OpenGLOptions& options) {
  debug_ = options.debug;
  if (!OpenEglDevice(options.gpu_filter, &egl_)) {
    BASE_LOGI("gpugl", "no EGL device with desktop GL 4.6 core");
    return false;
  }
  const EGLDisplay display = egl_.display;
  const EGLContext root = CreateGlContext(display, EGL_NO_CONTEXT, debug_);
  if (root == EGL_NO_CONTEXT) {
    BASE_LOGI("gpugl", "eglCreateContext failed: {:#x}", eglGetError());
    return false;
  }
  replayer_ = std::make_unique<Replayer>(*this);
  render_.Start(display, {root}, "gl-render", ContextInit(debug_));
  bool ok = false;
  render_.Run([&] { ok = InitRenderThread(); });
  if (!ok)
    return false;
  std::vector<EGLContext> resource{CreateGlContext(display, root, debug_)};
  std::vector<EGLContext> compile;
  for (u32 i = 0; i < std::max(options.compile_threads, 1u); i++)
    compile.push_back(CreateGlContext(display, root, debug_));
  waiter_context_ = CreateGlContext(display, root, false);
  bool contexts = resource[0] != EGL_NO_CONTEXT &&
                  waiter_context_ != EGL_NO_CONTEXT;
  for (EGLContext c : compile)
    contexts = contexts && c != EGL_NO_CONTEXT;
  if (!contexts) {
    BASE_LOGI("gpugl", "eglCreateContext failed: {:#x}", eglGetError());
    for (EGLContext c : resource)
      eglDestroyContext(display, c);
    for (EGLContext c : compile)
      eglDestroyContext(display, c);
    return false;
  }
  resource_.Start(display, resource, "gl-resource", ContextInit(debug_));
  compile_.Start(display, compile, "gl-compile", ContextInit(debug_));
  waiter_ = std::thread(&GlDevice::WaitLoop, this);
  BASE_LOGI("gpugl", "device: {}", device_name_.c_str());
  return true;
}

bool GlDevice::InitRenderThread() {
  if (GetInt(GL_MAJOR_VERSION) * 10 + GetInt(GL_MINOR_VERSION) < 46)
    return false;
  device_name_ = egl_.renderer;

  slot_limits_[0] = static_cast<u32>(GetInt(GL_MAX_UNIFORM_BUFFER_BINDINGS));
  slot_limits_[1] =
      static_cast<u32>(GetInt(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS));
  slot_limits_[2] =
      static_cast<u32>(GetInt(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS));
  slot_limits_[3] = static_cast<u32>(GetInt(GL_MAX_IMAGE_UNITS));
  glsl_.max_uniform_buffers = slot_limits_[0];
  glsl_.max_storage_buffers = slot_limits_[1];
  glsl_.max_textures = slot_limits_[2];
  glsl_.max_images = slot_limits_[3];
  glsl_.max_vertex_attribs = static_cast<u32>(GetInt(GL_MAX_VERTEX_ATTRIBS));
  slot_limits_[4] = kMaxPointers;
  glsl_.max_stage_storage_buffers = static_cast<u32>(
      std::min({GetInt(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS),
                GetInt(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS),
                GetInt(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS),
                GetInt(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)}));
  buffer_pointers_ = epoxy_has_gl_extension("GL_NV_shader_buffer_load") &&
                     epoxy_has_gl_extension("GL_NV_gpu_shader5");
  glsl_.buffer_pointers = buffer_pointers_;
  const bool bary_nv =
      epoxy_has_gl_extension("GL_NV_fragment_shader_barycentric");
  const bool bary_ext =
      epoxy_has_gl_extension("GL_EXT_fragment_shader_barycentric");
  glsl_.nv_barycentric_only = bary_nv && !bary_ext;

  caps_.backend = rhi::Backend::kOpenGL;
  caps_.device_name = device_name_.c_str();
  caps_.geometry_shader = true;
  caps_.fragment_barycentric = bary_nv || bary_ext;
  caps_.independent_blend = true;
  caps_.sampler_anisotropy = true;
  caps_.sampler_mirror_clamp = true;
  caps_.storage_image_write_without_format = true;
  caps_.texture_blit = true;
  caps_.dispatch_base = true;
  caps_.debug_labels = debug_;
  GLint bits = 0;
  glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
  caps_.timestamps = bits > 0;
  caps_.timestamp_period_ns = 1.0;
  caps_.subgroup_size = epoxy_has_gl_extension("GL_KHR_shader_subgroup")
                            ? static_cast<u32>(GetInt(GL_SUBGROUP_SIZE_KHR))
                            : 32;
  const u32 ubo_blocks = static_cast<u32>(
      std::min({GetInt(GL_MAX_VERTEX_UNIFORM_BLOCKS),
                GetInt(GL_MAX_FRAGMENT_UNIFORM_BLOCKS),
                GetInt(GL_MAX_COMPUTE_UNIFORM_BLOCKS)}));
  const u32 ssbo_blocks = static_cast<u32>(
      std::min({GetInt(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS),
                GetInt(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS),
                GetInt(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)}));
  caps_.max_dynamic_uniform_buffers = ubo_blocks > 1 ? ubo_blocks - 1 : 0;
  caps_.max_dynamic_storage_buffers = ssbo_blocks;
  caps_.max_storage_buffers_per_stage = ssbo_blocks;
  caps_.max_push_constant_bytes = kMaxPushBytes;
  GLint64 ssbo_size = 0;
  glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo_size);
  caps_.max_storage_buffer_range = static_cast<u64>(ssbo_size);
  caps_.uniform_offset_alignment =
      static_cast<u32>(GetInt(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT));
  caps_.storage_offset_alignment =
      static_cast<u32>(GetInt(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT));
  caps_.max_texture_size = static_cast<u32>(GetInt(GL_MAX_TEXTURE_SIZE));
  caps_.max_texture_size_3d =
      static_cast<u32>(GetInt(GL_MAX_3D_TEXTURE_SIZE));
  glGetFloatv(GL_MAX_TEXTURE_LOD_BIAS, &max_lod_bias_);
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max_anisotropy_);
  caps_.max_compute_resources =
      static_cast<u32>(GetInt(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS));

  for (size_t i = 1; i < static_cast<size_t>(rhi::Format::kCount); i++) {
    const auto format = static_cast<rhi::Format>(i);
    const GLenum internal = ToGl(format).internal;
    if (!internal)
      continue;
    auto query = [&](GLenum pname) {
      GLint v = GL_NONE;
      glGetInternalformativ(GL_TEXTURE_2D, internal, pname, 1, &v);
      return v;
    };
    if (query(GL_INTERNALFORMAT_SUPPORTED) != GL_TRUE)
      continue;
    const rhi::FormatInfo& fi = rhi::GetFormatInfo(format);
    u32 usage = rhi::kTextureCopySrc | rhi::kTextureCopyDst |
                rhi::kTextureMutableFormat | rhi::kTextureCubeCompatible |
                rhi::kTextureArrayCompatible;
    if (query(GL_FRAGMENT_TEXTURE) != GL_NONE)
      usage |= rhi::kTextureSampled;
    if (!fi.compressed && !fi.is_depth && !fi.is_stencil &&
        query(GL_SHADER_IMAGE_STORE) != GL_NONE)
      usage |= rhi::kTextureStorage;
    if (query(GL_FRAMEBUFFER_RENDERABLE) != GL_NONE)
      usage |= fi.is_depth || fi.is_stencil ? rhi::kTextureDepthTarget
                                            : rhi::kTextureColorTarget;
    format_usage_[i] = usage;
  }
  replayer_->Init();
  return glGetError() == GL_NO_ERROR;
}

rhi::Buffer* GlDevice::CreateBuffer(const rhi::BufferDesc& desc) {
  if (desc.host_pointer)
    return nullptr;
  auto buffer = std::make_unique<GlBuffer>(desc);
  bool ok = false;
  resource_.Run([&] {
    GLbitfield map = 0;
    GLbitfield storage = GL_DYNAMIC_STORAGE_BIT;
    switch (desc.memory) {
      case rhi::MemoryKind::kDevice:
        break;
      case rhi::MemoryKind::kUpload:
      case rhi::MemoryKind::kUploadDevice:
        map = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        break;
      case rhi::MemoryKind::kReadback:
        map = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        storage |= GL_CLIENT_STORAGE_BIT;
        break;
    }
    const GLsizeiptr size = static_cast<GLsizeiptr>(std::max<u64>(desc.size, 4));
    glCreateBuffers(1, &buffer->name);
    glNamedBufferStorage(buffer->name, size, nullptr, storage | map);
    if (map)
      buffer->set_mapped(static_cast<u8*>(
          glMapNamedBufferRange(buffer->name, 0, size, map)));
    if (buffer_pointers_ && (desc.usage & rhi::kBufferStorage)) {
      GLuint64EXT address = 0;
      glMakeNamedBufferResidentNV(buffer->name, GL_READ_WRITE);
      glGetNamedBufferParameterui64vNV(buffer->name, GL_BUFFER_GPU_ADDRESS_NV,
                                       &address);
      glMakeNamedBufferNonResidentNV(buffer->name);
      buffer->address = address;
    }
    if (desc.name && debug_)
      glObjectLabel(GL_BUFFER, buffer->name, -1, desc.name);
    ok = glGetError() == GL_NO_ERROR && (!map || buffer->mapped());
    if (!ok)
      glDeleteBuffers(1, &buffer->name);
    glFinish();
  });
  if (!ok)
    return nullptr;
  if (buffer->address) {
    // Residency is per context; the render thread's is the one that counts.
    render_.Post([name = buffer->name] {
      glMakeNamedBufferResidentNV(name, GL_READ_WRITE);
    });
  }
  return buffer.release();
}

rhi::Texture* GlDevice::CreateTexture(const rhi::TextureDesc& desc) {
  auto texture = std::make_unique<GlTexture>(desc);
  texture->internal = ToGl(desc.format).internal;
  if (!texture->internal)
    return nullptr;
  switch (desc.dim) {
    case rhi::TextureDim::k1D:
      texture->arrayed = desc.layers > 1;
      texture->target = texture->arrayed ? GL_TEXTURE_1D_ARRAY : GL_TEXTURE_1D;
      break;
    case rhi::TextureDim::k2D:
      // Cube views need an array original.
      texture->arrayed =
          desc.layers > 1 || (desc.usage & rhi::kTextureCubeCompatible);
      texture->target = texture->arrayed ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
      break;
    case rhi::TextureDim::k3D:
      texture->target = GL_TEXTURE_3D;
      break;
  }
  bool ok = false;
  resource_.Run([&] {
    GlTexture& t = *texture;
    const GLsizei mips = static_cast<GLsizei>(std::max(desc.mips, 1u));
    const GLsizei w = desc.width, h = desc.height;
    glCreateTextures(t.target, 1, &t.name);
    switch (t.target) {
      case GL_TEXTURE_1D:
        glTextureStorage1D(t.name, mips, t.internal, w);
        break;
      case GL_TEXTURE_1D_ARRAY:
        glTextureStorage2D(t.name, mips, t.internal, w, desc.layers);
        break;
      case GL_TEXTURE_2D:
        glTextureStorage2D(t.name, mips, t.internal, w, h);
        break;
      case GL_TEXTURE_2D_ARRAY:
        glTextureStorage3D(t.name, mips, t.internal, w, h, desc.layers);
        break;
      default:
        glTextureStorage3D(t.name, mips, t.internal, w, h, desc.depth);
        break;
    }
    if (desc.name && debug_)
      glObjectLabel(GL_TEXTURE, t.name, -1, desc.name);
    ok = glGetError() == GL_NO_ERROR;
    if (!ok)
      glDeleteTextures(1, &t.name);
    glFinish();
  });
  return ok ? texture.release() : nullptr;
}

rhi::TextureView* GlDevice::CreateView(rhi::Texture* texture,
                                       const rhi::TextureViewDesc& desc) {
  auto* t = static_cast<GlTexture*>(texture);
  auto view = std::make_unique<GlView>(texture, desc);
  view->format = desc.format == rhi::Format::kUndefined
                     ? texture->desc().format
                     : desc.format;
  view->internal = ToGl(view->format).internal;
  if (!view->internal)
    return nullptr;
  if (t->target == GL_TEXTURE_3D && desc.dim != rhi::ViewDim::k3D) {
    // GL cannot view 3D as 2D: address the slices of the texture itself.
    view->name = t->name;
    view->owns_name = false;
    view->internal = t->internal;
    view->level = static_cast<GLint>(desc.base_mip);
    view->layer = desc.layers == 1 ? static_cast<GLint>(desc.base_layer) : -1;
    return view.release();
  }
  const GLenum target = ViewTarget(desc.dim);
  if ((target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_2D_ARRAY) &&
      desc.layers == 1)
    view->layer = 0;
  bool ok = false;
  resource_.Run([&] {
    GlView& v = *view;
    glGenTextures(1, &v.name);
    glTextureView(v.name, target, t->name, v.internal, desc.base_mip,
                  std::max(desc.mips, 1u), desc.base_layer,
                  std::max(desc.layers, 1u));
    const GLint swizzle[4] = {ToGlSwizzle(desc.swizzle[0], GL_RED),
                              ToGlSwizzle(desc.swizzle[1], GL_GREEN),
                              ToGlSwizzle(desc.swizzle[2], GL_BLUE),
                              ToGlSwizzle(desc.swizzle[3], GL_ALPHA)};
    if (swizzle[0] != GL_RED || swizzle[1] != GL_GREEN ||
        swizzle[2] != GL_BLUE || swizzle[3] != GL_ALPHA)
      glTextureParameteriv(v.name, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
    const rhi::FormatInfo& fi = rhi::GetFormatInfo(v.format);
    if (fi.is_depth && fi.is_stencil && desc.aspect == rhi::kAspectStencil)
      glTextureParameteri(v.name, GL_DEPTH_STENCIL_TEXTURE_MODE,
                          GL_STENCIL_INDEX);
    ok = glGetError() == GL_NO_ERROR;
    if (!ok)
      glDeleteTextures(1, &v.name);
    glFinish();
  });
  return ok ? view.release() : nullptr;
}

rhi::Sampler* GlDevice::CreateSampler(const rhi::SamplerDesc& desc) {
  auto sampler = std::make_unique<GlSampler>();
  const bool filtered = desc.mag == rhi::Filter::kLinear ||
                        desc.min == rhi::Filter::kLinear ||
                        desc.mip == rhi::Filter::kLinear;
  resource_.Run([&] {
    auto make = [&](bool linear_ok) {
      GLuint s = 0;
      glCreateSamplers(1, &s);
      const bool mag = linear_ok && desc.mag == rhi::Filter::kLinear;
      const bool min = linear_ok && desc.min == rhi::Filter::kLinear;
      const bool mip = linear_ok && desc.mip == rhi::Filter::kLinear;
      glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, mag ? GL_LINEAR : GL_NEAREST);
      glSamplerParameteri(
          s, GL_TEXTURE_MIN_FILTER,
          min ? (mip ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR_MIPMAP_NEAREST)
              : (mip ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST));
      glSamplerParameteri(s, GL_TEXTURE_WRAP_S, ToGlAddress(desc.address_u));
      glSamplerParameteri(s, GL_TEXTURE_WRAP_T, ToGlAddress(desc.address_v));
      glSamplerParameteri(s, GL_TEXTURE_WRAP_R, ToGlAddress(desc.address_w));
      glSamplerParameterf(s, GL_TEXTURE_LOD_BIAS,
                          std::clamp(desc.lod_bias, -max_lod_bias_,
                                     max_lod_bias_));
      glSamplerParameterf(s, GL_TEXTURE_MIN_LOD, desc.min_lod);
      glSamplerParameterf(s, GL_TEXTURE_MAX_LOD, desc.max_lod);
      if (linear_ok && desc.max_anisotropy > 1.0f)
        glSamplerParameterf(s, GL_TEXTURE_MAX_ANISOTROPY,
                            std::min(desc.max_anisotropy, max_anisotropy_));
      if (desc.compare_enable) {
        glSamplerParameteri(s, GL_TEXTURE_COMPARE_MODE,
                            GL_COMPARE_REF_TO_TEXTURE);
        glSamplerParameteri(s, GL_TEXTURE_COMPARE_FUNC,
                            ToGlCompare(desc.compare));
      }
      static constexpr float kBorders[3][4] = {
          {0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
      glSamplerParameterfv(s, GL_TEXTURE_BORDER_COLOR,
                           kBorders[static_cast<u32>(desc.border) % 3]);
      return s;
    };
    sampler->name = make(true);
    sampler->nearest = filtered ? make(false) : sampler->name;
    glFinish();
  });
  return sampler.release();
}

rhi::BindGroupLayout* GlDevice::CreateBindGroupLayout(
    const rhi::BindGroupLayoutDesc& desc) {
  auto layout = std::make_unique<GlBindGroupLayout>(desc);
  // Dynamic offsets come in binding-number order.
  std::vector<std::pair<u32, size_t>> dynamic;
  for (size_t i = 0; i < desc.bindings.size(); i++) {
    const rhi::BindingType t = desc.bindings[i].type;
    if (t == rhi::BindingType::kUniformBufferDynamic ||
        t == rhi::BindingType::kStorageBufferDynamic)
      dynamic.push_back({desc.bindings[i].binding, i});
  }
  if (dynamic.size() > kMaxDynamicOffsets)
    return nullptr;
  std::sort(dynamic.begin(), dynamic.end());
  layout->dynamic_index.assign(desc.bindings.size(), kNotDynamic);
  for (size_t i = 0; i < dynamic.size(); i++)
    layout->dynamic_index[dynamic[i].second] = static_cast<u8>(i);
  return layout.release();
}

void GlDevice::Resolve(const GlBindGroupLayout& layout,
                       const rhi::BindingWrite& write,
                       BoundResource* out) const {
  const i32 pos = layout.Position(write.binding);
  if (pos < 0)
    return;
  BoundResource& r = out[pos];
  r = {};
  r.dynamic = layout.dynamic_index[pos];
  switch (layout.desc().bindings[pos].type) {
    case rhi::BindingType::kSampledTexture: {
      const auto* v = static_cast<const GlView*>(write.view);
      const auto* s = static_cast<const GlSampler*>(write.sampler);
      if (v) {
        r.name = v->name;
        const bool integer = rhi::GetFormatInfo(v->format).is_integer ||
                             v->desc().aspect == rhi::kAspectStencil;
        if (s)
          r.sampler = integer ? s->nearest : s->name;
      }
      break;
    }
    case rhi::BindingType::kStorageTexture: {
      const auto* v = static_cast<const GlView*>(write.view);
      if (v) {
        r.name = v->name;
        r.format = v->internal;
        r.level = v->owns_name ? 0 : v->level;
        r.layered = v->owns_name || v->layer < 0 ? GL_TRUE : GL_FALSE;
        r.layer = v->owns_name ? 0 : std::max(v->layer, 0);
      }
      break;
    }
    default: {
      const auto* b = static_cast<const GlBuffer*>(write.buffer);
      if (b) {
        r.name = b->name;
        r.address = b->address;
        r.offset = write.offset;
        r.buffer_size = b->desc().size;
        r.size = write.range ? write.range
                             : (r.buffer_size > write.offset
                                    ? r.buffer_size - write.offset
                                    : 0);
      }
      break;
    }
  }
}

rhi::BindGroup* GlDevice::CreateBindGroup(const rhi::BindGroupDesc& desc) {
  auto group = std::make_unique<GlBindGroup>();
  group->layout = static_cast<GlBindGroupLayout*>(desc.layout);
  group->entries.resize(group->layout->desc().bindings.size());
  UpdateBindGroup(group.get(), desc.writes.data(),
                  static_cast<u32>(desc.writes.size()));
  return group.release();
}

void GlDevice::UpdateBindGroup(rhi::BindGroup* group,
                               const rhi::BindingWrite* writes,
                               u32 count) {
  auto* g = static_cast<GlBindGroup*>(group);
  for (u32 i = 0; i < count; i++)
    Resolve(*g->layout, writes[i], g->entries.data());
}

rhi::PipelineLayout* GlDevice::CreatePipelineLayout(
    const rhi::PipelineLayoutDesc& desc) {
  if (desc.groups.size() > kMaxGroups ||
      desc.push_constant_bytes > kMaxPushBytes)
    return nullptr;
  return new GlPipelineLayout(desc);
}

const SlotMap* GlDevice::InternSlots(const rhi::PipelineLayoutDesc& layout,
                                     const ProgramInterface& program) {
  auto map = std::make_unique<SlotMap>();
  std::string key;
  for (const ResourceSlot& s : program.slots) {
    if (map->groups.size() <= s.set)
      map->groups.resize(s.set + 1);
    SlotMap::Group& g = map->groups[s.set];
    g.layout = static_cast<const GlBindGroupLayout*>(layout.groups[s.set]);
    g.slots.push_back({s.binding, g.layout->Position(s.binding), s.kind,
                       s.slot});
  }
  for (const SlotMap::Group& g : map->groups) {
    AppendBytes(key, g.layout);
    for (const SlotMap::Slot& s : g.slots) {
      AppendBytes(key, s.binding);
      AppendBytes(key, s.kind);
      AppendBytes(key, s.slot);
    }
    key += '|';
  }
  std::lock_guard<std::mutex> lock(intern_mutex_);
  auto [it, inserted] = slot_maps_.try_emplace(key, nullptr);
  if (inserted)
    it->second = std::move(map);
  return it->second.get();
}

VertexInput* GlDevice::InternVertexInput(
    const rhi::GraphicsPipelineDesc& desc,
    const ProgramInterface& program) {
  auto input = std::make_unique<VertexInput>();
  for (const rhi::VertexAttribute& a : desc.vertex_attributes) {
    // Attributes the shader does not read are left disabled.
    auto it = std::find_if(program.vertex_locations.begin(),
                           program.vertex_locations.end(),
                           [&](const auto& m) { return m.first == a.location; });
    if (it == program.vertex_locations.end())
      continue;
    const GlFormat& f = ToGl(a.format);
    VertexInput::Attribute attr;
    attr.location = it->second;
    attr.binding = a.buffer;
    attr.size = f.vertex_size;
    attr.type = f.vertex_type;
    attr.kind = rhi::GetFormatInfo(a.format).is_integer
                    ? 2
                    : (f.vertex_normalized ? 1 : 0);
    attr.offset = a.offset;
    input->attributes.push_back(attr);
  }
  for (const rhi::VertexBufferLayout& b : desc.vertex_buffers)
    input->per_instance.push_back(b.per_instance ? 1 : 0);
  std::string key;
  for (const auto& a : input->attributes) {
    AppendBytes(key, a.location);
    AppendBytes(key, a.binding);
    AppendBytes(key, a.size);
    AppendBytes(key, a.type);
    AppendBytes(key, a.kind);
    AppendBytes(key, a.offset);
  }
  key += '|';
  key.append(input->per_instance.begin(), input->per_instance.end());
  std::lock_guard<std::mutex> lock(intern_mutex_);
  auto [it, inserted] = vertex_inputs_.try_emplace(key, nullptr);
  if (inserted)
    it->second = std::move(input);
  return it->second.get();
}

rhi::Pipeline* GlDevice::BuildPipeline(
    const rhi::PipelineLayoutDesc& layout,
    const StageCode* stages,
    u32 count,
    const rhi::GraphicsPipelineDesc* graphics,
    GlPipeline* raw) {
  std::unique_ptr<GlPipeline> pipeline(raw);
  std::vector<const rhi::BindGroupLayoutDesc*> groups;
  for (rhi::BindGroupLayout* g : layout.groups)
    groups.push_back(g ? &g->desc() : nullptr);
  std::vector<std::string> glsl;
  ProgramInterface program;
  std::string error;
  if (!LowerProgram(stages, count, groups, glsl_, &glsl, &program, &error)) {
    BASE_LOGI("gpugl", "shader lowering failed: {}", error.c_str());
    return nullptr;
  }
  pipeline->slots = InternSlots(layout, program);
  if (graphics)
    pipeline->vertex = InternVertexInput(*graphics, program);
  pipeline->push_ubo = program.push_ubo;
  pipeline->pointer_count = program.pointer_count;
  if (program.pointer_count > kMaxPointers) {
    BASE_LOGI("gpugl", "too many storage buffers: {}", program.pointer_count);
    return nullptr;
  }
  pipeline->push_bytes = layout.push_constant_bytes;
  compile_.Run([&] {
    const GLuint prog = glCreateProgram();
    GLuint shaders[3] = {};
    bool ok = true;
    for (u32 i = 0; i < count && ok; i++) {
      GLenum type = GL_COMPUTE_SHADER;
      switch (stages[i].stage) {
        case rhi::kStageVertex:
          type = GL_VERTEX_SHADER;
          break;
        case rhi::kStageGeometry:
          type = GL_GEOMETRY_SHADER;
          break;
        case rhi::kStageFragment:
          type = GL_FRAGMENT_SHADER;
          break;
        default:
          break;
      }
      shaders[i] = glCreateShader(type);
      const char* src = glsl[i].c_str();
      glShaderSource(shaders[i], 1, &src, nullptr);
      glCompileShader(shaders[i]);
      GLint status = 0;
      glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &status);
      if (!status) {
        error = InfoLog(shaders[i], false);
        ok = false;
      }
      glAttachShader(prog, shaders[i]);
    }
    if (ok) {
      glLinkProgram(prog);
      GLint status = 0;
      glGetProgramiv(prog, GL_LINK_STATUS, &status);
      if (!status) {
        error = InfoLog(prog, true);
        ok = false;
      }
    }
    for (GLuint s : shaders)
      if (s) {
        glDetachShader(prog, s);
        glDeleteShader(s);
      }
    if (!ok) {
      glDeleteProgram(prog);
      return;
    }
    pipeline->program = prog;
    if (program.pointer_count)
      pipeline->pointer_location = glGetUniformLocation(prog, kPointerTable);
    pipeline->base_location = glGetUniformLocation(prog, kDispatchBase);
    for (const PushUniform& u : program.push_uniforms) {
      GlPipeline::Push& p = pipeline->push[pipeline->push_count++];
      p.location = glGetUniformLocation(prog, u.name.c_str());
      p.vec4_count = u.vec4_count;
      p.type = u.type;
    }
    glFinish();
  });
  if (!pipeline->program) {
    BASE_LOGI("gpugl", "program build failed: {}", error.c_str());
    return nullptr;
  }
  return pipeline.release();
}

rhi::Pipeline* GlDevice::CreateGraphicsPipeline(
    const rhi::GraphicsPipelineDesc& desc) {
  if (desc.mesh || desc.vertex.empty() ||
      desc.vertex_buffers.size() > kMaxVertexBuffers)
    return nullptr;
  auto pipeline = std::make_unique<GlPipeline>();
  StageCode stages[3];
  u32 count = 0;
  stages[count++] = {rhi::kStageVertex, desc.vertex};
  if (!desc.geometry.empty())
    stages[count++] = {rhi::kStageGeometry, desc.geometry};
  if (!desc.fragment.empty())
    stages[count++] = {rhi::kStageFragment, desc.fragment};

  for (size_t i = 0; i < desc.vertex_buffers.size(); i++)
    pipeline->strides[i] = desc.vertex_buffers[i].stride;
  pipeline->mode = ToGlMode(desc.topology);
  RasterState& r = pipeline->raster;
  r.primitive_restart = desc.primitive_restart;
  r.cull = desc.cull != rhi::CullMode::kNone;
  r.cull_face = desc.cull == rhi::CullMode::kFront       ? GL_FRONT
                : desc.cull == rhi::CullMode::kFrontAndBack ? GL_FRONT_AND_BACK
                                                             : GL_BACK;
  r.front_ccw = desc.front_ccw;
  r.depth_clamp = desc.depth_clamp;
  r.depth_test = desc.depth_test;
  r.depth_write = desc.depth_write;
  r.depth_func = ToGlCompare(desc.depth_compare);
  r.stencil_test = desc.stencil_test;
  r.front = ToGlStencil(desc.stencil_front);
  r.back = ToGlStencil(desc.stencil_back);
  for (u32 i = 0; i < 8; i++) {
    BlendState& b = r.blend[i];
    if (i >= desc.color_count ||
        desc.color_formats[i] == rhi::Format::kUndefined) {
      b.mask = 0;
      continue;
    }
    const rhi::BlendAttachment& a = desc.blend[i];
    b.enable = a.enable;
    b.src_rgb = ToGlBlendFactor(a.src_color);
    b.dst_rgb = ToGlBlendFactor(a.dst_color);
    b.op_rgb = ToGlBlendOp(a.color_op);
    b.src_alpha = ToGlBlendFactor(a.src_alpha);
    b.dst_alpha = ToGlBlendFactor(a.dst_alpha);
    b.op_alpha = ToGlBlendOp(a.alpha_op);
    b.mask = a.write_mask & 0xF;
  }
  const auto* layout = static_cast<GlPipelineLayout*>(desc.layout);
  return BuildPipeline(layout->desc(), stages, count, &desc,
                       pipeline.release());
}

rhi::Pipeline* GlDevice::CreateComputePipeline(
    const rhi::ComputePipelineDesc& desc) {
  auto pipeline = std::make_unique<GlPipeline>();
  pipeline->compute = true;
  const StageCode stage{rhi::kStageCompute, desc.code, desc.dispatch_base};
  const auto* layout = static_cast<GlPipelineLayout*>(desc.layout);
  return BuildPipeline(layout->desc(), &stage, 1, nullptr, pipeline.release());
}

rhi::TimestampPool* GlDevice::CreateTimestampPool(u32 count) {
  auto pool = std::make_unique<GlTimestampPool>();
  pool->count = count;
  bool ok = false;
  resource_.Run([&] {
    const GLbitfield map =
        GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    const GLsizeiptr size = std::max<GLsizeiptr>(count, 1) * 8;
    glCreateBuffers(1, &pool->buffer);
    glNamedBufferStorage(pool->buffer, size, nullptr,
                         map | GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT);
    glClearNamedBufferData(pool->buffer, GL_R32UI, GL_RED_INTEGER,
                           GL_UNSIGNED_INT, nullptr);
    pool->results = static_cast<const u64*>(
        glMapNamedBufferRange(pool->buffer, 0, size, map));
    ok = pool->results && glGetError() == GL_NO_ERROR;
    glFinish();
  });
  return ok ? pool.release() : nullptr;
}

rhi::CommandList* GlDevice::CreateCommandList() {
  return new GlCommandList(*this);
}

void GlDevice::Destroy(rhi::Object* object) {
  if (!object)
    return;
  // After every submission queued so far, and after the render thread
  // stopped naming the object in its caches.
  render_.Post([this, object] {
    replayer_->Forget(object);
    if (auto* b = dynamic_cast<GlBuffer*>(object)) {
      if (b->mapped())
        glUnmapNamedBuffer(b->name);
      glDeleteBuffers(1, &b->name);
    } else if (auto* t = dynamic_cast<GlTexture*>(object)) {
      glDeleteTextures(1, &t->name);
    } else if (auto* v = dynamic_cast<GlView*>(object)) {
      if (v->owns_name)
        glDeleteTextures(1, &v->name);
    } else if (auto* s = dynamic_cast<GlSampler*>(object)) {
      if (s->nearest != s->name)
        glDeleteSamplers(1, &s->nearest);
      glDeleteSamplers(1, &s->name);
    } else if (auto* p = dynamic_cast<GlPipeline*>(object)) {
      glDeleteProgram(p->program);
    } else if (auto* q = dynamic_cast<GlTimestampPool*>(object)) {
      if (!q->queries.empty())
        glDeleteQueries(static_cast<GLsizei>(q->queries.size()),
                        q->queries.data());
      glUnmapNamedBuffer(q->buffer);
      glDeleteBuffers(1, &q->buffer);
    }
    delete object;
  });
}

void GlDevice::SetName(rhi::Object* object, const char* name) {
  if (!debug_ || !object || !name)
    return;
  GLenum type = 0;
  GLuint handle = 0;
  if (auto* b = dynamic_cast<GlBuffer*>(object)) {
    type = GL_BUFFER;
    handle = b->name;
  } else if (auto* t = dynamic_cast<GlTexture*>(object)) {
    type = GL_TEXTURE;
    handle = t->name;
  } else if (auto* p = dynamic_cast<GlPipeline*>(object)) {
    type = GL_PROGRAM;
    handle = p->program;
  } else {
    return;
  }
  render_.Post([type, handle, label = std::string(name)] {
    glObjectLabel(type, handle, -1, label.c_str());
  });
}

bool GlDevice::SupportsFormat(rhi::Format format, u32 usage) const {
  const size_t i = static_cast<size_t>(format);
  if (i == 0 || i >= static_cast<size_t>(rhi::Format::kCount))
    return false;
  return (format_usage_[i] & usage) == usage;
}

u64 GlDevice::Submit(rhi::CommandList* const* lists, u32 count) {
  std::vector<const GlCommandList*> work(count);
  for (u32 i = 0; i < count; i++)
    work[i] = static_cast<const GlCommandList*>(lists[i]);
  std::lock_guard<std::mutex> lock(submit_mutex_);
  const u64 id = submitted_ + 1;
  render_.Post([this, work = std::move(work), id] {
    for (const GlCommandList* list : work)
      replayer_->Execute(*list);
    const GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    {
      std::lock_guard<std::mutex> fence_lock(fence_mutex_);
      fences_.push_back({id, fence});
    }
    fence_cv_.notify_one();
  });
  submitted_ = id;
  return id;
}

void GlDevice::WaitLoop() {
  pthread_setname_np(pthread_self(), "gl-fence");
  eglBindAPI(EGL_OPENGL_API);
  eglMakeCurrent(egl_.display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                 waiter_context_);
  while (true) {
    std::pair<u64, GLsync> fence;
    {
      std::unique_lock<std::mutex> lock(fence_mutex_);
      fence_cv_.wait(lock, [&] { return stop_waiter_ || !fences_.empty(); });
      if (fences_.empty())
        break;
      fence = fences_.front();
      fences_.pop_front();
    }
    GLenum r;
    do {
      r = glClientWaitSync(fence.second, 0, 100'000'000);
    } while (r == GL_TIMEOUT_EXPIRED);
    if (r == GL_WAIT_FAILED) {
      BASE_LOGI("gpugl", "fence wait failed: device lost?");
      lost_ = true;
    }
    glDeleteSync(fence.second);
    {
      std::lock_guard<std::mutex> lock(completed_mutex_);
      completed_ = fence.first;
    }
    completed_cv_.notify_all();
  }
  eglMakeCurrent(egl_.display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                 EGL_NO_CONTEXT);
  eglReleaseThread();
}

bool GlDevice::IsComplete(u64 submission) {
  return completed_ >= submission;
}

bool GlDevice::Wait(u64 submission, u64 timeout_ns) {
  auto done = [&] { return completed_ >= submission || lost_; };
  std::unique_lock<std::mutex> lock(completed_mutex_);
  if (timeout_ns == ~0ull)
    completed_cv_.wait(lock, done);
  else
    completed_cv_.wait_for(lock, std::chrono::nanoseconds(timeout_ns), done);
  return completed_ >= submission;
}

void GlDevice::WaitIdle() {
  render_.Run([] { glFinish(); });
  Wait(submitted_, ~0ull);
}

bool GlDevice::ReadTimestamps(rhi::TimestampPool* pool,
                              u32 first,
                              u32 count,
                              u64* out) {
  auto* p = static_cast<GlTimestampPool*>(pool);
  if (first + count > p->count)
    return false;
  for (u32 i = 0; i < count; i++) {
    const u64 v = p->results[first + i];
    if (!v)
      return false;
    out[i] = v;
  }
  return true;
}

std::unique_ptr<rhi::Device> CreateOpenGLDevice(const OpenGLOptions& options) {
  auto device = std::make_unique<GlDevice>();
  if (!device->Init(options))
    return nullptr;
  return device;
}

}  // namespace gpu::opengl
