/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cmath>
#include <cstring>

#include <base/logging.h>

#include "gpu/opengl/gl_rhi_internal.h"
#include <base/algorithm.h>
#include <base/math/value_bounds.h>

namespace gpu::opengl {

namespace {

template <typename T>
const T& As(const u8* p) {
  return *reinterpret_cast<const T*>(p);
}

template <typename T>
const u8* Payload(const T& c) {
  return reinterpret_cast<const u8*>(&c) + sizeof(T);
}

void Enable(GLenum cap, bool on) {
  if (on)
    glEnable(cap);
  else
    glDisable(cap);
}

float EncodeSrgb(float v) {
  v = base::Clamp(v, 0.0f, 1.0f);
  return v <= 0.0031308f ? v * 12.92f
                         : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

u32 MipExtent(u32 extent, u32 mip) {
  return base::Max(extent >> mip, 1u);
}

// The attachment point for the planes of a depth/stencil format in `aspect`.
GLenum DepthAttachment(u8 aspect) {
  const bool depth = aspect & rhi::kAspectDepth;
  const bool stencil = aspect & rhi::kAspectStencil;
  return depth && stencil ? GL_DEPTH_STENCIL_ATTACHMENT
         : depth          ? GL_DEPTH_ATTACHMENT
                          : GL_STENCIL_ATTACHMENT;
}

// Planes of `format` selected by `aspect`.
u8 Planes(rhi::Format format, u8 aspect) {
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(format);
  u8 planes = 0;
  if (fi.is_depth)
    planes |= rhi::kAspectDepth;
  if (fi.is_stencil)
    planes |= rhi::kAspectStencil;
  return planes ? planes & aspect : rhi::kAspectColor;
}

}  // namespace

void Replayer::Init() {
  glEnable(GL_FRAMEBUFFER_SRGB);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_DITHER);
  glEnable(GL_SCISSOR_TEST);
  scissor_test_ = true;
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // Window coordinates are framebuffer rows, so the point sprite's t grows
  // downwards as in Vulkan.
  glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
  glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
  upper_left_ = false;
  glFrontFace(GL_CCW);
  front_face_ = GL_CCW;

  // Bring GL to RasterState's defaults where its own differ.
  raster_ = {};
  glDepthMask(GL_FALSE);
  glDepthFunc(GL_ALWAYS);
  for (GLenum face : {GL_FRONT, GL_BACK}) {
    glStencilFuncSeparate(face, GL_ALWAYS, 0, 0xFF);
    glStencilMaskSeparate(face, 0xFF);
  }

  glCreateFramebuffers(1, &scratch_read_);
  glCreateFramebuffers(1, &scratch_draw_);
  glNamedFramebufferReadBuffer(scratch_read_, GL_COLOR_ATTACHMENT0);
  glNamedFramebufferDrawBuffer(scratch_draw_, GL_COLOR_ATTACHMENT0);
  glCreateBuffers(1, &push_ubo_);
  glNamedBufferStorage(push_ubo_, kMaxPushBytes, nullptr,
                       GL_DYNAMIC_STORAGE_BIT);
}

void Replayer::Forget(rhi::Object* object) {
  if (auto* v = dynamic_cast<GlView*>(object)) {
    for (const FboKey& key : v->fbos) {
      auto it = fbos_.find(key);
      if (it == fbos_.end())
        continue;
      if (fbo_ == it->second) {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        fbo_ = 0;
      }
      glDeleteFramebuffers(1, &it->second);
      fbos_.erase(it);
    }
    v->fbos.clear();
  } else if (auto* b = dynamic_cast<GlBuffer*>(object)) {
    // A deleted buffer is unbound, and its name may come back.
    if (pixel_pack_ == b->name)
      pixel_pack_ = 0;
    if (pixel_unpack_ == b->name)
      pixel_unpack_ = 0;
  } else if (auto* p = dynamic_cast<GlPipeline*>(object)) {
    if (pipeline_ == p)
      pipeline_ = nullptr;
    if (program_ == p->program)
      program_ = ~0u;
  }
}

GLuint Replayer::Framebuffer(const FboKey& key) {
  auto it = fbos_.find(key);
  if (it != fbos_.end())
    return it->second;
  GLuint fbo = 0;
  glCreateFramebuffers(1, &fbo);
  auto attach = [&](GLenum point, const GlView* v) {
    if (v->layer < 0)
      glNamedFramebufferTexture(fbo, point, v->name, v->level);
    else
      glNamedFramebufferTextureLayer(fbo, point, v->name, v->level, v->layer);
    auto& keys = v->fbos;
    if (base::Find(keys.begin(), keys.end(), key) == keys.end())
      keys.push_back(key);
  };
  GLenum draw[8];
  GLsizei count = 0;
  for (u32 i = 0; i < 8; i++) {
    draw[i] = GL_NONE;
    if (!key.colors[i])
      continue;
    attach(GL_COLOR_ATTACHMENT0 + i, key.colors[i]);
    draw[i] = GL_COLOR_ATTACHMENT0 + i;
    count = static_cast<GLsizei>(i + 1);
  }
  glNamedFramebufferDrawBuffers(fbo, count, draw);
  if (key.depth && key.depth_planes)
    attach(DepthAttachment(key.depth_planes), key.depth);
  const GLenum status = glCheckNamedFramebufferStatus(fbo, GL_DRAW_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    BASE_LOGI("gpugl", "framebuffer incomplete: {:#x}", status);
  fbos_.emplace(key, fbo);
  return fbo;
}

void Replayer::SetScissorTest(bool enable) {
  if (scissor_test_ != enable) {
    Enable(GL_SCISSOR_TEST, enable);
    scissor_test_ = enable;
  }
}

void Replayer::SetScissor(i32 x, i32 y, u32 width, u32 height) {
  const i32 want[4] = {x, y, static_cast<i32>(width),
                       static_cast<i32>(height)};
  if (std::memcmp(want, gl_scissor_, sizeof(want))) {
    glScissorIndexed(0, x, y, static_cast<GLsizei>(width),
                     static_cast<GLsizei>(height));
    std::memcpy(gl_scissor_, want, sizeof(want));
  }
}

void Replayer::ResetMasks() {
  for (u32 i = 0; i < 8; i++)
    if (raster_.blend[i].mask != 0xF) {
      glColorMaski(i, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      raster_.blend[i].mask = 0xF;
    }
  if (!raster_.depth_write) {
    glDepthMask(GL_TRUE);
    raster_.depth_write = true;
  }
  if (raster_.front.write_mask != 0xFF) {
    glStencilMaskSeparate(GL_FRONT, 0xFF);
    raster_.front.write_mask = 0xFF;
  }
  if (raster_.back.write_mask != 0xFF) {
    glStencilMaskSeparate(GL_BACK, 0xFF);
    raster_.back.write_mask = 0xFF;
  }
}

void Replayer::PrepareClear(i32 x, i32 y, u32 width, u32 height) {
  SetScissorTest(true);
  SetScissor(x, y, width, height);
  ResetMasks();
}

void Replayer::ClearFramebuffer(GLuint fbo,
                                u32 index,
                                ClearKind kind,
                                const rhi::ClearColor& color) {
  switch (kind) {
    case ClearKind::kFloat:
      glClearNamedFramebufferfv(fbo, GL_COLOR, static_cast<GLint>(index),
                                color.f);
      break;
    case ClearKind::kUint:
      glClearNamedFramebufferuiv(fbo, GL_COLOR, static_cast<GLint>(index),
                                 color.u);
      break;
    case ClearKind::kInt:
      glClearNamedFramebufferiv(fbo, GL_COLOR, static_cast<GLint>(index),
                                color.i);
      break;
  }
}

void Replayer::ClearDepthStencilFbo(GLuint fbo,
                                    u8 aspect,
                                    float depth,
                                    u8 stencil) {
  const bool d = aspect & rhi::kAspectDepth;
  const bool s = aspect & rhi::kAspectStencil;
  const GLint si = stencil;
  if (d && s)
    glClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, depth, si);
  else if (d)
    glClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &depth);
  else if (s)
    glClearNamedFramebufferiv(fbo, GL_STENCIL, 0, &si);
}

void Replayer::BeginPass(const CmdBeginPass& c) {
  const GLuint fbo = Framebuffer(c.key);
  pass_read_only_ = c.depth_read_only;
  if (fbo != fbo_) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    fbo_ = fbo;
  }
  if (c.discard_mask || c.depth_discard) {
    GLenum points[9];
    GLsizei n = 0;
    for (u32 i = 0; i < 8; i++)
      if (c.discard_mask & (1u << i))
        points[n++] = GL_COLOR_ATTACHMENT0 + i;
    if (c.depth_discard)
      points[n++] = DepthAttachment(c.depth_discard);
    glInvalidateNamedFramebufferSubData(fbo, n, points, c.x, c.y,
                                        static_cast<GLsizei>(c.width),
                                        static_cast<GLsizei>(c.height));
  }
  if (c.clear_mask || c.depth_clear) {
    PrepareClear(c.x, c.y, c.width, c.height);
    for (u32 i = 0; i < 8; i++)
      if (c.clear_mask & (1u << i))
        ClearFramebuffer(fbo, i, c.kinds[i], c.clear[i]);
    if (c.depth_clear)
      ClearDepthStencilFbo(fbo, c.depth_clear, c.clear_depth,
                           c.clear_stencil);
  }
}

void Replayer::EndPass(const CmdEndPass& c) {
  if (!c.discard_mask)
    return;
  GLenum points[8];
  GLsizei n = 0;
  for (u32 i = 0; i < 8; i++)
    if (c.discard_mask & (1u << i))
      points[n++] = GL_COLOR_ATTACHMENT0 + i;
  glInvalidateNamedFramebufferSubData(fbo_, n, points, c.x, c.y,
                                      static_cast<GLsizei>(c.width),
                                      static_cast<GLsizei>(c.height));
}

void Replayer::UseProgram(GLuint program) {
  if (program != program_) {
    glUseProgram(program);
    program_ = program;
  }
}

GLuint Replayer::Vao(VertexInput* input) {
  if (input->vao)
    return input->vao;
  GLuint vao = 0;
  glCreateVertexArrays(1, &vao);
  for (const VertexInput::Attribute& a : input->attributes) {
    glEnableVertexArrayAttrib(vao, a.location);
    if (a.kind == 2)
      glVertexArrayAttribIFormat(vao, a.location, a.size, a.type, a.offset);
    else
      glVertexArrayAttribFormat(vao, a.location, a.size, a.type,
                                a.kind == 1 ? GL_TRUE : GL_FALSE, a.offset);
    glVertexArrayAttribBinding(vao, a.location, a.binding);
  }
  for (size_t i = 0; i < input->per_instance.size(); i++)
    glVertexArrayBindingDivisor(vao, static_cast<GLuint>(i),
                                input->per_instance[i]);
  input->vao = vao;
  return vao;
}

void Replayer::BindVao(VertexInput* input) {
  const GLuint vao = Vao(input);
  if (vao != vao_) {
    glBindVertexArray(vao);
    vao_ = vao;
  }
}

void Replayer::ApplyStencil(GLenum face,
                            const StencilState& s,
                            StencilState& have) {
  if (s.func != have.func || s.ref != have.ref ||
      s.compare_mask != have.compare_mask)
    glStencilFuncSeparate(face, s.func, s.ref, s.compare_mask);
  if (s.fail != have.fail || s.depth_fail != have.depth_fail ||
      s.pass != have.pass)
    glStencilOpSeparate(face, s.fail, s.depth_fail, s.pass);
  if (s.write_mask != have.write_mask)
    glStencilMaskSeparate(face, s.write_mask);
  have = s;
}

void Replayer::ApplyBlend(u32 i, const BlendState& b) {
  BlendState& have = raster_.blend[i];
  if (b.enable != have.enable) {
    if (b.enable)
      glEnablei(GL_BLEND, i);
    else
      glDisablei(GL_BLEND, i);
    have.enable = b.enable;
  }
  if (b.enable &&
      (b.src_rgb != have.src_rgb || b.dst_rgb != have.dst_rgb ||
       b.src_alpha != have.src_alpha || b.dst_alpha != have.dst_alpha)) {
    glBlendFuncSeparatei(i, b.src_rgb, b.dst_rgb, b.src_alpha, b.dst_alpha);
    have.src_rgb = b.src_rgb;
    have.dst_rgb = b.dst_rgb;
    have.src_alpha = b.src_alpha;
    have.dst_alpha = b.dst_alpha;
  }
  if (b.enable && (b.op_rgb != have.op_rgb || b.op_alpha != have.op_alpha)) {
    glBlendEquationSeparatei(i, b.op_rgb, b.op_alpha);
    have.op_rgb = b.op_rgb;
    have.op_alpha = b.op_alpha;
  }
  if (b.mask != have.mask) {
    glColorMaski(i, b.mask & 1, (b.mask >> 1) & 1, (b.mask >> 2) & 1,
                 (b.mask >> 3) & 1);
    have.mask = b.mask;
  }
}

void Replayer::ApplyGraphics() {
  const GlPipeline& p = *pipeline_;
  UseProgram(p.program);
  BindVao(p.vertex);
  SetScissorTest(true);
  SetScissor(scissor_[0], scissor_[1], static_cast<u32>(scissor_[2]),
             static_cast<u32>(scissor_[3]));
  const RasterState& r = p.raster;
  if (r.primitive_restart != raster_.primitive_restart) {
    Enable(GL_PRIMITIVE_RESTART_FIXED_INDEX, r.primitive_restart);
    raster_.primitive_restart = r.primitive_restart;
  }
  if (r.cull != raster_.cull) {
    Enable(GL_CULL_FACE, r.cull);
    raster_.cull = r.cull;
  }
  if (r.cull && r.cull_face != raster_.cull_face) {
    glCullFace(r.cull_face);
    raster_.cull_face = r.cull_face;
  }
  // A lower-left origin mirrors the framebuffer y the winding is measured
  // in, so GL's notion of counter-clockwise flips with it.
  const GLenum front = r.front_ccw == upper_left_ ? GL_CCW : GL_CW;
  if (front != front_face_) {
    glFrontFace(front);
    front_face_ = front;
  }
  if (r.depth_clamp != raster_.depth_clamp) {
    Enable(GL_DEPTH_CLAMP, r.depth_clamp);
    raster_.depth_clamp = r.depth_clamp;
  }
  if (r.depth_test != raster_.depth_test) {
    Enable(GL_DEPTH_TEST, r.depth_test);
    raster_.depth_test = r.depth_test;
  }
  // A read-only attachment keeps its contents whatever the pipeline says.
  const bool depth_write =
      r.depth_write && !(pass_read_only_ & rhi::kAspectDepth);
  if (depth_write != raster_.depth_write) {
    glDepthMask(depth_write ? GL_TRUE : GL_FALSE);
    raster_.depth_write = depth_write;
  }
  if (r.depth_func != raster_.depth_func) {
    glDepthFunc(r.depth_func);
    raster_.depth_func = r.depth_func;
  }
  if (r.stencil_test != raster_.stencil_test) {
    Enable(GL_STENCIL_TEST, r.stencil_test);
    raster_.stencil_test = r.stencil_test;
  }
  StencilState front_state = r.front, back_state = r.back;
  if (pass_read_only_ & rhi::kAspectStencil)
    front_state.write_mask = back_state.write_mask = 0;
  if (!(front_state == raster_.front))
    ApplyStencil(GL_FRONT, front_state, raster_.front);
  if (!(back_state == raster_.back))
    ApplyStencil(GL_BACK, back_state, raster_.back);
  for (u32 i = 0; i < 8; i++)
    if (!(r.blend[i] == raster_.blend[i]))
      ApplyBlend(i, r.blend[i]);
}

void Replayer::Push(const CmdPush& c) {
  const GlPipeline* p = pipeline_;
  if (!p)
    return;
  const u8* data = Payload(c);
  for (u32 i = 0; i < p->push_count; i++) {
    const GlPipeline::Push& u = p->push[i];
    if (u.location < 0 || u.vec4_count * 16 > c.bytes)
      continue;
    const GLsizei n = static_cast<GLsizei>(u.vec4_count);
    switch (u.type) {
      case 'f':
        glProgramUniform4fv(p->program, u.location, n,
                            reinterpret_cast<const GLfloat*>(data));
        break;
      case 'i':
        glProgramUniform4iv(p->program, u.location, n,
                            reinterpret_cast<const GLint*>(data));
        break;
      default:
        glProgramUniform4uiv(p->program, u.location, n,
                             reinterpret_cast<const GLuint*>(data));
        break;
    }
  }
  if (p->push_ubo) {
    glNamedBufferSubData(push_ubo_, 0, c.bytes, data);
    glBindBufferBase(GL_UNIFORM_BUFFER, kPushUboSlot, push_ubo_);
  }
}

void Replayer::BufferToTexture(const CmdBufferTexture& c, bool upload) {
  const GlTexture& t = *c.texture;
  const rhi::TextureDesc& td = t.desc();
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(td.format);
  const GlFormat& f = ToGl(td.format);
  const GLenum binding = upload ? GL_PIXEL_UNPACK_BUFFER : GL_PIXEL_PACK_BUFFER;
  GLuint& bound = upload ? pixel_unpack_ : pixel_pack_;
  if (bound != c.buffer->name) {
    glBindBuffer(binding, c.buffer->name);
    bound = c.buffer->name;
  }
  const GLenum row_length = upload ? GL_UNPACK_ROW_LENGTH : GL_PACK_ROW_LENGTH;
  const GLenum image_height =
      upload ? GL_UNPACK_IMAGE_HEIGHT : GL_PACK_IMAGE_HEIGHT;
  if (fi.compressed) {
    const bool u = upload;
    glPixelStorei(u ? GL_UNPACK_COMPRESSED_BLOCK_WIDTH
                    : GL_PACK_COMPRESSED_BLOCK_WIDTH, 4);
    glPixelStorei(u ? GL_UNPACK_COMPRESSED_BLOCK_HEIGHT
                    : GL_PACK_COMPRESSED_BLOCK_HEIGHT, 4);
    glPixelStorei(u ? GL_UNPACK_COMPRESSED_BLOCK_DEPTH
                    : GL_PACK_COMPRESSED_BLOCK_DEPTH, 1);
    glPixelStorei(u ? GL_UNPACK_COMPRESSED_BLOCK_SIZE
                    : GL_PACK_COMPRESSED_BLOCK_SIZE, fi.bytes);
  }
  const auto* regions = reinterpret_cast<const rhi::BufferTextureCopy*>(
      Payload(c));
  for (u32 i = 0; i < c.count; i++) {
    const rhi::BufferTextureCopy& r = regions[i];
    const rhi::TextureRegion& g = r.region;
    GLenum format = f.format, type = f.type;
    if (fi.is_depth && fi.is_stencil) {
      // Buffer copies carry one plane.
      if (g.aspect & rhi::kAspectStencil) {
        format = GL_STENCIL_INDEX;
        type = GL_UNSIGNED_BYTE;
      } else {
        format = GL_DEPTH_COMPONENT;
        type = GL_FLOAT;
      }
    }
    glPixelStorei(row_length, static_cast<GLint>(r.row_length));
    glPixelStorei(image_height, static_cast<GLint>(r.image_height));
    GLint x = g.x, y = g.y, z = g.z;
    GLsizei w = g.width, h = g.height, d = g.depth;
    if (t.target == GL_TEXTURE_1D_ARRAY) {
      y = static_cast<GLint>(g.base_layer);
      h = static_cast<GLsizei>(g.layers);
    } else if (t.target == GL_TEXTURE_2D_ARRAY) {
      z = static_cast<GLint>(g.base_layer);
      d = static_cast<GLsizei>(g.layers);
    }
    const auto offset = reinterpret_cast<void*>(r.buffer_offset);
    const GLsizei room = static_cast<GLsizei>(
        base::Min<u64>(c.buffer->desc().size - r.buffer_offset, INT32_MAX));
    const GLint mip = static_cast<GLint>(g.mip);
    if (!upload) {
      if (fi.compressed)
        glGetCompressedTextureSubImage(t.name, mip, x, y, z, w, h, d, room,
                                       offset);
      else
        glGetTextureSubImage(t.name, mip, x, y, z, w, h, d, format, type, room,
                             offset);
      continue;
    }
    if (fi.compressed) {
      const GLsizei size = static_cast<GLsizei>(
          ((w + 3) / 4) * ((h + 3) / 4) * d * fi.bytes);
      if (t.target == GL_TEXTURE_2D)
        glCompressedTextureSubImage2D(t.name, mip, x, y, w, h, t.internal,
                                      size, offset);
      else
        glCompressedTextureSubImage3D(t.name, mip, x, y, z, w, h, d,
                                      t.internal, size, offset);
    } else if (t.target == GL_TEXTURE_1D) {
      glTextureSubImage1D(t.name, mip, x, w, format, type, offset);
    } else if (t.target == GL_TEXTURE_1D_ARRAY || t.target == GL_TEXTURE_2D) {
      glTextureSubImage2D(t.name, mip, x, y, w, h, format, type, offset);
    } else {
      glTextureSubImage3D(t.name, mip, x, y, z, w, h, d, format, type,
                          offset);
    }
  }
}

void Replayer::CopyTexture(const CmdCopyTexture& c) {
  auto place = [](const GlTexture& t, const rhi::TextureRegion& r, GLint* y,
                  GLint* z, GLsizei* h, GLsizei* d) {
    *y = r.y;
    *z = r.z;
    *h = static_cast<GLsizei>(r.height);
    *d = static_cast<GLsizei>(r.depth);
    if (t.target == GL_TEXTURE_1D_ARRAY) {
      *y = static_cast<GLint>(r.base_layer);
      *h = static_cast<GLsizei>(r.layers);
    } else if (t.target == GL_TEXTURE_2D_ARRAY) {
      *z = static_cast<GLint>(r.base_layer);
      *d = static_cast<GLsizei>(r.layers);
    }
  };
  GLint sy, sz, dy, dz;
  GLsizei h, d, dh, dd;
  place(*c.src, c.src_region, &sy, &sz, &h, &d);
  place(*c.dst, c.dst_region, &dy, &dz, &dh, &dd);
  glCopyImageSubData(c.src->name, c.src->target,
                     static_cast<GLint>(c.src_region.mip), c.src_region.x, sy,
                     sz, c.dst->name, c.dst->target,
                     static_cast<GLint>(c.dst_region.mip), c.dst_region.x, dy,
                     dz, static_cast<GLsizei>(c.src_region.width), h, d);
}

void Replayer::AttachScratch(GLuint fbo,
                             GLenum attachment,
                             const GlTexture* t,
                             u32 mip,
                             u32 layer) {
  if (t->target == GL_TEXTURE_2D || t->target == GL_TEXTURE_1D)
    glNamedFramebufferTexture(fbo, attachment, t->name,
                              static_cast<GLint>(mip));
  else
    glNamedFramebufferTextureLayer(fbo, attachment, t->name,
                                   static_cast<GLint>(mip),
                                   static_cast<GLint>(layer));
}

void Replayer::DetachScratch(GLuint fbo) {
  for (GLenum a : {GL_COLOR_ATTACHMENT0, GL_DEPTH_STENCIL_ATTACHMENT})
    glNamedFramebufferTexture(fbo, a, 0, 0);
}

void Replayer::Blit(const CmdBlit& c) {
  const u8 planes = Planes(c.src->desc().format, c.src_region.aspect);
  GLbitfield mask = 0;
  if (planes & rhi::kAspectColor)
    mask |= GL_COLOR_BUFFER_BIT;
  if (planes & rhi::kAspectDepth)
    mask |= GL_DEPTH_BUFFER_BIT;
  if (planes & rhi::kAspectStencil)
    mask |= GL_STENCIL_BUFFER_BIT;
  const GLenum point = (planes & rhi::kAspectColor) ? GL_COLOR_ATTACHMENT0
                                                    : DepthAttachment(planes);
  const bool is_3d = c.src->target == GL_TEXTURE_3D;
  const u32 slices =
      is_3d ? base::Min(c.src_region.depth, c.dst_region.depth)
            : base::Min(c.src_region.layers, c.dst_region.layers);
  SetScissorTest(false);
  ResetMasks();
  const rhi::TextureRegion& s = c.src_region;
  const rhi::TextureRegion& d = c.dst_region;
  for (u32 i = 0; i < slices; i++) {
    const u32 src_layer = is_3d ? s.z + i : s.base_layer + i;
    const u32 dst_layer =
        c.dst->target == GL_TEXTURE_3D ? d.z + i : d.base_layer + i;
    AttachScratch(scratch_read_, point, c.src, s.mip, src_layer);
    AttachScratch(scratch_draw_, point, c.dst, d.mip, dst_layer);
    glBlitNamedFramebuffer(
        scratch_read_, scratch_draw_, s.x, s.y,
        s.x + static_cast<GLint>(s.width), s.y + static_cast<GLint>(s.height),
        d.x, d.y, d.x + static_cast<GLint>(d.width),
        d.y + static_cast<GLint>(d.height), mask,
        mask == GL_COLOR_BUFFER_BIT ? c.filter : GL_NEAREST);
  }
  DetachScratch(scratch_read_);
  DetachScratch(scratch_draw_);
}

void Replayer::ClearTexture(const CmdClearTexture& c) {
  const GlTexture& t = *c.texture;
  const rhi::TextureDesc& td = t.desc();
  const rhi::FormatInfo& fi = rhi::GetFormatInfo(td.format);
  if (fi.compressed)
    return;
  const GlFormat& f = ToGl(td.format);
  GLenum format = GL_RGBA, type = GL_FLOAT;
  rhi::ClearColor value = c.color;
  if (fi.is_integer) {
    format = GL_RGBA_INTEGER;
    type = f.type == GL_BYTE || f.type == GL_SHORT || f.type == GL_INT
               ? GL_INT
               : GL_UNSIGNED_INT;
  } else if (fi.is_srgb) {
    // Texture clears store the value as given; Vulkan's are linear.
    for (u32 i = 0; i < 3; i++)
      value.f[i] = EncodeSrgb(value.f[i]);
  }
  for (u32 m = c.range.base_mip; m < c.range.base_mip + c.range.mips; m++) {
    GLint y = 0, z = 0;
    GLsizei h = static_cast<GLsizei>(MipExtent(td.height, m));
    GLsizei d = static_cast<GLsizei>(MipExtent(td.depth, m));
    if (t.target == GL_TEXTURE_1D_ARRAY) {
      y = static_cast<GLint>(c.range.base_layer);
      h = static_cast<GLsizei>(c.range.layers);
    } else if (t.target == GL_TEXTURE_2D_ARRAY) {
      z = static_cast<GLint>(c.range.base_layer);
      d = static_cast<GLsizei>(c.range.layers);
    } else if (t.target != GL_TEXTURE_3D) {
      d = 1;
    }
    glClearTexSubImage(t.name, static_cast<GLint>(m), 0, y, z,
                       static_cast<GLsizei>(MipExtent(td.width, m)), h, d,
                       format, type, &value);
  }
}

void Replayer::ClearDepthStencil(const CmdClearDepthStencil& c) {
  const GlTexture& t = *c.texture;
  const rhi::TextureDesc& td = t.desc();
  const u8 planes = Planes(td.format, c.range.aspect);
  if (planes & rhi::kAspectColor)
    return;
  ResetMasks();
  SetScissorTest(false);
  const GLenum point = DepthAttachment(planes);
  for (u32 m = c.range.base_mip; m < c.range.base_mip + c.range.mips; m++)
    for (u32 l = c.range.base_layer; l < c.range.base_layer + c.range.layers;
         l++) {
      AttachScratch(scratch_draw_, point, &t, m, l);
      ClearDepthStencilFbo(scratch_draw_, planes, c.depth, c.stencil);
    }
  DetachScratch(scratch_draw_);
}

void Replayer::Execute(const GlCommandList& list) {
  pipeline_ = nullptr;
  const CommandStream& stream = list.stream();
  for (const u8* p = stream.begin(); p < stream.end();
       p += reinterpret_cast<const Cmd*>(p)->size) {
    switch (reinterpret_cast<const Cmd*>(p)->op) {
      case Op::kBeginPass:
        BeginPass(As<CmdBeginPass>(p));
        break;
      case Op::kEndPass:
        EndPass(As<CmdEndPass>(p));
        break;
      case Op::kPipeline:
        pipeline_ = As<CmdPipeline>(p).pipeline;
        break;
      case Op::kBindBuffer: {
        const auto& c = As<CmdBindBuffer>(p);
        if (c.name)
          glBindBufferRange(c.target, c.slot, c.name,
                            static_cast<GLintptr>(c.offset),
                            static_cast<GLsizeiptr>(c.size));
        else
          glBindBufferBase(c.target, c.slot, 0);
        break;
      }
      case Op::kBindTexture: {
        const auto& c = As<CmdBindTexture>(p);
        glBindTextureUnit(c.unit, c.name);
        glBindSampler(c.unit, c.sampler);
        break;
      }
      case Op::kBindImage: {
        const auto& c = As<CmdBindImage>(p);
        glBindImageTexture(c.unit, c.name, c.level, c.layered, c.layer,
                           GL_READ_WRITE, c.format);
        break;
      }
      case Op::kPush:
        Push(As<CmdPush>(p));
        break;
      case Op::kPointers: {
        const auto& c = As<CmdPointers>(p);
        if (pipeline_ && pipeline_->pointer_location >= 0)
          glProgramUniform4uiv(pipeline_->program, pipeline_->pointer_location,
                               static_cast<GLsizei>(c.count),
                               reinterpret_cast<const GLuint*>(Payload(c)));
        break;
      }
      case Op::kVertexBuffers: {
        const auto& c = As<CmdVertexBuffers>(p);
        glVertexArrayVertexBuffers(Vao(c.input), c.first,
                                   static_cast<GLsizei>(c.count), c.names,
                                   c.offsets, c.strides);
        break;
      }
      case Op::kIndexBuffer: {
        const auto& c = As<CmdIndexBuffer>(p);
        glVertexArrayElementBuffer(Vao(c.input), c.name);
        break;
      }
      case Op::kViewport: {
        const auto& c = As<CmdViewport>(p);
        // A negative height rasterises y-up: GL takes that as an upper-left
        // clip origin over the same, positive, rectangle.
        const bool upper = c.height < 0;
        if (upper != upper_left_) {
          glClipControl(upper ? GL_UPPER_LEFT : GL_LOWER_LEFT,
                        GL_ZERO_TO_ONE);
          upper_left_ = upper;
        }
        glViewportIndexedf(0, c.x, upper ? c.y + c.height : c.y, c.width,
                           std::fabs(c.height));
        glDepthRangeIndexed(0, c.min_depth, c.max_depth);
        break;
      }
      case Op::kScissor: {
        const auto& c = As<CmdScissor>(p);
        scissor_[0] = c.x;
        scissor_[1] = c.y;
        scissor_[2] = static_cast<i32>(c.width);
        scissor_[3] = static_cast<i32>(c.height);
        break;
      }
      case Op::kBlendConstants: {
        const auto& c = As<CmdBlendConstants>(p);
        glBlendColor(c.rgba[0], c.rgba[1], c.rgba[2], c.rgba[3]);
        break;
      }
      case Op::kDraw: {
        const auto& c = As<CmdDraw>(p);
        ApplyGraphics();
        glDrawArraysInstancedBaseInstance(
            pipeline_->mode, static_cast<GLint>(c.first_vertex),
            static_cast<GLsizei>(c.vertex_count),
            static_cast<GLsizei>(c.instance_count), c.first_instance);
        break;
      }
      case Op::kDrawIndexed: {
        const auto& c = As<CmdDrawIndexed>(p);
        ApplyGraphics();
        glDrawElementsInstancedBaseVertexBaseInstance(
            pipeline_->mode, static_cast<GLsizei>(c.index_count), c.type,
            reinterpret_cast<const void*>(c.offset),
            static_cast<GLsizei>(c.instance_count), c.vertex_offset,
            c.first_instance);
        break;
      }
      case Op::kDispatch: {
        const auto& c = As<CmdDispatch>(p);
        UseProgram(pipeline_->program);
        if (pipeline_->base_location >= 0 &&
            std::memcmp(pipeline_->base, c.base, sizeof(c.base))) {
          glProgramUniform3ui(pipeline_->program, pipeline_->base_location,
                              c.base[0], c.base[1], c.base[2]);
          std::memcpy(pipeline_->base, c.base, sizeof(c.base));
        }
        glDispatchCompute(c.x, c.y, c.z);
        break;
      }
      case Op::kClearAttachment: {
        const auto& c = As<CmdClearAttachment>(p);
        PrepareClear(c.x, c.y, c.width, c.height);
        if (c.attachment == ~0u)
          ClearDepthStencilFbo(fbo_, c.aspect, c.depth, c.stencil);
        else
          ClearFramebuffer(fbo_, c.attachment, c.kind, c.color);
        break;
      }
      case Op::kCopyBuffer: {
        const auto& c = As<CmdCopyBuffer>(p);
        glCopyNamedBufferSubData(c.src, c.dst,
                                 static_cast<GLintptr>(c.src_offset),
                                 static_cast<GLintptr>(c.dst_offset),
                                 static_cast<GLsizeiptr>(c.bytes));
        break;
      }
      case Op::kBufferToTexture:
        BufferToTexture(As<CmdBufferToTexture>(p), true);
        break;
      case Op::kTextureToBuffer:
        BufferToTexture(As<CmdTextureToBuffer>(p), false);
        break;
      case Op::kCopyTexture:
        CopyTexture(As<CmdCopyTexture>(p));
        break;
      case Op::kBlit:
        Blit(As<CmdBlit>(p));
        break;
      case Op::kClearTexture:
        ClearTexture(As<CmdClearTexture>(p));
        break;
      case Op::kClearDepthStencil:
        ClearDepthStencil(As<CmdClearDepthStencil>(p));
        break;
      case Op::kFillBuffer: {
        const auto& c = As<CmdFillBuffer>(p);
        glClearNamedBufferSubData(c.name, GL_R32UI,
                                  static_cast<GLintptr>(c.offset),
                                  static_cast<GLsizeiptr>(c.bytes),
                                  GL_RED_INTEGER, GL_UNSIGNED_INT, &c.value);
        break;
      }
      case Op::kUpdateBuffer: {
        const auto& c = As<CmdUpdateBuffer>(p);
        glNamedBufferSubData(c.name, static_cast<GLintptr>(c.offset),
                             static_cast<GLsizeiptr>(c.bytes), Payload(c));
        break;
      }
      case Op::kBarrier:
        glMemoryBarrier(As<CmdBarrier>(p).bits);
        break;
      case Op::kResetTimestamps: {
        const auto& c = As<CmdResetTimestamps>(p);
        glClearNamedBufferSubData(c.pool->buffer, GL_R32UI,
                                  static_cast<GLintptr>(c.first) * 8,
                                  static_cast<GLsizeiptr>(c.count) * 8,
                                  GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        break;
      }
      case Op::kWriteTimestamp: {
        const auto& c = As<CmdWriteTimestamp>(p);
        GlTimestampPool& pool = *c.pool;
        if (pool.queries.empty()) {
          pool.queries.resize(pool.count);
          glCreateQueries(GL_TIMESTAMP, static_cast<GLsizei>(pool.count),
                          pool.queries.data());
        }
        const GLuint q = pool.queries[c.first];
        glQueryCounter(q, GL_TIMESTAMP);
        glGetQueryBufferObjectui64v(q, pool.buffer, GL_QUERY_RESULT,
                                    static_cast<GLintptr>(c.first) * 8);
        break;
      }
      case Op::kPushLabel:
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                         reinterpret_cast<const char*>(
                             Payload(As<CmdPushLabel>(p))));
        break;
      case Op::kPopLabel:
        glPopDebugGroup();
        break;
      case Op::kInsertLabel:
        glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER,
                             0, GL_DEBUG_SEVERITY_NOTIFICATION, -1,
                             reinterpret_cast<const char*>(
                                 Payload(As<CmdInsertLabel>(p))));
        break;
    }
  }
}

}  // namespace gpu::opengl
