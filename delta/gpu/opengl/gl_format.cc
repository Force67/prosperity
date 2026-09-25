/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/opengl/gl_rhi_internal.h"

namespace gpu::opengl {

namespace {

constexpr GLenum kUB = GL_UNSIGNED_BYTE, kB = GL_BYTE;
constexpr GLenum kUS = GL_UNSIGNED_SHORT, kS = GL_SHORT;
constexpr GLenum kUI = GL_UNSIGNED_INT, kI = GL_INT;
constexpr GLenum kF = GL_FLOAT, kH = GL_HALF_FLOAT;

// Texture: internal format, transfer format and type. Vertex: size, type
// and whether it is normalized; an integer texture format is an integer
// attribute, and a scaled one converts to float without normalizing.
constexpr GlFormat Tex(GLenum internal, GLenum format, GLenum type) {
  return {internal, format, type, 0, 0, false};
}
constexpr GlFormat Both(GLenum internal,
                        GLenum format,
                        GLenum type,
                        GLint size,
                        bool normalized) {
  return {internal, format, type, size, type, normalized};
}
constexpr GlFormat Vtx(GLint size, GLenum type) {
  return {0, 0, 0, size, type, false};
}

// In rhi::Format order.
constexpr GlFormat kFormats[] = {
    {},
    Both(GL_R8, GL_RED, kUB, 1, true),
    Both(GL_R8_SNORM, GL_RED, kB, 1, true),
    Both(GL_R8UI, GL_RED_INTEGER, kUB, 1, false),
    Both(GL_R8I, GL_RED_INTEGER, kB, 1, false),
    Vtx(1, kUB),
    Vtx(1, kB),
    Both(GL_RG8, GL_RG, kUB, 2, true),
    Both(GL_RG8_SNORM, GL_RG, kB, 2, true),
    Both(GL_RG8UI, GL_RG_INTEGER, kUB, 2, false),
    Both(GL_RG8I, GL_RG_INTEGER, kB, 2, false),
    Vtx(2, kUB),
    Vtx(2, kB),
    Both(GL_RGBA8, GL_RGBA, kUB, 4, true),
    Both(GL_RGBA8_SNORM, GL_RGBA, kB, 4, true),
    Both(GL_RGBA8UI, GL_RGBA_INTEGER, kUB, 4, false),
    Both(GL_RGBA8I, GL_RGBA_INTEGER, kB, 4, false),
    Vtx(4, kUB),
    Vtx(4, kB),
    Tex(GL_SRGB8_ALPHA8, GL_RGBA, kUB),
    // Stored as RGBA; transfers swap to and from the BGRA memory order.
    Both(GL_RGBA8, GL_BGRA, kUB, GL_BGRA, true),
    Tex(GL_SRGB8_ALPHA8, GL_BGRA, kUB),
    Both(GL_R16, GL_RED, kUS, 1, true),
    Both(GL_R16_SNORM, GL_RED, kS, 1, true),
    Both(GL_R16UI, GL_RED_INTEGER, kUS, 1, false),
    Both(GL_R16I, GL_RED_INTEGER, kS, 1, false),
    Vtx(1, kUS),
    Vtx(1, kS),
    Both(GL_R16F, GL_RED, kH, 1, false),
    Both(GL_RG16, GL_RG, kUS, 2, true),
    Both(GL_RG16_SNORM, GL_RG, kS, 2, true),
    Both(GL_RG16UI, GL_RG_INTEGER, kUS, 2, false),
    Both(GL_RG16I, GL_RG_INTEGER, kS, 2, false),
    Vtx(2, kUS),
    Vtx(2, kS),
    Both(GL_RG16F, GL_RG, kH, 2, false),
    Both(GL_RGBA16, GL_RGBA, kUS, 4, true),
    Both(GL_RGBA16_SNORM, GL_RGBA, kS, 4, true),
    Both(GL_RGBA16UI, GL_RGBA_INTEGER, kUS, 4, false),
    Both(GL_RGBA16I, GL_RGBA_INTEGER, kS, 4, false),
    Vtx(4, kUS),
    Vtx(4, kS),
    Both(GL_RGBA16F, GL_RGBA, kH, 4, false),
    Both(GL_R32UI, GL_RED_INTEGER, kUI, 1, false),
    Both(GL_R32I, GL_RED_INTEGER, kI, 1, false),
    Both(GL_R32F, GL_RED, kF, 1, false),
    Both(GL_RG32UI, GL_RG_INTEGER, kUI, 2, false),
    Both(GL_RG32I, GL_RG_INTEGER, kI, 2, false),
    Both(GL_RG32F, GL_RG, kF, 2, false),
    Both(GL_RGB32F, GL_RGB, kF, 3, false),
    Both(GL_RGBA32UI, GL_RGBA_INTEGER, kUI, 4, false),
    Both(GL_RGBA32I, GL_RGBA_INTEGER, kI, 4, false),
    Both(GL_RGBA32F, GL_RGBA, kF, 4, false),
    Both(GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, 3,
         false),
    Both(GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 4, true),
    Both(GL_RGB10_A2, GL_BGRA, GL_UNSIGNED_INT_2_10_10_10_REV, GL_BGRA, true),
    Tex(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 0, 0),
    Tex(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, 0, 0),
    Tex(GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, 0, 0),
    Tex(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, 0, 0),
    Tex(GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 0, 0),
    Tex(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, 0, 0),
    Tex(GL_COMPRESSED_RED_RGTC1, 0, 0),
    Tex(GL_COMPRESSED_SIGNED_RED_RGTC1, 0, 0),
    Tex(GL_COMPRESSED_RG_RGTC2, 0, 0),
    Tex(GL_COMPRESSED_SIGNED_RG_RGTC2, 0, 0),
    Tex(GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, 0, 0),
    Tex(GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, 0, 0),
    Tex(GL_COMPRESSED_RGBA_BPTC_UNORM, 0, 0),
    Tex(GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, 0, 0),
    Tex(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, kF),
    Tex(GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL,
        GL_FLOAT_32_UNSIGNED_INT_24_8_REV),
    Tex(GL_STENCIL_INDEX8, GL_STENCIL_INDEX, kUB),
};

static_assert(sizeof(kFormats) / sizeof(kFormats[0]) ==
                  static_cast<size_t>(rhi::Format::kCount),
              "every rhi::Format needs an entry");

}  // namespace

const GlFormat& ToGl(rhi::Format format) {
  const size_t i = static_cast<size_t>(format);
  return kFormats[i < static_cast<size_t>(rhi::Format::kCount) ? i : 0];
}

}  // namespace gpu::opengl
