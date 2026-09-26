/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The OpenGL 4.6 core implementation of rhi::Device, on a headless EGL
// context.

#include <memory>

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::opengl {

struct OpenGLOptions {
  // Substring of GL_RENDERER to prefer over the default choice.
  const char* gpu_filter = nullptr;
  // Debug contexts: driver messages are logged, labels and names reach
  // capture tools.
  bool debug = false;
  // Threads compiling pipelines, each with its own shared context.
  u32 compile_threads = 2;
};

// Null when no GL 4.6 core device is available.
std::unique_ptr<rhi::Device> CreateOpenGLDevice(const OpenGLOptions& options);

}  // namespace gpu::opengl
