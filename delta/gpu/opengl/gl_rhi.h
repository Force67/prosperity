/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/memory/unique_pointer.h"
#include "gpu/rhi/device.h"

namespace gpu::opengl {

struct OpenGLOptions {
  const char* gpu_filter = nullptr;
  bool debug = false;
  u32 compile_threads = 2;
};

base::UniquePointer<rhi::Device> CreateOpenGLDevice(
    const OpenGLOptions& options);

}  // namespace gpu::opengl
