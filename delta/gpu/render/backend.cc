/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/backend.h"
#include "gpu/render/renderer.h"

#include "gpu/vulkan/vk_rhi.h"
#if DELTA_GPU_OPENGL
#include "gpu/opengl/gl_rhi.h"
#endif
#if DELTA_HAVE_D3D12
#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "gpu/d3d12/d3d12_rhi.h"
#endif

namespace gpu::render {

base::Vector<rhi::Backend> CompiledBackends() {
  return {
      rhi::Backend::kVulkan,
#if DELTA_GPU_OPENGL
      rhi::Backend::kOpenGL,
#endif
#if DELTA_HAVE_D3D12
      rhi::Backend::kD3D12,
#endif
  };
}

base::Vector<base::String> GraphicsBackends() {
  base::Vector<base::String> names;
  for (const auto backend : CompiledBackends())
    names.emplace_back(rhi::BackendName(backend));
  return names;
}

base::UniquePointer<rhi::Device> CreateBackendDevice(
    rhi::Backend backend,
    const BackendOptions& options) {
  const base::String cache =
      options.cache_dir ? base::String(options.cache_dir) + "/pipeline.bin"
                        : base::String();
  switch (backend) {
    case rhi::Backend::kVulkan: {
      vk::VulkanOptions vo;
      vo.gpu_filter = options.gpu_filter;
      vo.debug_utils = options.debug_labels;
      vo.validation_layer =
          options.validation ? "VK_LAYER_KHRONOS_validation" : nullptr;
      vo.on_message = options.on_message;
      vo.sync_validation = options.sync_validation;
      vo.checkpoints = options.checkpoints;
      vo.pipeline_cache_path = cache.empty() ? nullptr : cache.c_str();
      return vk::CreateVulkanDevice(vo);
    }
#if DELTA_GPU_OPENGL
    case rhi::Backend::kOpenGL: {
      opengl::OpenGLOptions go;
      go.gpu_filter = options.gpu_filter;
      go.debug = options.debug_labels;
      return opengl::CreateOpenGLDevice(go);
    }
#endif
#if DELTA_HAVE_D3D12
    case rhi::Backend::kD3D12: {
      d3d12::D3D12Options d3;
      d3.gpu_filter = options.gpu_filter;
      d3.debug_layer = options.validation;
      d3.debug_labels = options.debug_labels;
      return d3d12::CreateD3D12Device(d3);
    }
#endif
    default:
      return nullptr;
  }
}

}  // namespace gpu::render
