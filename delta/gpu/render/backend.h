#pragma once

#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct BackendOptions {
  const char* gpu_filter = nullptr;
  bool debug_labels = false;
  bool validation = false;
  bool sync_validation = false;
  void (*on_message)(const char*,
                     const char*,
                     const char*,
                     const char*) = nullptr;
  bool checkpoints = false;
  const char* cache_dir = nullptr;
};

base::Vector<rhi::Backend> CompiledBackends();

base::UniquePointer<rhi::Device> CreateBackendDevice(
    rhi::Backend backend,
    const BackendOptions& options = {});

}  // namespace gpu::render
