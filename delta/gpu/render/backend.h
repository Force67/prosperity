#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which graphics APIs this build can run on, and the one place that creates a
 * device for one of them. Nothing else in the renderer names a backend.
 */

#include <memory>
#include <vector>

#include "gpu/rhi/device.h"

namespace gpu::render {

struct BackendOptions {
  const char* gpu_filter = nullptr;  // adapter name substring
  bool debug_labels = false;
  const char* validation_layer = nullptr;
  bool sync_validation = false;
  bool checkpoints = false;
  // Directory for driver pipeline caches; null disables them.
  const char* cache_dir = nullptr;
};

std::vector<rhi::Backend> CompiledBackends();

// Null when the backend is not compiled in or has no usable device. Call
// from a plain host thread.
std::unique_ptr<rhi::Device> CreateBackendDevice(
    rhi::Backend backend,
    const BackendOptions& options = {});

}  // namespace gpu::render
