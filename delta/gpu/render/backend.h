#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which graphics APIs this build can run on, and the one place that creates a
 * device for one of them. Nothing else in the renderer names a backend.
 */


#include "gpu/rhi/device.h"
#include <base/containers/vector.h>
#include <base/memory/unique_pointer.h>

namespace gpu::render {

struct BackendOptions {
  const char* gpu_filter = nullptr;  // adapter name substring
  bool debug_labels = false;
  bool validation = false;  // the API's validation layer, when it has one
  bool sync_validation = false;
  // Receives validation and driver messages: severity, message id, the
  // active label stack, text.
  void (*on_message)(const char*, const char*, const char*, const char*) =
      nullptr;
  bool checkpoints = false;
  // Directory for driver pipeline caches; null disables them.
  const char* cache_dir = nullptr;
};

base::Vector<rhi::Backend> CompiledBackends();

// Null when the backend is not compiled in or has no usable device. Call
// from a plain host thread.
base::UniquePointer<rhi::Device> CreateBackendDevice(
    rhi::Backend backend,
    const BackendOptions& options = {});

}  // namespace gpu::render
