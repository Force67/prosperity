#pragma once

#include "base/arch.h"

namespace ui {

struct ShaderActivity {
  u32 active = 0;
  u64 active_since_ns = 0;
  u64 last_slow_compile_ns = 0;
};

ShaderActivity GetShaderActivity();

// Tracks compiler work across the guest, optimizer, and driver threads.
class ShaderCompilation {
 public:
  ShaderCompilation();
  ~ShaderCompilation();
  ShaderCompilation(const ShaderCompilation&) = delete;
  ShaderCompilation& operator=(const ShaderCompilation&) = delete;

 private:
  u64 started_ns_;
};

}  // namespace ui
