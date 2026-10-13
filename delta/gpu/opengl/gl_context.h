/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <epoxy/egl.h>

#include "base/memory/unique_pointer.h"
#include "base/threading/thread.h"

#include "base/arch.h"
#include "base/containers/deque.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/mutex.h"
#include "gpu/opengl/gl_shader_lowering.h"

namespace gpu::opengl {

struct EglDevice {
  EGLDisplay display = EGL_NO_DISPLAY;
  base::String renderer;
};

bool OpenEglDevice(const char* filter, EglDevice* out);

EGLContext CreateGlContext(EGLDisplay display, EGLContext share, bool debug);

GlslFeatures QueryGlslFeatures();

class GlWorker {
 public:
  GlWorker() = default;
  GlWorker(const GlWorker&) = delete;
  GlWorker& operator=(const GlWorker&) = delete;
  ~GlWorker() { Stop(); }

  void Start(EGLDisplay display,
             base::Vector<EGLContext> contexts,
             const char* name,
             base::Function<void()> init = {});
  void Stop();

  void Post(base::Function<void()> task);
  void Run(const base::Function<void()>& task);

 private:
  void Loop(EGLContext context, const char* name);

  EGLDisplay display_ = EGL_NO_DISPLAY;
  base::Vector<EGLContext> contexts_;
  base::Vector<base::UniquePointer<base::Thread>> threads_;
  base::Function<void()> init_;
  base::Mutex mutex_;
  base::ConditionVariable cv_;
  base::SimpleDeque<base::Function<void()>> tasks_;
  bool stop_ = false;
};

}  // namespace gpu::opengl
