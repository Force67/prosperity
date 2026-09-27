/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Headless desktop GL 4.6 core contexts over EGL, and the threads that keep
// one current.

#include <epoxy/egl.h>

#include <base/memory/unique_pointer.h>
#include <base/threading/thread.h>

#include "base/arch.h"
#include "gpu/opengl/gl_shader_lowering.h"
#include <base/containers/deque.h>
#include <base/containers/vector.h>
#include <base/functional/function.h>
#include <base/strings/xstring.h>
#include <base/threading/condition_variable.h>
#include <base/threading/mutex.h>

namespace gpu::opengl {

struct EglDevice {
  EGLDisplay display = EGL_NO_DISPLAY;
  base::String renderer;  // GL_RENDERER
};

// The best EGL device with GL 4.6 core: a hardware GPU over a software
// rasteriser, or the one whose GL_RENDERER contains `filter`.
bool OpenEglDevice(const char* filter, EglDevice* out);

// A surfaceless GL 4.6 core context sharing objects with `share`.
EGLContext CreateGlContext(EGLDisplay display, EGLContext share, bool debug);

// What the lowering may use, from the current context.
GlslFeatures QueryGlslFeatures();

// Threads that each keep their own context current and drain one FIFO of
// tasks between them.
class GlWorker {
 public:
  GlWorker() = default;
  GlWorker(const GlWorker&) = delete;
  GlWorker& operator=(const GlWorker&) = delete;
  ~GlWorker() { Stop(); }

  // One thread per context; the worker owns the contexts from here on.
  // `init` runs once on each thread once its context is current.
  void Start(EGLDisplay display,
             base::Vector<EGLContext> contexts,
             const char* name,
             base::Function<void()> init = {});
  void Stop();

  void Post(base::Function<void()> task);
  // Runs `task` on a worker thread and waits for it.
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
