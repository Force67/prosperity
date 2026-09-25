/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Headless desktop GL 4.6 core contexts over EGL, and the threads that keep
// one current.

#include <epoxy/egl.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "base/arch.h"

namespace gpu::opengl {

struct EglDevice {
  EGLDisplay display = EGL_NO_DISPLAY;
  std::string renderer;  // GL_RENDERER
};

// The best EGL device with GL 4.6 core: a hardware GPU over a software
// rasteriser, or the one whose GL_RENDERER contains `filter`.
bool OpenEglDevice(const char* filter, EglDevice* out);

// A surfaceless GL 4.6 core context sharing objects with `share`.
EGLContext CreateGlContext(EGLDisplay display, EGLContext share, bool debug);

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
             std::vector<EGLContext> contexts,
             const char* name,
             std::function<void()> init = {});
  void Stop();

  void Post(std::function<void()> task);
  // Runs `task` on a worker thread and waits for it.
  void Run(const std::function<void()>& task);

 private:
  void Loop(EGLContext context, const char* name);

  EGLDisplay display_ = EGL_NO_DISPLAY;
  std::vector<EGLContext> contexts_;
  std::vector<std::thread> threads_;
  std::function<void()> init_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool stop_ = false;
};

}  // namespace gpu::opengl
