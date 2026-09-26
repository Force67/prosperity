/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/opengl/gl_context.h"

#include <epoxy/gl.h>
#include <pthread.h>

#include <algorithm>
#include <cstring>

namespace gpu::opengl {

namespace {

// Scores a display by what it renders on; < 0 when it has no GL 4.6 core.
int Probe(EGLDisplay display, const char* filter, std::string* renderer) {
  EGLint major = 0, minor = 0;
  if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor))
    return -1;
  EGLContext ctx = CreateGlContext(display, EGL_NO_CONTEXT, false);
  if (ctx == EGL_NO_CONTEXT)
    return -1;
  int score = -1;
  if (eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
    const char* name =
        reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    *renderer = name ? name : "";
    score = 2;
    if (strstr(renderer->c_str(), "llvmpipe") ||
        strstr(renderer->c_str(), "softpipe") ||
        strstr(renderer->c_str(), "SwiftShader"))
      score = 0;
    if (filter && *filter && strstr(renderer->c_str(), filter))
      score = 100;
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  }
  eglDestroyContext(display, ctx);
  return score;
}

}  // namespace

bool OpenEglDevice(const char* filter, EglDevice* out) {
  std::vector<EGLDisplay> candidates;
  if (epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_EXT_device_enumeration") &&
      epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_EXT_platform_device")) {
    EGLDeviceEXT devices[16];
    EGLint n = 0;
    if (eglQueryDevicesEXT(16, devices, &n))
      for (EGLint i = 0; i < n; i++)
        candidates.push_back(eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                                      devices[i], nullptr));
  }
  if (epoxy_has_egl_extension(EGL_NO_DISPLAY, "EGL_MESA_platform_surfaceless"))
    candidates.push_back(eglGetPlatformDisplayEXT(
        EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr));
  candidates.push_back(eglGetDisplay(EGL_DEFAULT_DISPLAY));

  const bool filtered = filter && *filter;
  int best = -1;
  for (EGLDisplay d : candidates) {
    std::string renderer;
    const int score = Probe(d, filter, &renderer);
    if (score > best) {
      best = score;
      out->display = d;
      out->renderer = renderer;
    }
    // Probing a display loads its driver: stop at the first good one.
    if (best == 100 || (!filtered && best == 2))
      break;
  }
  return best >= 0;
}

EGLContext CreateGlContext(EGLDisplay display, EGLContext share, bool debug) {
  if (!eglBindAPI(EGL_OPENGL_API))
    return EGL_NO_CONTEXT;
  const EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION,
                          4,
                          EGL_CONTEXT_MINOR_VERSION,
                          6,
                          EGL_CONTEXT_OPENGL_PROFILE_MASK,
                          EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                          EGL_CONTEXT_OPENGL_DEBUG,
                          debug ? EGL_TRUE : EGL_FALSE,
                          EGL_NONE};
  return eglCreateContext(display, EGL_NO_CONFIG_KHR, share, attrs);
}

GlslFeatures QueryGlslFeatures() {
  auto get = [](GLenum pname) {
    GLint v = 0;
    glGetIntegerv(pname, &v);
    return static_cast<u32>(std::max(v, 0));
  };
  GlslFeatures f;
  f.nv_barycentric_only =
      epoxy_has_gl_extension("GL_NV_fragment_shader_barycentric") &&
      !epoxy_has_gl_extension("GL_EXT_fragment_shader_barycentric");
  f.max_uniform_buffers = get(GL_MAX_UNIFORM_BUFFER_BINDINGS);
  f.max_storage_buffers = get(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS);
  f.max_textures = get(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
  f.max_images = get(GL_MAX_IMAGE_UNITS);
  f.max_stage_storage_buffers =
      std::min({get(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS),
                get(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS),
                get(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS),
                get(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)});
  f.max_vertex_attribs = get(GL_MAX_VERTEX_ATTRIBS);
  f.buffer_pointers = epoxy_has_gl_extension("GL_NV_shader_buffer_load") &&
                      epoxy_has_gl_extension("GL_NV_gpu_shader5");
  f.khr_subgroup = epoxy_has_gl_extension("GL_KHR_shader_subgroup");
  return f;
}

void GlWorker::Start(EGLDisplay display,
                     std::vector<EGLContext> contexts,
                     const char* name,
                     std::function<void()> init) {
  display_ = display;
  contexts_ = std::move(contexts);
  init_ = std::move(init);
  stop_ = false;
  for (EGLContext c : contexts_)
    threads_.emplace_back([this, c, name] { Loop(c, name); });
}

void GlWorker::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread& t : threads_)
    t.join();
  threads_.clear();
  for (EGLContext c : contexts_)
    eglDestroyContext(display_, c);
  contexts_.clear();
}

void GlWorker::Post(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push_back(std::move(task));
  }
  cv_.notify_one();
}

void GlWorker::Run(const std::function<void()>& task) {
  for (const std::thread& t : threads_)
    if (t.get_id() == std::this_thread::get_id()) {
      task();
      return;
    }
  std::mutex m;
  std::condition_variable cv;
  bool done = false;
  Post([&] {
    task();
    std::lock_guard<std::mutex> lock(m);
    done = true;
    cv.notify_one();
  });
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&] { return done; });
}

void GlWorker::Loop(EGLContext context, const char* name) {
  pthread_setname_np(pthread_self(), name);
  eglBindAPI(EGL_OPENGL_API);
  eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
  if (init_)
    init_();
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
      if (tasks_.empty())
        break;
      task = std::move(tasks_.front());
      tasks_.pop_front();
    }
    task();
  }
  eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglReleaseThread();
}

}  // namespace gpu::opengl
