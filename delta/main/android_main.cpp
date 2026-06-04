/*
 * PS4Delta : PS4 emulation and research project
 *
 * Android NativeActivity entry (DELTA_ANDROID_APP). android_native_app_glue
 * calls android_main on its own thread; we redirect stdout/stderr to logcat,
 * point the loader at the app's external files dir (modules/ + game.pkg pushed
 * there by adb), bring up the emulator once the window exists, and feed touch
 * input to the gfx pad. Rendering reaches the screen via gfx_android.cpp.
 */
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>

#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/strings/xstring.h>

#include "cpu/cpu_backend.h"
#include "dcore.h"
#include "gfx/gfx.h"
#include "gfx/gfx_android.h"
#include <logger/logger.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "prosperity", __VA_ARGS__)

namespace {

// Pump stdout/stderr (the emulator logs through both) into logcat so a plain
// `adb logcat -s prosperity` shows the whole boot.
void *logPump(void *arg) {
  int fd = *static_cast<int *>(arg);
  char buf[2048];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof(buf) - 1)) > 0) {
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
      --n;
    buf[n] = '\0';
    __android_log_write(ANDROID_LOG_INFO, "prosperity", buf);
  }
  return nullptr;
}

void redirectStdioToLogcat() {
  static int pfd[2];
  if (::pipe(pfd) != 0)
    return;
  ::dup2(pfd[1], STDOUT_FILENO);
  ::dup2(pfd[1], STDERR_FILENO);
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  pthread_t t;
  pthread_create(&t, nullptr, logPump, &pfd[0]);
  pthread_detach(t);
}

struct AppState {
  deltaCore *core = nullptr;
  base::String dataDir;
  bool booted = false;
};

// The launcher Activity writes the absolute path of the pkg it wants to run to
// <dataDir>/boot.cfg (one line) before starting this NativeActivity. Read it;
// fall back to the legacy <dataDir>/game.pkg when no launcher config exists
// (e.g. data staged by hand over adb).
base::String resolveBootPkg(const base::String &dataDir) {
  base::String cfg = dataDir;
  cfg += "/boot.cfg";
  if (FILE *f = ::fopen(cfg.c_str(), "rb")) {
    char buf[4096];
    size_t n = ::fread(buf, 1, sizeof(buf) - 1, f);
    ::fclose(f);
    buf[n] = '\0';
    char *p = buf;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
      ++p;
    size_t len = std::strlen(p);
    while (len > 0 && (p[len - 1] == '\n' || p[len - 1] == '\r' ||
                       p[len - 1] == ' ' || p[len - 1] == '\t'))
      p[--len] = '\0';
    if (len > 0 && ::access(p, R_OK) == 0)
      return base::String(p);
    LOGI("boot.cfg path unusable ('%s'); falling back to game.pkg", p);
  }
  base::String fallback = dataDir;
  fallback += "/game.pkg";
  return fallback;
}

void bootOnce(AppState *s) {
  if (s->booted)
    return;
  s->booted = true;
  s->core = new deltaCore();
  s->core->init();
  base::String pkg = resolveBootPkg(s->dataDir);
  LOGI("booting pkg %s", pkg.c_str());
  s->core->boot(pkg);  // mounts pkg, runs the guest on a detached thread
}

// Forward the currently-down touch points (surface pixel coords) to gfx, which
// owns the on-screen control layout and maps them to the DS4 pad + overlay.
void forwardTouch(AInputEvent *ev) {
  int action = AMotionEvent_getAction(ev);
  int kind = action & AMOTION_EVENT_ACTION_MASK;
  if (kind == AMOTION_EVENT_ACTION_UP || kind == AMOTION_EVENT_ACTION_CANCEL) {
    gfx::setAndroidTouches(nullptr, 0);  // last finger up
    return;
  }
  int upIdx = -1;
  if (kind == AMOTION_EVENT_ACTION_POINTER_UP)
    upIdx = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >>
            AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
  int n = AMotionEvent_getPointerCount(ev);
  gfx::Touch pts[8];
  int c = 0;
  for (int i = 0; i < n && c < 8; i++) {
    if (i == upIdx)
      continue;  // this pointer is lifting
    pts[c].x = AMotionEvent_getX(ev, i);
    pts[c].y = AMotionEvent_getY(ev, i);
    c++;
  }
  gfx::setAndroidTouches(pts, c);
}

void onCmd(android_app *app, int32_t cmd) {
  auto *s = static_cast<AppState *>(app->userData);
  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (app->window) {
      gfx::setAndroidWindow(app->window);
      bootOnce(s);  // first window: start the emulator (renderer needs a window)
    }
    break;
  case APP_CMD_TERM_WINDOW:
    gfx::setAndroidWindow(nullptr);
    break;
  default:
    break;
  }
}

int32_t onInput(android_app *, AInputEvent *ev) {
  if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_MOTION) {
    forwardTouch(ev);
    return 1;
  }
  return 0;
}

}  // namespace

extern "C" void android_main(android_app *app) {
  redirectStdioToLogcat();
  cpu::earlyInit();  // reserve the FEX heap before any large guest mapping
  utl::createLogger(true);

  AppState state;
  const char *ext = app->activity->externalDataPath;
  state.dataDir = base::String(ext ? ext : "/data/local/tmp/prosperity");
  setenv("DELTA_DATA_DIR", state.dataDir.c_str(), 1);
  LOGI("data dir = %s", state.dataDir.c_str());

  app->userData = &state;
  app->onAppCmd = onCmd;
  app->onInputEvent = onInput;

  while (!app->destroyRequested) {
    int events;
    android_poll_source *source;
    if (ALooper_pollOnce(-1, nullptr, &events,
                         reinterpret_cast<void **>(&source)) < 0)
      continue;
    if (source)
      source->process(app, source);
  }
}
