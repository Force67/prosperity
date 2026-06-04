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
  int32_t surfaceW = 0, surfaceH = 0;
};

void bootOnce(AppState *s) {
  if (s->booted)
    return;
  s->booted = true;
  s->core = new deltaCore();
  s->core->init();
  base::String pkg = s->dataDir;
  pkg += "/game.pkg";
  LOGI("booting pkg %s", pkg.c_str());
  s->core->boot(pkg);  // mounts pkg, runs the guest on a detached thread
}

// Touch -> DS4. Left half drives the move/left-stick, right half the aim/right-
// stick: 8-way from the touch position relative to that half's centre. A short
// tap (no significant drag) also presses cross so menus advance.
void applyTouch(AppState *s, AInputEvent *ev) {
  gfx::PadKeys k;  // neutral
  int n = AMotionEvent_getPointerCount(ev);
  int action = AMotionEvent_getAction(ev) & AMOTION_EVENT_ACTION_MASK;
  bool anyDown = action != AMOTION_EVENT_ACTION_UP &&
                 action != AMOTION_EVENT_ACTION_CANCEL;
  float W = s->surfaceW ? s->surfaceW : 1920.0f;
  float H = s->surfaceH ? s->surfaceH : 1080.0f;
  for (int i = 0; anyDown && i < n; i++) {
    float x = AMotionEvent_getX(ev, i);
    float y = AMotionEvent_getY(ev, i);
    bool right = x > W * 0.5f;
    float cx = right ? W * 0.75f : W * 0.25f;
    float cy = H * 0.5f;
    float dx = x - cx, dy = y - cy;
    float dz = W * 0.06f;  // dead zone
    int hx = dx > dz ? 1 : (dx < -dz ? -1 : 0);
    int hy = dy > dz ? 1 : (dy < -dz ? -1 : 0);
    if (right) {
      k.rx = hx < 0 ? 0 : (hx > 0 ? 255 : 128);
      k.ry = hy < 0 ? 0 : (hy > 0 ? 255 : 128);
    } else {
      k.left = hx < 0; k.right = hx > 0; k.up = hy < 0; k.down = hy > 0;
      k.lx = hx < 0 ? 0 : (hx > 0 ? 255 : 128);
      k.ly = hy < 0 ? 0 : (hy > 0 ? 255 : 128);
      if (hx == 0 && hy == 0)
        k.cross = true;  // a centre tap confirms
    }
  }
  gfx::setAndroidPad(k);
}

void onCmd(android_app *app, int32_t cmd) {
  auto *s = static_cast<AppState *>(app->userData);
  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (app->window) {
      s->surfaceW = ANativeWindow_getWidth(app->window);
      s->surfaceH = ANativeWindow_getHeight(app->window);
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

int32_t onInput(android_app *app, AInputEvent *ev) {
  auto *s = static_cast<AppState *>(app->userData);
  if (AInputEvent_getType(ev) == AINPUT_EVENT_TYPE_MOTION) {
    applyTouch(s, ev);
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
