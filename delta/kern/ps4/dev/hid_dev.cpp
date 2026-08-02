#include <base.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <utl/options.h>

#include "file_dev.h"
#include "hid_dev.h"

namespace {
// The opt-in that turns /dev/hid into a real device. Off (default) mirrors the
// real kernel where only system processes can use /dev/hid and games reach
// input through libScePad; commands soft-succeed. On, the guest read commands
// pull the host's evdev state and the write commands drive a uinput device.
DELTA_OPTION(bool, kHidPassthrough, "DELTA_HID_PASSTHROUGH", false);

// ---------------------------------------------------------------------------
// evdev/uinput ABI. The host sysroot ships no <linux/input.h>, so the handful
// of constants and layouts the passthrough needs are declared here. They are
// the stable Linux input ABI.
// ---------------------------------------------------------------------------

constexpr uint32_t kIocRead = 2, kIocWrite = 1;
constexpr uint32_t ioc(uint32_t dir, uint32_t type, uint32_t nr,
                       uint32_t size) {
  return (dir << 30) | (type << 8) | (nr) | (size << 16);
}
constexpr uint32_t ior(uint32_t type, uint32_t nr, uint32_t size) {
  return ioc(kIocRead, type, nr, size);
}
constexpr uint32_t iow(uint32_t type, uint32_t nr, uint32_t size) {
  return ioc(kIocWrite, type, nr, size);
}

constexpr uint32_t kEvKey = 0x01, kEvRel = 0x02, kEvAbs = 0x03, kEvLed = 0x11,
                   kEvFf = 0x15;
constexpr uint32_t kKeyMax = 0x2ff, kRelMax = 0x0b, kAbsMax = 0x3f;
// EVIOCGBIT(ev, len)
constexpr uint32_t eviocgbit(uint32_t ev, uint32_t len) {
  return ior('E', 0x20 + ev, len);
}
constexpr uint32_t kEviocGKey = eviocgbit(kEvKey, (kKeyMax + 8) / 8);
constexpr uint32_t kEviocGRel = eviocgbit(kEvRel, (kRelMax + 8) / 8);
constexpr uint32_t kEviocGAbs = eviocgbit(kEvAbs, (kAbsMax + 8) / 8);

struct inputEvent {
  uint64_t sec, usec;
  uint16_t type, code;
  int32_t value;
};
static_assert(sizeof(inputEvent) == 24, "input_event layout");

constexpr uint32_t kUiSetEvbit = iow('U', 100, 4);
constexpr uint32_t kUiSetKeybit = iow('U', 101, 4);
constexpr uint32_t kUiSetAbsbit = iow('U', 103, 4);
constexpr uint32_t kUiSetLedbit = iow('U', 105, 4);
constexpr uint32_t kUiSetFfbit = iow('U', 107, 4);
constexpr uint32_t kUiDevCreate = ioc(0, 'U', 1, 0);   // 0x5501
constexpr uint32_t kUiDevDestroy = ioc(0, 'U', 2, 0);  // 0x5502

struct uinputUserDev {
  char name[80];
  struct {
    uint16_t bustype, vendor, product, version;
  } id;
  uint32_t ffEffectsMax;
  int32_t absmax[kAbsMax + 1];
  int32_t absmin[kAbsMax + 1];
  int32_t absfuzz[kAbsMax + 1];
  int32_t absflat[kAbsMax + 1];
};

constexpr uint32_t kFfRumble = 0x50;
// UI_BEGIN_FF_UPLOAD/UI_END_FF_UPLOAD carry two ff_effect (48 B each) plus an
// id + retval pair.
struct ffEnvelope {
  uint16_t attackLength, attackLevel, fadeLength, fadeLevel;
};
struct ffTrigger {
  uint16_t button, interval;
};
struct ffReplay {
  uint16_t length, delay;
};
struct ffRumble {
  uint16_t strongMagnitude, weakMagnitude;
};
struct ffEffect {
  uint16_t type;
  int16_t id;
  uint16_t direction;
  ffTrigger trigger;
  ffReplay replay;
  union {
    ffRumble rumble;
    // The real union holds a pointer member (custom_data), forcing 8-byte
    // alignment and a 48-byte struct.
    alignas(8) uint8_t pad[32];
  } u;
};
static_assert(sizeof(ffEffect) == 48, "ff_effect layout");
struct uinputFfUpload {
  uint32_t requestId;
  int32_t retval;
  ffEffect effect;
  ffEffect old;
};
constexpr uint32_t kUiBeginFfUpload =
    ioc(kIocWrite | kIocRead, 'U', 200, sizeof(uinputFfUpload));
constexpr uint32_t kUiEndFfUpload =
    iow('U', 201, sizeof(uinputFfUpload));

// ---------------------------------------------------------------------------
// Host input source: matches the guest read/write commands to evdev/uinput.
// ---------------------------------------------------------------------------

// The guest ioctl arg block (offsets verified from the kernel's dispatch):
//   +0x00 device handle, +0x08 output buffer, +0x10 requested report count,
//   +0x18 guest pointer that receives the produced count.
struct guestArgs {
  uint32_t handle;
  uint32_t pad0;
  void *buffer;
  uint32_t count;
  uint32_t pad1;
  void *outCount;
};

// Keyboard report (32 B, size verified). Layout follows the standard HID boot
// keyboard report plus a report id and padding.
struct KeyReport {
  uint8_t reportId;    // 0x01
  uint8_t modifier1;   // left ctrl/shift/alt/meta
  uint8_t modifier2;   // right ctrl/shift/alt/meta
  uint8_t reserved;
  uint8_t keycode[6];
  uint8_t pad[22];
};
static_assert(sizeof(KeyReport) == 32, "keyboard report size");

// Mouse report (16 B, size and layout verified against the SDK).
struct MouseReport {
  uint8_t buttons;
  uint8_t reserved0[3];
  int16_t relX;
  int16_t relY;
  int16_t wheel;
  int16_t tilt;
  uint8_t reserved1[2];
  uint8_t timestamp;
  uint8_t padding;
};
static_assert(sizeof(MouseReport) == 16, "mouse report size");

// Controller report (152 B, size verified; field layout best-known from the
// DS4 HID report the kernel forwards). Sticks/triggers are the 0..255 DS4
// range; buttons follow the DS4 button bytes.
struct CtrlReport {
  uint8_t reportId;  // 0x01
  uint8_t reserved;
  uint8_t buttons[8];
  uint8_t pad[6];
  int16_t leftStickX;
  int16_t leftStickY;
  int16_t rightStickX;
  int16_t rightStickY;
  int16_t leftTrigger;
  int16_t rightTrigger;
  uint8_t rest[0x98 - 0x1C];
};
static_assert(sizeof(CtrlReport) == 0x98, "controller report size");

// One open evdev node plus its role.
struct evdevDevice {
  int fd = -1;
  bool isKeyboard = false;
  bool isMouse = false;
  bool isPad = false;
  // Current state, updated from the event stream.
  uint8_t keyState[(kKeyMax + 8) / 8] = {};
  bool mouseButtons[5] = {};
  int16_t mouseRel[3] = {};      // x, y, wheel
  int32_t absState[kAbsMax + 1] = {};
  bool padButtons[16] = {};
  // Absolute-axis ranges for normalizing raw values to 0..255.
  struct {
    int32_t min, max;
  } absRange[kAbsMax + 1];
  bool haveAbs[kAbsMax + 1] = {};
  // Tracks whether a report was produced since the last read.
  bool dirty = false;
};

// Lazily-opened host devices (kept for the life of the process).
struct hidSource {
  evdevDevice keyboard, mouse, pad;
  bool scanned = false;
  int uinputFd = -1;
  int ffId = -1;

  void scan();
  void drain(evdevDevice &dev);
  int64_t nowUs();
};

// ---------------------------------------------------------------- evdev glue

static bool testBit(const uint8_t *bits, uint32_t code) {
  return (bits[code >> 3] >> (code & 7)) & 1;
}

void hidSource::scan() {
  if (scanned)
    return;
  scanned = true;

  // /dev/input/eventN in order; keep the first device of each role.
  for (int n = 0; n < 32 && (!keyboard.isKeyboard || !mouse.isMouse ||
                             !pad.isPad);
       n++) {
    char path[32];
    std::snprintf(path, sizeof(path), "/dev/input/event%d", n);
    int fd = ::open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0)
      continue;

    uint8_t keyBits[(kKeyMax + 8) / 8] = {};
    uint8_t relBits[(kRelMax + 8) / 8] = {};
    uint8_t absBits[(kAbsMax + 8) / 8] = {};
    const bool haveKeys =
        ::ioctl(fd, kEviocGKey, keyBits) >= 0 &&
        ::ioctl(fd, kEviocGRel, relBits) >= 0 &&
        ::ioctl(fd, kEviocGAbs, absBits) >= 0;
    if (!haveKeys) {
      ::close(fd);
      continue;
    }

    const bool hasGamepadKeys = testBit(keyBits, 0x130) /*BTN_SOUTH*/;
    const bool hasMouseBtns =
        testBit(keyBits, 0x110) /*BTN_LEFT*/ || testBit(keyBits, 0x111);
    const bool hasSticks = testBit(absBits, 0x00) /*ABS_X*/;

    evdevDevice *slot = nullptr;
    if (hasSticks && hasGamepadKeys)
      slot = &pad;
    else if (testBit(relBits, 0x00) /*REL_X*/ && hasMouseBtns)
      slot = &mouse;
    else if (testBit(keyBits, 0x1e) /*KEY_A*/ && !hasSticks)
      slot = &keyboard;

    if (slot && slot->fd < 0) {
      slot->fd = fd;
      slot->isKeyboard = (slot == &keyboard);
      slot->isMouse = (slot == &mouse);
      slot->isPad = (slot == &pad);
      if (slot->isPad) {
        // Cache axis ranges for the axes a pad reports.
        for (uint32_t ax = 0; ax <= kAbsMax; ax++) {
          if (!testBit(absBits, ax))
            continue;
          struct absInfo {
            int32_t value, minimum, maximum, fuzz, flat, resolution;
          } info = {};
          if (::ioctl(fd, ior('E', 0x40 + ax, sizeof(info)), &info) >= 0) {
            slot->absRange[ax] = {info.minimum, info.maximum};
            slot->haveAbs[ax] = true;
          }
        }
      }
      continue;
    }
    ::close(fd);
  }
}

// Non-blocking drain of one device's pending events into its state.
void hidSource::drain(evdevDevice &dev) {
  if (dev.fd < 0)
    return;
  inputEvent ev;
  while (true) {
    const ssize_t n = ::read(dev.fd, &ev, sizeof(ev));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      // Device gone: drop it and let the next scan reopen it.
      ::close(dev.fd);
      dev.fd = -1;
      break;
    }
    if (n != static_cast<ssize_t>(sizeof(ev)))
      continue;
    if (ev.type == kEvKey) {
      if (ev.code < (kKeyMax + 1) / 8 * 8 && ev.value >= 0 && ev.value <= 2) {
        const bool down = ev.value != 0;
        if (dev.isKeyboard) {
          const uint8_t bit = static_cast<uint8_t>(1u << (ev.code & 7));
          if ((dev.keyState[ev.code >> 3] & bit) != (down ? bit : 0))
            dev.dirty = true;
          if (down)
            dev.keyState[ev.code >> 3] |= bit;
          else
            dev.keyState[ev.code >> 3] &= static_cast<uint8_t>(~bit);
        } else if (dev.isMouse) {
          if (ev.code >= 0x110 && ev.code <= 0x114) {
            const int idx = ev.code - 0x110;
            if (dev.mouseButtons[idx] != down) {
              dev.mouseButtons[idx] = down;
              dev.dirty = true;
            }
          }
        } else if (dev.isPad) {
          // DS4-mapped pad buttons (BTN_SOUTH..BTN_THUMBR, d-pad, PS).
          static const uint16_t kPadCodes[] = {
              0x130, 0x131, 0x132, 0x133, 0x134, 0x135, 0x136, 0x137,
              0x138, 0x139, 0x13a, 0x13b, 0x13c, 0x13d, 0x13e, 0x220};
          for (int i = 0; i < 16; i++) {
            if (ev.code == kPadCodes[i]) {
              if (dev.padButtons[i] != down) {
                dev.padButtons[i] = down;
                dev.dirty = true;
              }
              break;
            }
          }
        }
      }
    } else if (ev.type == kEvRel && dev.isMouse) {
      int idx = -1;
      if (ev.code == 0x00)
        idx = 0;
      else if (ev.code == 0x01)
        idx = 1;
      else if (ev.code == 0x08)
        idx = 2;  // REL_WHEEL
      if (idx >= 0) {
        dev.mouseRel[idx] =
            static_cast<int16_t>(dev.mouseRel[idx] + ev.value);
        dev.dirty = true;
      }
    } else if (ev.type == kEvAbs && dev.isPad && ev.code <= kAbsMax) {
      dev.absState[ev.code] = ev.value;
      dev.dirty = true;
    }
  }
}

int64_t hidSource::nowUs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

// Normalize a raw absolute value into the DS4 0..255 range.
static int16_t toAxis(const evdevDevice &dev, uint32_t code) {
  if (!dev.haveAbs[code])
    return 128;
  const int32_t min = dev.absRange[code].min, max = dev.absRange[code].max;
  if (max <= min)
    return 128;
  const int32_t v = dev.absState[code];
  const int32_t scaled = (v - min) * 255 / (max - min);
  return static_cast<int16_t>(scaled < 0 ? 0 : scaled > 255 ? 255 : scaled);
}

// ---------------------------------------------------------- report synthesis

int fillKeyboard(hidSource &src, void *buffer, uint32_t count) {
  evdevDevice &dev = src.keyboard;
  src.drain(dev);
  if (dev.fd < 0)
    return 0;
  const uint32_t produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  KeyReport rep = {};
  rep.reportId = 0x01;
  // Modifier bits (HID order: L ctrl/shift/alt/meta, R ctrl/shift/alt/meta).
  const struct {
    uint32_t code;
    uint8_t bit;
    bool right;
  } kMods[] = {{0x1d, 0x01, false}, {0x2a, 0x02, false}, {0x38, 0x04, false},
               {0x7d, 0x08, false}, {0x61, 0x01, true},  {0x36, 0x02, true},
               {0x64, 0x04, true},  {0x7e, 0x08, true}};
  for (const auto &m : kMods) {
    if (testBit(dev.keyState, m.code)) {
      if (m.right)
        rep.modifier2 |= m.bit;
      else
        rep.modifier1 |= m.bit;
    }
  }
  // Up to six pressed non-modifier keys, in HID usage order.
  // evdev KEY_* -> HID keyboard usage for the common keys.
  uint8_t hid[(kKeyMax + 1) / 8 * 8] = {};
  const auto map = [&](uint32_t code, uint8_t usage) {
    if (code < sizeof(hid) * 8 && testBit(dev.keyState, code))
      hid[usage >> 3] |= static_cast<uint8_t>(1u << (usage & 7));
  };
  for (uint32_t i = 0; i < 10; i++)  // digits: usage 0x1e..0x27
    map(2 + i, static_cast<uint8_t>(0x1e + i));
  map(0x1d, 0xe0); map(0x2a, 0xe1); map(0x38, 0xe2); map(0x7d, 0xe3);
  map(0x61, 0xe4); map(0x36, 0xe5); map(0x64, 0xe6); map(0x7e, 0xe7);
  map(0x1e, 0x04); map(0x30, 0x05); map(0x2e, 0x06); map(0x20, 0x07);
  map(0x12, 0x08); map(0x21, 0x09); map(0x22, 0x0a); map(0x23, 0x0b);
  map(0x17, 0x0c); map(0x24, 0x0d); map(0x25, 0x0e); map(0x26, 0x0f);
  map(0x32, 0x10); map(0x31, 0x11); map(0x18, 0x12); map(0x19, 0x13);
  map(0x10, 0x14); map(0x13, 0x15); map(0x1f, 0x16); map(0x14, 0x17);
  map(0x16, 0x18); map(0x2f, 0x19); map(0x11, 0x1a); map(0x2d, 0x1b);
  map(0x15, 0x1c); map(0x2c, 0x1d);
  map(0x1c, 0x28); map(0x01, 0x29); map(0x0e, 0x2a); map(0x0f, 0x2b);
  map(0x39, 0x2c); map(0x0c, 0x2d); map(0x0d, 0x2e); map(0x1a, 0x2f);
  map(0x1b, 0x30); map(0x2b, 0x31); map(0x27, 0x33); map(0x28, 0x34);
  map(0x29, 0x35); map(0x33, 0x36); map(0x34, 0x37); map(0x35, 0x38);
  map(0x3a, 0x39);  // CapsLock
  for (uint32_t i = 0; i < 10; i++)  // F1..F10: usage 0x3a..0x43
    map(0x3b + i, static_cast<uint8_t>(0x3a + i));
  map(0x57, 0x44);  // F11
  map(0x58, 0x45);  // F12
  uint8_t n = 0;
  for (uint32_t usage = 4; usage < 0xe8 && n < 6; usage++) {
    if (hid[usage >> 3] >> (usage & 7) & 1)
      rep.keycode[n++] = static_cast<uint8_t>(usage);
  }

  auto *out = static_cast<uint8_t *>(buffer);
  std::memcpy(out, &rep, sizeof(rep));
  return produced;
}

int fillMouse(hidSource &src, void *buffer, uint32_t count) {
  evdevDevice &dev = src.mouse;
  src.drain(dev);
  if (dev.fd < 0)
    return 0;
  const uint32_t produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  MouseReport rep = {};
  rep.buttons = static_cast<uint8_t>(
      (dev.mouseButtons[0] ? 1u : 0u) | (dev.mouseButtons[1] ? 2u : 0u) |
      (dev.mouseButtons[2] ? 4u : 0u) | (dev.mouseButtons[3] ? 8u : 0u) |
      (dev.mouseButtons[4] ? 0x10u : 0u));
  rep.relX = dev.mouseRel[0];
  rep.relY = dev.mouseRel[1];
  rep.wheel = dev.mouseRel[2];
  rep.timestamp = static_cast<uint8_t>(src.nowUs() & 0xff);
  dev.mouseRel[0] = dev.mouseRel[1] = dev.mouseRel[2] = 0;

  std::memcpy(buffer, &rep, sizeof(rep));
  return produced;
}

int fillController(hidSource &src, void *buffer, uint32_t count) {
  evdevDevice &dev = src.pad;
  src.drain(dev);
  if (dev.fd < 0)
    return 0;
  const uint32_t produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  CtrlReport rep = {};
  rep.reportId = 0x01;
  for (int i = 0; i < 16; i++) {
    if (dev.padButtons[i])
      rep.buttons[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  }
  rep.leftStickX = toAxis(dev, 0x00);
  rep.leftStickY = toAxis(dev, 0x01);
  rep.rightStickX = toAxis(dev, 0x03);
  rep.rightStickY = toAxis(dev, 0x04);
  rep.leftTrigger = toAxis(dev, 0x02);
  rep.rightTrigger = toAxis(dev, 0x05);

  std::memcpy(buffer, &rep, sizeof(rep));
  return produced;
}

// ------------------------------------------------------------------- uinput

void openUinput(hidSource &src) {
  if (src.uinputFd >= 0)
    return;
  const int fd = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
  if (fd < 0)
    return;
  // A virtual gamepad: sticks + DS4-style buttons + rumble + a light.
  auto ok = [&](uint32_t ioctl, void *arg) { return ::ioctl(fd, ioctl, arg) >= 0; };
  int v = kEvAbs;
  if (!ok(kUiSetEvbit, &v))
    goto fail;
  v = kEvKey;
  if (!ok(kUiSetEvbit, &v))
    goto fail;
  v = kEvFf;
  if (!ok(kUiSetEvbit, &v))
    goto fail;
  v = kEvLed;
  if (!ok(kUiSetEvbit, &v))
    goto fail;
  for (uint32_t ax = 0; ax <= kAbsMax; ax++) {
    v = static_cast<int>(ax);
    if (!ok(kUiSetAbsbit, &v))
      goto fail;
  }
  for (const uint32_t code : {0x130u, 0x131u, 0x133u, 0x134u, 0x136u, 0x137u,
                              0x138u, 0x139u, 0x13au, 0x13bu, 0x13cu, 0x13du,
                              0x13eu, 0x220u, 0x221u, 0x222u, 0x223u}) {
    v = static_cast<int>(code);
    if (!ok(kUiSetKeybit, &v))
      goto fail;
  }
  v = static_cast<int>(kFfRumble);
  if (!ok(kUiSetFfbit, &v))
    goto fail;
  v = 0x08;  // LED_MISC
  if (!ok(kUiSetLedbit, &v))
    goto fail;

  {
    uinputUserDev dev = {};
    std::snprintf(dev.name, sizeof(dev.name), "Delta Virtual DS4");
    dev.id.bustype = 0x06;  // BUS_VIRTUAL
    dev.id.vendor = 0x054c;
    dev.id.product = 0x09cc;
    dev.ffEffectsMax = 1;
    for (uint32_t ax = 0; ax <= kAbsMax; ax++) {
      dev.absmin[ax] = 0;
      dev.absmax[ax] = 255;
    }
    if (::write(fd, &dev, sizeof(dev)) != static_cast<ssize_t>(sizeof(dev)))
      goto fail;
  }
  if (!ok(kUiDevCreate, nullptr))
    goto fail;
  src.uinputFd = fd;
  return;

fail:
  ::close(fd);
}

// Set rumble. `large`/`small` are the DS4 motor intensities (0..255).
void setRumble(hidSource &src, uint8_t large, uint8_t small) {
  if (src.uinputFd < 0)
    return;
  if (large == 0 && small == 0 && src.ffId >= 0) {
    inputEvent ev = {};
    ev.type = kEvFf;
    ev.code = static_cast<uint16_t>(src.ffId);
    ev.value = 0;  // stop
    ::write(src.uinputFd, &ev, sizeof(ev));
    return;
  }
  uinputFfUpload up = {};
  if (::ioctl(src.uinputFd, kUiBeginFfUpload, &up) < 0)
    return;
  up.effect.type = kFfRumble;
  up.effect.id = -1;
  up.effect.u.rumble.strongMagnitude =
      static_cast<uint16_t>(large * 0xffff / 255);
  up.effect.u.rumble.weakMagnitude =
      static_cast<uint16_t>(small * 0xffff / 255);
  if (::ioctl(src.uinputFd, kUiEndFfUpload, &up) < 0)
    return;
  src.ffId = up.effect.id;
  inputEvent ev = {};
  ev.type = kEvFf;
  ev.code = static_cast<uint16_t>(src.ffId);
  ev.value = 1;  // play
  ::write(src.uinputFd, &ev, sizeof(ev));
}

void setLight(hidSource &src, uint8_t r, uint8_t g, uint8_t b) {
  if (src.uinputFd < 0)
    return;
  inputEvent ev = {};
  ev.type = kEvLed;
  ev.code = 0x08;  // LED_MISC
  ev.value = (r || g || b) ? 1 : 0;
  ::write(src.uinputFd, &ev, sizeof(ev));
}

// ------------------------------------------------------------------ dispatch

struct guestArgsAt {
  // The kernel's read handlers copy reports into the guest buffer and return
  // how many they produced. data points at the guest arg block, which is
  // identity-mapped so its pointers are usable directly.
  static bool valid(void *data) { return data != nullptr; }
  static uint32_t handle(void *data) {
    return static_cast<guestArgs *>(data)->handle;
  }
  static void *buffer(void *data) {
    return static_cast<guestArgs *>(data)->buffer;
  }
  static uint32_t count(void *data) {
    return static_cast<guestArgs *>(data)->count;
  }
  static void *outCount(void *data) {
    return static_cast<guestArgs *>(data)->outCount;
  }
  static void writeCount(void *data, uint32_t produced) {
    void *out = outCount(data);
    if (out)
      std::memcpy(out, &produced, 4);
  }
};

int runRead(hidSource &src, void *data, int (*fill)(hidSource &, void *,
                                                    uint32_t)) {
  if (!guestArgsAt::valid(data))
    return 0;
  const uint32_t count = guestArgsAt::count(data);
  void *buffer = guestArgsAt::buffer(data);
  if (!buffer || count == 0)
    return 0;
  const uint32_t cap = (count > 16) ? 16 : count;
  const int produced = fill(src, buffer, cap);
  guestArgsAt::writeCount(data, produced > 0 ? static_cast<uint32_t>(produced) : 0);
  return produced > 0 ? produced : 0;
}

// SetVibration: { large, small, reserved[2] }.
int runVibration(hidSource &src, void *data) {
  if (!guestArgsAt::valid(data))
    return 0;
  const uint8_t *p = static_cast<const uint8_t *>(guestArgsAt::buffer(data));
  if (!p)
    return 0;
  openUinput(src);
  setRumble(src, p[0], p[1]);
  return 0;
}

// SetLightBar / ResetLightBar: { r, g, b, reserved } (reset passes zeroes).
int runLight(hidSource &src, void *data, bool reset) {
  if (!guestArgsAt::valid(data))
    return 0;
  const uint8_t *p = static_cast<const uint8_t *>(guestArgsAt::buffer(data));
  openUinput(src);
  if (p)
    setLight(src, reset ? 0 : p[0], reset ? 0 : p[1], reset ? 0 : p[2]);
  else
    setLight(src, 0, 0, 0);
  return 0;
}

}  // namespace

namespace krnl {
hidDevice::hidDevice(proc *p) : device(p) {}

int32_t hidDevice::ioctl(uint32_t cmd, void *data) {
  // Passthrough off: mirror the real device's system-only soft-fail.
  if (!kHidPassthrough) {
    if (data && (cmd & 0x40000000u)) {
      const uint32_t len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(data, 0, len);
    }
    return 0;
  }

  static hidSource src;
  src.scan();

  switch (cmd) {
    case 0x80204814:  // KeyboardRead
    case 0x80204815:  // KeyboardReadPort
      return runRead(src, data, fillKeyboard);
    case 0x80204819:  // MouseRead
    case 0x8020481a:  // MouseReadPort
      return runRead(src, data, fillMouse);
    case 0x80204820:  // ControllerRead
    case 0x80204829:  // ControllerReadPort
    case 0x8028482e:  // ControllerRead2
      return runRead(src, data, fillController);
    case 0x80104822:  // SetVibration
      return runVibration(src, data);
    case 0x80104821:  // SetLightBar
      return runLight(src, data, false);
    case 0x80044825:  // ResetLightBar
      return runLight(src, data, true);
    default:
      // The system-only and informational commands the kernel gates on
      // system credentials: soft-succeed with a zeroed out-buffer.
      if (data && (cmd & 0x40000000u)) {
        const uint32_t len = (cmd >> 16) & 0x1fff;
        if (len)
          std::memset(data, 0, len);
      }
      return 0;
  }
}

int64_t hidDevice::lseek(int64_t, int) { return 0; }

int hidDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
