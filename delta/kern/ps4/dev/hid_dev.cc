#include "base/arch.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include "host_memory/host_memory.h"
#include "options/options.h"

#include "kern/ps4/controller_ports.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/hid_dev.h"

namespace {
// Legacy host evdev/uinput access. Pad handle management is independent of it.
DELTA_OPTION(bool, kHidPassthrough, "DELTA_HID_PASSTHROUGH", false);

// ---------------------------------------------------------------------------
// evdev/uinput ABI. The host sysroot ships no <linux/input.h>, so the handful
// of constants and layouts the passthrough needs are declared here. They are
// the stable Linux input ABI.
// ---------------------------------------------------------------------------

constexpr u32 kIocRead = 2, kIocWrite = 1;
constexpr u32 Ioc(u32 dir, u32 type, u32 nr, u32 size) {
  return (dir << 30) | (type << 8) | (nr) | (size << 16);
}
constexpr u32 Ior(u32 type, u32 nr, u32 size) {
  return Ioc(kIocRead, type, nr, size);
}
constexpr u32 Iow(u32 type, u32 nr, u32 size) {
  return Ioc(kIocWrite, type, nr, size);
}

constexpr u32 kEvKey = 0x01, kEvRel = 0x02, kEvAbs = 0x03, kEvLed = 0x11,
              kEvFf = 0x15;
constexpr u32 kKeyMax = 0x2ff, kRelMax = 0x0b, kAbsMax = 0x3f;
// EVIOCGBIT(ev, len)
constexpr u32 Eviocgbit(u32 ev, u32 len) {
  return Ior('E', 0x20 + ev, len);
}
constexpr u32 kEviocGKey = Eviocgbit(kEvKey, (kKeyMax + 8) / 8);
constexpr u32 kEviocGRel = Eviocgbit(kEvRel, (kRelMax + 8) / 8);
constexpr u32 kEviocGAbs = Eviocgbit(kEvAbs, (kAbsMax + 8) / 8);

struct InputEvent {
  u64 sec, usec;
  u16 type, code;
  i32 value;
};
static_assert(sizeof(InputEvent) == 24, "input_event layout");

constexpr u32 kUiSetEvbit = Iow('U', 100, 4);
constexpr u32 kUiSetKeybit = Iow('U', 101, 4);
constexpr u32 kUiSetAbsbit = Iow('U', 103, 4);
constexpr u32 kUiSetLedbit = Iow('U', 105, 4);
constexpr u32 kUiSetFfbit = Iow('U', 107, 4);
constexpr u32 kUiDevCreate = Ioc(0, 'U', 1, 0);   // 0x5501
constexpr u32 kUiDevDestroy = Ioc(0, 'U', 2, 0);  // 0x5502

struct UinputUserDev {
  char name[80];
  struct {
    u16 bustype, vendor, product, version;
  } id;
  u32 ff_effects_max;
  i32 absmax[kAbsMax + 1];
  i32 absmin[kAbsMax + 1];
  i32 absfuzz[kAbsMax + 1];
  i32 absflat[kAbsMax + 1];
};

constexpr u32 kFfRumble = 0x50;
// UI_BEGIN_FF_UPLOAD/UI_END_FF_UPLOAD carry two ff_effect (48 B each) plus an
// id + retval pair.
struct FfEnvelope {
  u16 attack_length, attack_level, fade_length, fade_level;
};
struct FfTrigger {
  u16 button, interval;
};
struct FfReplay {
  u16 length, delay;
};
struct FfRumble {
  u16 strong_magnitude, weak_magnitude;
};
struct FfEffect {
  u16 type;
  i16 id;
  u16 direction;
  FfTrigger trigger;
  FfReplay replay;
  union {
    FfRumble rumble;
    // The real union holds a pointer member (custom_data), forcing 8-byte
    // alignment and a 48-byte struct.
    alignas(8) u8 pad[32];
  } u;
};
static_assert(sizeof(FfEffect) == 48, "ff_effect layout");
struct UinputFfUpload {
  u32 request_id;
  i32 retval;
  FfEffect effect;
  FfEffect old;
};
constexpr u32 kUiBeginFfUpload =
    Ioc(kIocWrite | kIocRead, 'U', 200, sizeof(UinputFfUpload));
constexpr u32 kUiEndFfUpload = Iow('U', 201, sizeof(UinputFfUpload));

// ---------------------------------------------------------------------------
// Host input source: matches the guest read/write commands to evdev/uinput.
// ---------------------------------------------------------------------------

// The guest ioctl arg block (offsets verified from the kernel's dispatch):
//   +0x00 device handle, +0x08 output buffer, +0x10 requested report count,
//   +0x18 guest pointer that receives the produced count.
struct GuestArgs {
  u32 handle;
  u32 pad0;
  void* buffer;
  u32 count;
  u32 pad1;
  void* out_count;
};

// Keyboard report (32 B, size verified). Layout follows the standard HID boot
// keyboard report plus a report id and padding.
struct KeyReport {
  u8 report_id;  // 0x01
  u8 modifier1;  // left ctrl/shift/alt/meta
  u8 modifier2;  // right ctrl/shift/alt/meta
  u8 reserved;
  u8 keycode[6];
  u8 pad[22];
};
static_assert(sizeof(KeyReport) == 32, "keyboard report size");

// Mouse report (16 B, size and layout verified against the SDK).
struct MouseReport {
  u8 buttons;
  u8 reserved0[3];
  i16 rel_x;
  i16 rel_y;
  i16 wheel;
  i16 tilt;
  u8 reserved1[2];
  u8 timestamp;
  u8 padding;
};
static_assert(sizeof(MouseReport) == 16, "mouse report size");

// Controller report (152 B, size verified; field layout best-known from the
// DS4 HID report the kernel forwards). Sticks/triggers are the 0..255 DS4
// range; buttons follow the DS4 button bytes.
struct CtrlReport {
  u8 report_id;  // 0x01
  u8 reserved;
  u8 buttons[8];
  u8 pad[6];
  i16 left_stick_x;
  i16 left_stick_y;
  i16 right_stick_x;
  i16 right_stick_y;
  i16 left_trigger;
  i16 right_trigger;
  u8 rest[0x98 - 0x1C];
};
static_assert(sizeof(CtrlReport) == 0x98, "controller report size");

// One open evdev node plus its role.
struct EvdevDevice {
  int fd = -1;
  bool is_keyboard = false;
  bool is_mouse = false;
  bool is_pad = false;
  // Current state, updated from the event stream.
  u8 key_state[(kKeyMax + 8) / 8] = {};
  bool mouse_buttons[5] = {};
  i16 mouse_rel[3] = {};  // x, y, wheel
  i32 abs_state[kAbsMax + 1] = {};
  bool pad_buttons[16] = {};
  // Absolute-axis ranges for normalizing raw values to 0..255.
  struct {
    i32 min, max;
  } abs_range[kAbsMax + 1];
  bool have_abs[kAbsMax + 1] = {};
  // Tracks whether a report was produced since the last read.
  bool dirty = false;
};

// Lazily-opened host devices (kept for the life of the process).
struct HidSource {
  EvdevDevice keyboard, mouse, pad;
  bool scanned = false;
  int uinput_fd = -1;
  int ff_id = -1;

  void Scan();
  void Drain(EvdevDevice& dev);
  i64 NowUs();
};

// ---------------------------------------------------------------- evdev glue

static bool TestBit(const u8* bits, u32 code) {
  return (bits[code >> 3] >> (code & 7)) & 1;
}

void HidSource::Scan() {
  if (scanned)
    return;
  scanned = true;

  // /dev/input/eventN in order; keep the first device of each role.
  for (int n = 0;
       n < 32 && (!keyboard.is_keyboard || !mouse.is_mouse || !pad.is_pad);
       n++) {
    char path[32];
    std::snprintf(path, sizeof(path), "/dev/input/event%d", n);
    int fd = ::open(path, O_RDONLY | O_NONBLOCK);
    if (fd < 0)
      continue;

    u8 key_bits[(kKeyMax + 8) / 8] = {};
    u8 rel_bits[(kRelMax + 8) / 8] = {};
    u8 abs_bits[(kAbsMax + 8) / 8] = {};
    const bool have_keys = ::ioctl(fd, kEviocGKey, key_bits) >= 0 &&
                           ::ioctl(fd, kEviocGRel, rel_bits) >= 0 &&
                           ::ioctl(fd, kEviocGAbs, abs_bits) >= 0;
    if (!have_keys) {
      ::close(fd);
      continue;
    }

    const bool has_gamepad_keys = TestBit(key_bits, 0x130) /*BTN_SOUTH*/;
    const bool has_mouse_btns =
        TestBit(key_bits, 0x110) /*BTN_LEFT*/ || TestBit(key_bits, 0x111);
    const bool has_sticks = TestBit(abs_bits, 0x00) /*ABS_X*/;

    EvdevDevice* slot = nullptr;
    if (has_sticks && has_gamepad_keys)
      slot = &pad;
    else if (TestBit(rel_bits, 0x00) /*REL_X*/ && has_mouse_btns)
      slot = &mouse;
    else if (TestBit(key_bits, 0x1e) /*KEY_A*/ && !has_sticks)
      slot = &keyboard;

    if (slot && slot->fd < 0) {
      slot->fd = fd;
      slot->is_keyboard = (slot == &keyboard);
      slot->is_mouse = (slot == &mouse);
      slot->is_pad = (slot == &pad);
      if (slot->is_pad) {
        // Cache axis ranges for the axes a pad reports.
        for (u32 ax = 0; ax <= kAbsMax; ax++) {
          if (!TestBit(abs_bits, ax))
            continue;
          struct AbsInfo {
            i32 value, minimum, maximum, fuzz, flat, resolution;
          } info = {};
          if (::ioctl(fd, Ior('E', 0x40 + ax, sizeof(info)), &info) >= 0) {
            slot->abs_range[ax] = {info.minimum, info.maximum};
            slot->have_abs[ax] = true;
          }
        }
      }
      continue;
    }
    ::close(fd);
  }
}

// Non-blocking drain of one device's pending events into its state.
void HidSource::Drain(EvdevDevice& dev) {
  if (dev.fd < 0)
    return;
  InputEvent ev;
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
        if (dev.is_keyboard) {
          const u8 bit = static_cast<u8>(1u << (ev.code & 7));
          if ((dev.key_state[ev.code >> 3] & bit) != (down ? bit : 0))
            dev.dirty = true;
          if (down)
            dev.key_state[ev.code >> 3] |= bit;
          else
            dev.key_state[ev.code >> 3] &= static_cast<u8>(~bit);
        } else if (dev.is_mouse) {
          if (ev.code >= 0x110 && ev.code <= 0x114) {
            const int idx = ev.code - 0x110;
            if (dev.mouse_buttons[idx] != down) {
              dev.mouse_buttons[idx] = down;
              dev.dirty = true;
            }
          }
        } else if (dev.is_pad) {
          // DS4-mapped pad buttons (BTN_SOUTH..BTN_THUMBR, d-pad, PS).
          static const u16 kPadCodes[] = {
              0x130, 0x131, 0x132, 0x133, 0x134, 0x135, 0x136, 0x137,
              0x138, 0x139, 0x13a, 0x13b, 0x13c, 0x13d, 0x13e, 0x220};
          for (int i = 0; i < 16; i++) {
            if (ev.code == kPadCodes[i]) {
              if (dev.pad_buttons[i] != down) {
                dev.pad_buttons[i] = down;
                dev.dirty = true;
              }
              break;
            }
          }
        }
      }
    } else if (ev.type == kEvRel && dev.is_mouse) {
      int idx = -1;
      if (ev.code == 0x00)
        idx = 0;
      else if (ev.code == 0x01)
        idx = 1;
      else if (ev.code == 0x08)
        idx = 2;  // REL_WHEEL
      if (idx >= 0) {
        dev.mouse_rel[idx] = static_cast<i16>(dev.mouse_rel[idx] + ev.value);
        dev.dirty = true;
      }
    } else if (ev.type == kEvAbs && dev.is_pad && ev.code <= kAbsMax) {
      dev.abs_state[ev.code] = ev.value;
      dev.dirty = true;
    }
  }
}

i64 HidSource::NowUs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<i64>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

// Normalize a raw absolute value into the DS4 0..255 range.
static i16 ToAxis(const EvdevDevice& dev, u32 code) {
  if (!dev.have_abs[code])
    return 128;
  const i32 min = dev.abs_range[code].min, max = dev.abs_range[code].max;
  if (max <= min)
    return 128;
  const i32 v = dev.abs_state[code];
  const i32 scaled = (v - min) * 255 / (max - min);
  return static_cast<i16>(scaled < 0 ? 0 : scaled > 255 ? 255 : scaled);
}

// ---------------------------------------------------------- report synthesis

int FillKeyboard(HidSource& src, void* buffer, u32 count) {
  EvdevDevice& dev = src.keyboard;
  src.Drain(dev);
  if (dev.fd < 0)
    return 0;
  const u32 produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  KeyReport rep = {};
  rep.report_id = 0x01;
  // Modifier bits (HID order: L ctrl/shift/alt/meta, R ctrl/shift/alt/meta).
  const struct {
    u32 code;
    u8 bit;
    bool right;
  } kMods[] = {{0x1d, 0x01, false}, {0x2a, 0x02, false}, {0x38, 0x04, false},
               {0x7d, 0x08, false}, {0x61, 0x01, true},  {0x36, 0x02, true},
               {0x64, 0x04, true},  {0x7e, 0x08, true}};
  for (const auto& m : kMods) {
    if (TestBit(dev.key_state, m.code)) {
      if (m.right)
        rep.modifier2 |= m.bit;
      else
        rep.modifier1 |= m.bit;
    }
  }
  // Up to six pressed non-modifier keys, in HID usage order.
  // evdev KEY_* -> HID keyboard usage for the common keys.
  u8 hid[(kKeyMax + 1) / 8 * 8] = {};
  const auto map = [&](u32 code, u8 usage) {
    if (code < sizeof(hid) * 8 && TestBit(dev.key_state, code))
      hid[usage >> 3] |= static_cast<u8>(1u << (usage & 7));
  };
  for (u32 i = 0; i < 10; i++)  // digits: usage 0x1e..0x27
    map(2 + i, static_cast<u8>(0x1e + i));
  map(0x1d, 0xe0);
  map(0x2a, 0xe1);
  map(0x38, 0xe2);
  map(0x7d, 0xe3);
  map(0x61, 0xe4);
  map(0x36, 0xe5);
  map(0x64, 0xe6);
  map(0x7e, 0xe7);
  map(0x1e, 0x04);
  map(0x30, 0x05);
  map(0x2e, 0x06);
  map(0x20, 0x07);
  map(0x12, 0x08);
  map(0x21, 0x09);
  map(0x22, 0x0a);
  map(0x23, 0x0b);
  map(0x17, 0x0c);
  map(0x24, 0x0d);
  map(0x25, 0x0e);
  map(0x26, 0x0f);
  map(0x32, 0x10);
  map(0x31, 0x11);
  map(0x18, 0x12);
  map(0x19, 0x13);
  map(0x10, 0x14);
  map(0x13, 0x15);
  map(0x1f, 0x16);
  map(0x14, 0x17);
  map(0x16, 0x18);
  map(0x2f, 0x19);
  map(0x11, 0x1a);
  map(0x2d, 0x1b);
  map(0x15, 0x1c);
  map(0x2c, 0x1d);
  map(0x1c, 0x28);
  map(0x01, 0x29);
  map(0x0e, 0x2a);
  map(0x0f, 0x2b);
  map(0x39, 0x2c);
  map(0x0c, 0x2d);
  map(0x0d, 0x2e);
  map(0x1a, 0x2f);
  map(0x1b, 0x30);
  map(0x2b, 0x31);
  map(0x27, 0x33);
  map(0x28, 0x34);
  map(0x29, 0x35);
  map(0x33, 0x36);
  map(0x34, 0x37);
  map(0x35, 0x38);
  map(0x3a, 0x39);              // CapsLock
  for (u32 i = 0; i < 10; i++)  // F1..F10: usage 0x3a..0x43
    map(0x3b + i, static_cast<u8>(0x3a + i));
  map(0x57, 0x44);  // F11
  map(0x58, 0x45);  // F12
  u8 n = 0;
  for (u32 usage = 4; usage < 0xe8 && n < 6; usage++) {
    if (hid[usage >> 3] >> (usage & 7) & 1)
      rep.keycode[n++] = static_cast<u8>(usage);
  }

  auto* out = static_cast<u8*>(buffer);
  std::memcpy(out, &rep, sizeof(rep));
  return produced;
}

int FillMouse(HidSource& src, void* buffer, u32 count) {
  EvdevDevice& dev = src.mouse;
  src.Drain(dev);
  if (dev.fd < 0)
    return 0;
  const u32 produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  MouseReport rep = {};
  rep.buttons = static_cast<u8>(
      (dev.mouse_buttons[0] ? 1u : 0u) | (dev.mouse_buttons[1] ? 2u : 0u) |
      (dev.mouse_buttons[2] ? 4u : 0u) | (dev.mouse_buttons[3] ? 8u : 0u) |
      (dev.mouse_buttons[4] ? 0x10u : 0u));
  rep.rel_x = dev.mouse_rel[0];
  rep.rel_y = dev.mouse_rel[1];
  rep.wheel = dev.mouse_rel[2];
  rep.timestamp = static_cast<u8>(src.NowUs() & 0xff);
  dev.mouse_rel[0] = dev.mouse_rel[1] = dev.mouse_rel[2] = 0;

  std::memcpy(buffer, &rep, sizeof(rep));
  return produced;
}

int FillController(HidSource& src, void* buffer, u32 count) {
  EvdevDevice& dev = src.pad;
  src.Drain(dev);
  if (dev.fd < 0)
    return 0;
  const u32 produced = dev.dirty ? 1u : 0u;
  if (!produced)
    return 0;
  dev.dirty = false;

  CtrlReport rep = {};
  rep.report_id = 0x01;
  for (int i = 0; i < 16; i++) {
    if (dev.pad_buttons[i])
      rep.buttons[i / 8] |= static_cast<u8>(1u << (i % 8));
  }
  rep.left_stick_x = ToAxis(dev, 0x00);
  rep.left_stick_y = ToAxis(dev, 0x01);
  rep.right_stick_x = ToAxis(dev, 0x03);
  rep.right_stick_y = ToAxis(dev, 0x04);
  rep.left_trigger = ToAxis(dev, 0x02);
  rep.right_trigger = ToAxis(dev, 0x05);

  std::memcpy(buffer, &rep, sizeof(rep));
  return produced;
}

// ------------------------------------------------------------------- uinput

void OpenUinput(HidSource& src) {
  if (src.uinput_fd >= 0)
    return;
  const int fd = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
  if (fd < 0)
    return;
  // A virtual gamepad: sticks + DS4-style buttons + rumble + a light.
  auto ok = [&](u32 ioctl, void* arg) { return ::ioctl(fd, ioctl, arg) >= 0; };
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
  for (u32 ax = 0; ax <= kAbsMax; ax++) {
    v = static_cast<int>(ax);
    if (!ok(kUiSetAbsbit, &v))
      goto fail;
  }
  for (const u32 code :
       {0x130u, 0x131u, 0x133u, 0x134u, 0x136u, 0x137u, 0x138u, 0x139u, 0x13au,
        0x13bu, 0x13cu, 0x13du, 0x13eu, 0x220u, 0x221u, 0x222u, 0x223u}) {
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
    UinputUserDev dev = {};
    std::snprintf(dev.name, sizeof(dev.name), "Delta Virtual DS4");
    dev.id.bustype = 0x06;  // BUS_VIRTUAL
    dev.id.vendor = 0x054c;
    dev.id.product = 0x09cc;
    dev.ff_effects_max = 1;
    for (u32 ax = 0; ax <= kAbsMax; ax++) {
      dev.absmin[ax] = 0;
      dev.absmax[ax] = 255;
    }
    if (::write(fd, &dev, sizeof(dev)) != static_cast<ssize_t>(sizeof(dev)))
      goto fail;
  }
  if (!ok(kUiDevCreate, nullptr))
    goto fail;
  src.uinput_fd = fd;
  return;

fail:
  ::close(fd);
}

// Set rumble. `large`/`small` are the DS4 motor intensities (0..255).
void SetRumble(HidSource& src, u8 large, u8 small) {
  if (src.uinput_fd < 0)
    return;
  if (large == 0 && small == 0 && src.ff_id >= 0) {
    InputEvent ev = {};
    ev.type = kEvFf;
    ev.code = static_cast<u16>(src.ff_id);
    ev.value = 0;  // stop
    ::write(src.uinput_fd, &ev, sizeof(ev));
    return;
  }
  UinputFfUpload up = {};
  if (::ioctl(src.uinput_fd, kUiBeginFfUpload, &up) < 0)
    return;
  up.effect.type = kFfRumble;
  up.effect.id = -1;
  up.effect.u.rumble.strong_magnitude = static_cast<u16>(large * 0xffff / 255);
  up.effect.u.rumble.weak_magnitude = static_cast<u16>(small * 0xffff / 255);
  if (::ioctl(src.uinput_fd, kUiEndFfUpload, &up) < 0)
    return;
  src.ff_id = up.effect.id;
  InputEvent ev = {};
  ev.type = kEvFf;
  ev.code = static_cast<u16>(src.ff_id);
  ev.value = 1;  // play
  ::write(src.uinput_fd, &ev, sizeof(ev));
}

// ------------------------------------------------------------------ dispatch

struct GuestArgsAt {
  // The kernel's read handlers copy reports into the guest buffer and return
  // how many they produced. data points at the guest arg block, which is
  // identity-mapped so its pointers are usable directly.
  static bool Valid(void* data) { return data != nullptr; }
  static u32 Handle(void* data) {
    return static_cast<GuestArgs*>(data)->handle;
  }
  static void* Buffer(void* data) {
    return static_cast<GuestArgs*>(data)->buffer;
  }
  static u32 Count(void* data) { return static_cast<GuestArgs*>(data)->count; }
  static void* OutCount(void* data) {
    return static_cast<GuestArgs*>(data)->out_count;
  }
  static void WriteCount(void* data, u32 produced) {
    void* out = OutCount(data);
    if (out)
      std::memcpy(out, &produced, 4);
  }
};

int RunRead(HidSource& src, void* data, int (*fill)(HidSource&, void*, u32)) {
  if (!GuestArgsAt::Valid(data))
    return 0;
  const u32 count = GuestArgsAt::Count(data);
  void* buffer = GuestArgsAt::Buffer(data);
  if (!buffer || count == 0)
    return 0;
  const u32 cap = (count > 16) ? 16 : count;
  const int produced = fill(src, buffer, cap);
  GuestArgsAt::WriteCount(data, produced > 0 ? static_cast<u32>(produced) : 0);
  return produced > 0 ? produced : 0;
}

// SetVibration: { large, small, reserved[2] }.
int RunVibration(HidSource& src, void* data) {
  if (!GuestArgsAt::Valid(data))
    return 0;
  const u8* p = static_cast<const u8*>(GuestArgsAt::Buffer(data));
  if (!p)
    return 0;
  OpenUinput(src);
  SetRumble(src, p[0], p[1]);
  return 0;
}

}  // namespace

namespace kern {
HidDevice::HidDevice(ObjectTable& objects) : Device(objects) {}

HidDevice::~HidDevice() {
  for (u32 handle : controller_handles_)
    ReleaseControllerHandle(handle);
}

i32 HidDevice::ReadControllerReports(u32 handle,
                                     void* reports,
                                     u32 count,
                                     u32* connection,
                                     u32* extended,
                                     u32 stride) {
  if (!controller_handles_.count(handle) || !IsControllerHandle(handle))
    return -static_cast<i32>(SysError::eBADF);
  if (count == 0 || count > 64)
    return -static_cast<i32>(SysError::eINVAL);
  if (!host_memory::IsMemoryRangeMapped(reports, count * stride) ||
      !host_memory::IsMemoryRangeMapped(connection, sizeof(u32)) ||
      !host_memory::IsMemoryRangeMapped(extended, sizeof(u32)))
    return -static_cast<i32>(SysError::eFAULT);
  *extended = 0;
  guest::ControllerState state;
  if (!ReadControllerPort(handle, state)) {
    *connection = 0;
    return 0;
  }
  *connection = 1;
  u8 report[0xb8]{};
  std::memcpy(report, &state.timestamp, sizeof(state.timestamp));
  std::memcpy(report + 0xc, &state.buttons, sizeof(state.buttons));
  report[0x10] = state.lx;
  report[0x11] = state.ly;
  report[0x12] = state.rx;
  report[0x13] = state.ry;
  report[0x14] = state.l2;
  report[0x15] = state.r2;
  std::memcpy(reports, report, stride);
  return 1;
}

i32 HidDevice::Ioctl(u32 cmd, void* data) {
  if (cmd == 0x80104850) {
    struct ControlArgs {
      u8 operation, reserved[7];
      u32 handle, reserved1;
    };
    static_assert(sizeof(ControlArgs) == 16);
    if (!host_memory::IsMemoryRangeMapped(data, sizeof(ControlArgs)))
      return -static_cast<i32>(SysError::eFAULT);
    const auto& args = *static_cast<const ControlArgs*>(data);
    if (args.operation != 4)
      return -static_cast<i32>(SysError::eNOTSUP);
    std::lock_guard lock(handles_mutex_);
    if (!controller_handles_.count(args.handle))
      return -static_cast<i32>(SysError::eBADF);
    return ResetControllerLight(args.handle);
  }
  if (cmd == 0x80104821 || cmd == 0x80044825) {
    struct LightArgs {
      u32 handle, reserved;
      const u8* color;
    };
    if (!host_memory::IsMemoryRangeMapped(data, cmd == 0x80044825 ? 4 : 16))
      return -static_cast<i32>(SysError::eFAULT);
    const u32 handle = *static_cast<const u32*>(data);
    std::lock_guard lock(handles_mutex_);
    if (!controller_handles_.count(handle))
      return -static_cast<i32>(SysError::eBADF);
    if (cmd == 0x80044825)
      return ResetControllerLight(handle);
    const auto& args = *static_cast<const LightArgs*>(data);
    if (!host_memory::IsMemoryRangeMapped(args.color, 4))
      return -static_cast<i32>(SysError::eFAULT);
    if (args.color[3])
      return -static_cast<i32>(SysError::eNOTSUP);
    return LightControllerPort(handle, args.color[0], args.color[1],
                               args.color[2]);
  }
  if (cmd == 0xc0484851) {
    struct ReadArgs {
      u8 operation, reserved[3];
      u32 handle, count, reserved1, flags, reserved2;
      u64 reserved3;
      u32* connection;
      u32* extended;
      u64 reserved4[2];
      void* reports;
    };
    static_assert(sizeof(ReadArgs) == 0x48);
    if (!host_memory::IsMemoryRangeMapped(data, sizeof(ReadArgs)))
      return -static_cast<i32>(SysError::eFAULT);
    const auto& args = *static_cast<const ReadArgs*>(data);
    if (args.operation != 6)
      return -static_cast<i32>(SysError::eNOTSUP);
    std::lock_guard lock(handles_mutex_);
    return ReadControllerReports(args.handle, args.reports, args.count,
                                 args.connection, args.extended, 0xb8);
  }
  if (cmd == 0x8008485b) {
    if (!host_memory::IsMemoryRangeMapped(data, 8))
      return -static_cast<i32>(SysError::eFAULT);
    const auto* args = static_cast<const u32*>(data);
    std::lock_guard lock(handles_mutex_);
    if (!controller_handles_.count(args[0]) || !IsControllerHandle(args[0]))
      return -static_cast<i32>(SysError::eBADF);
    // Both compatibility modes read the virtual controller's normalized data.
    return args[1] <= 1 ? 0 : -static_cast<i32>(SysError::eINVAL);
  }
  if (cmd == 0xc0184852) {
    struct InfoArgs {
      u8 operation, reserved[3];
      u32 id;
      u64 reserved1;
      void* output;
    };
    static_assert(sizeof(InfoArgs) == 24);
    if (!host_memory::IsMemoryRangeMapped(data, sizeof(InfoArgs)))
      return -static_cast<i32>(SysError::eFAULT);
    const auto& args = *static_cast<const InfoArgs*>(data);
    if (args.operation > 1)
      return -static_cast<i32>(SysError::eNOTSUP);
    if (!host_memory::IsMemoryRangeMapped(args.output, 0xa8))
      return -static_cast<i32>(SysError::eFAULT);
    if (args.operation == 0 ? args.id != 1 || !HasControllerSource()
                            : !ControllerPortConnected(args.id))
      return -static_cast<i32>(SysError::eNODEV);
    u8 info[0xa8]{};
    auto word = [&](u32 offset, u16 value) {
      std::memcpy(info + offset, &value, sizeof(value));
    };
    word(0x10, 0x054c);
    word(0x12, 0x05c4);
    if (args.operation == 1) {
      const u32 physical = 1;
      std::memcpy(info + 4, &physical, sizeof(physical));
      word(0x9c, 4486);
      word(0xa0, 1920);
      word(0xa2, 942);
      info[0xa4] = info[0xa5] = 12;
    }
    std::memcpy(args.output, info, sizeof(info));
    return 0;
  }
  if (cmd == 0x80184822) {
    struct VibrationArgs {
      u32 handle, reserved;
      const u8* motors;
      u8 flags, reserved1[7];
    };
    static_assert(sizeof(VibrationArgs) == 24);
    if (!host_memory::IsMemoryRangeMapped(data, sizeof(VibrationArgs)))
      return -static_cast<i32>(SysError::eFAULT);
    const auto& args = *static_cast<const VibrationArgs*>(data);
    std::lock_guard lock(handles_mutex_);
    if (!controller_handles_.count(args.handle))
      return -static_cast<i32>(SysError::eBADF);
    if (!host_memory::IsMemoryRangeMapped(args.motors, 4))
      return -static_cast<i32>(SysError::eFAULT);
    if (args.flags)
      return -static_cast<i32>(SysError::eNOTSUP);
    return VibrateControllerPort(args.handle, args.motors[0], args.motors[1]);
  }
  if (cmd == 0x80104801 || cmd == 0x8010480e) {
    struct InfoArgs {
      u32 id, reserved;
      void* output;
    };
    if (!host_memory::IsMemoryRangeMapped(data, sizeof(InfoArgs)))
      return -SysError::eFAULT;
    const auto& args = *static_cast<const InfoArgs*>(data);
    if (!host_memory::IsMemoryRangeMapped(args.output, 0x40))
      return -SysError::eFAULT;
    if (cmd == 0x80104801 ? args.id != 1 || !HasControllerSource()
                          : !ControllerPortConnected(args.id))
      return -SysError::eNODEV;
    u8 info[0x40]{};
    auto word = [&](u32 offset, u16 value) {
      std::memcpy(info + offset, &value, sizeof(value));
    };
    if (cmd == 0x80104801) {
      word(0x10, 0x054c);
      word(0x12, 0x05c4);
    } else {
      const u32 physical = 1;
      std::memcpy(info + 4, &physical, sizeof(physical));
      word(0x34, 4486);
      word(0x38, 1920);
      word(0x3a, 942);
      info[0x3c] = info[0x3d] = 12;
    }
    std::memcpy(args.output, info, sizeof(info));
    return 0;
  }
  // PS4 controller open, close, and read request layouts.
  if (cmd == 0x800c4802 || cmd == 0x80084803 || cmd == 0x8030482e) {
    if (!host_memory::IsMemoryRangeMapped(data, (cmd >> 16) & 0x1fff))
      return -SysError::eFAULT;
    std::lock_guard lock(handles_mutex_);
    if (cmd == 0x800c4802) {
      const auto* args = static_cast<const u32*>(data);
      if (args[0] != ~u32(0) || args[1] != 3 || args[2] != 0)
        return -SysError::eINVAL;
      const i32 handle = AllocateControllerHandle();
      if (handle < 0)
        return handle;
      controller_handles_.insert(handle);
      return handle;
    }
    if (cmd == 0x80084803) {
      const auto* args = static_cast<const u32*>(data);
      if (args[0] != 3)
        return -SysError::eINVAL;
      if (!controller_handles_.erase(args[1]))
        return -SysError::eBADF;
      return ReleaseControllerHandle(args[1]) ? 0 : -SysError::eBADF;
    }
    struct ControllerReadArgs {
      u32 handle, reserved;
      void* buffer;
      u32 count, reserved1;
      u32* connection;
      u32* extended;
      u32 flags, reserved2;
    };
    static_assert(sizeof(ControllerReadArgs) == 0x30);
    const auto& args = *static_cast<const ControllerReadArgs*>(data);
    return ReadControllerReports(args.handle, args.buffer, args.count,
                                 args.connection, args.extended, 0xa8);
  }

  if (!kHidPassthrough) {
    if (data && (cmd & 0x40000000u)) {
      const u32 len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(data, 0, len);
    }
    return 0;
  }

  static HidSource src;
  src.Scan();

  switch (cmd) {
    case 0x80204814:  // KeyboardRead
    case 0x80204815:  // KeyboardReadPort
      return RunRead(src, data, FillKeyboard);
    case 0x80204819:  // MouseRead
    case 0x8020481a:  // MouseReadPort
      return RunRead(src, data, FillMouse);
    case 0x80204820:  // ControllerRead
    case 0x80204829:  // ControllerReadPort
    case 0x8028482e:  // ControllerRead2
      return RunRead(src, data, FillController);
    case 0x80104822:  // SetVibration
      return RunVibration(src, data);
    default:
      // The system-only and informational commands the kernel gates on
      // system credentials: soft-succeed with a zeroed out-buffer.
      if (data && (cmd & 0x40000000u)) {
        const u32 len = (cmd >> 16) & 0x1fff;
        if (len)
          std::memset(data, 0, len);
      }
      return 0;
  }
}

i64 HidDevice::Lseek(i64, int) {
  return 0;
}

int HidDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace kern
