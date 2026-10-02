
// Copyright (C) Force67

// This file was generated on 10/12/2019

#include "base/arch.h"
#include "runtime/vprx/vprx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/logging.h"

#include <cctype>
#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "host/window.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kMemWatch, "DELTA_MEMWATCH", nullptr);
DELTA_OPTION(const char*, kMemPoke, "DELTA_MEMPOKE", nullptr);
DELTA_OPTION(const char*, kPadScript, "DELTA_PAD_SCRIPT", nullptr);
DELTA_OPTION(u64, kAutoskipStop, "DELTA_PAD_AUTOSKIP_STOP", 0);
DELTA_OPTION(u64, kAutoskipStart, "DELTA_PAD_AUTOSKIP_START", 0);
DELTA_OPTION(bool, kPadKeyboard, "DELTA_PAD_KEYBOARD", true);
DELTA_OPTION(u64, kExploreReads, "DELTA_PAD_EXPLORE_READS", 130);
DELTA_OPTION(int, kExploreDir, "DELTA_PAD_EXPLORE_DIR", 0);
DELTA_OPTION(bool, kPadAutoskip, "DELTA_PAD_AUTOSKIP", false);
DELTA_OPTION(bool, kPadAutoskipDoom, "DELTA_PAD_AUTOSKIP_DOOM", false);
DELTA_OPTION(bool, kPadAutoskipNav, "DELTA_PAD_AUTOSKIP_NAV", false);
DELTA_OPTION(bool, kPadAutoskipNoopt, "DELTA_PAD_AUTOSKIP_NOOPT", false);
DELTA_OPTION(bool, kPadAutoskipSweep, "DELTA_PAD_AUTOSKIP_SWEEP", false);
DELTA_OPTION(bool, kPadExplore, "DELTA_PAD_EXPLORE", false);
DELTA_OPTION(bool, kPadExploreCont, "DELTA_PAD_EXPLORE_CONT", false);
DELTA_OPTION(bool, kPadTrace, "DELTA_PAD_TRACE", false);
}  // namespace

// HLE controller: one connected DS4, neutral state (sticks centered, no
// buttons). DELTA_PAD_AUTOSKIP=1 pulses confirm/back/start so a headless run
// advances past the intro/title into the menu for verification.
namespace {

// Orbis button bitmasks (ScePadButtonDataOffset).
enum : u32 {
  kL3 = 0x0002,
  kR3 = 0x0004,
  kOptions = 0x0008,
  kUp = 0x0010,
  kRight = 0x0020,
  kDown = 0x0040,
  kLeft = 0x0080,
  kL2 = 0x0100,
  kR2 = 0x0200,
  kL1 = 0x0400,
  kR1 = 0x0800,
  kTriangle = 0x1000,
  kCircle = 0x2000,
  kCross = 0x4000,
  kSquare = 0x8000,
  kTouchPad = 0x100000,
};

struct AnalogStick {
  u8 x, y;
};
struct AnalogButtons {
  u8 l2, r2;
};
struct FQuaternion {
  float x, y, z, w;
};
struct FVector3 {
  float x, y, z;
};
struct PadTouch {
  u16 x, y;
  u8 id;
  u8 reserve[3];
};
struct PadTouchData {
  u8 touch_num;
  u8 reserve[3];
  u32 reserve1;
  PadTouch touch[2];
};
struct PadExtUnitData {
  u32 id;
  u8 reserve;
  u8 data_len;
  u8 data[10];
};

// ScePadData: offsets verified against the orbis layout (connected@0x4C,
// timestamp@0x50). Written into the game's buffer on read.
struct PadData {
  u32 buttons;                   // 0x00
  AnalogStick left_stick;        // 0x04
  AnalogStick right_stick;       // 0x06
  AnalogButtons analog_buttons;  // 0x08
  u8 pad0[2];                    // 0x0A
  FQuaternion orientation;       // 0x0C
  FVector3 acceleration;         // 0x1C
  FVector3 angular_velocity;     // 0x28
  PadTouchData touch_data;       // 0x34
  bool connected;                // 0x4C
  u8 pad1[3];
  u64 timestamp;            // 0x50
  PadExtUnitData ext_unit;  // 0x58
  u8 connected_count;       // 0x68
  u8 reserve[2];
  u8 device_unique_data_len;  // 0x6B
  u8 device_unique_data[12];  // 0x6C
};
static_assert(sizeof(PadData) >= 0x78, "PadData layout");

struct PadControllerInformation {
  float touchpad_density;    // 0x00
  u16 touch_resolution_x;    // 0x04
  u16 touch_resolution_y;    // 0x06
  u8 stick_dead_zone_left;   // 0x08
  u8 stick_dead_zone_right;  // 0x09
  u8 connection_type;        // 0x0A
  u8 connected_count;        // 0x0B
  bool connected;            // 0x0C
  u32 device_class;          // 0x10 (ORBIS_PAD_DEVICE_CLASS_STANDARD = 0)
  u8 reserve[8];
};
static_assert(sizeof(PadControllerInformation) == 0x1c);

u64 g_read_seq = 0;

// DELTA_MEMWATCH=addr[,...]: background thread printing the qword at each guest
// VA every ~250ms with a wall-clock delta, to correlate a global's lifecycle
// with boot/crash timing. Started once from the first pad read (any polling
// title runs it).
void StartMemWatch() {
  const char* e = kMemWatch;
  if (!e)
    return;
  base::Vector<u64> addrs;
  for (const char* p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    char* end = nullptr;
    u64 v = std::strtoull(p, &end, 0);
    if (end == p)
      break;
    if (v >= 0x1000)
      addrs.push_back(v);
    p = end;
  }
  if (addrs.empty())
    return;
  base::SpawnDetachedThread("libScePad", [addrs] {
    const auto t0 = base::TimeTicks::Now();
    base::Vector<u64> last(addrs.size(), 0xdeadbeefdeadbeefull);
    for (;;) {
      double t = (base::TimeTicks::Now() - t0).InSecondsF();
      bool any = false;
      for (size_t i = 0; i < addrs.size(); i++) {
        u64 cur = *reinterpret_cast<volatile u64*>(addrs[i]);
        if (cur != last[i]) {
          BASE_LOGI("memwatch", "t={:.2f}  *{:#x}: {:016x} -> {:016x}", t,
                    (unsigned long long)addrs[i], (unsigned long long)last[i],
                    (unsigned long long)cur);
          last[i] = cur;
          any = true;
        }
      }
      (void)any;
      base::SleepForMilliseconds(250);
    }
  });
}

// DELTA_MEMPOKE=<spec>[,...], spec = addr:width:value[:delayMs]. addr is a
// literal hex VA or *PTR+OFF (resolves a singleton every apply, so a
// not-yet-constructed pointer is followed once live). Writes are RE-APPLIED
// every 200ms so the value is HELD against the guest's own updates, pinning a
// guest global to probe what a change triggers. Guest memory is
// identity-mapped.
struct PokeSpec {
  bool indirect = false;
  u64 ptr_addr = 0;  // for indirect: address holding the object pointer
  u64 off = 0;       // offset added to literal addr or to *ptrAddr
  int width = 8;
  u64 value = 0;
  u64 delay_ms = 0;
};

void StartMemPoke() {
  const char* e = kMemPoke;
  if (!e)
    return;
  base::Vector<PokeSpec> specs;
  const base::String in(e);
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    base::String s =
        in.substr(i, comma == base::String::npos ? comma : comma - i);
    i = comma == base::String::npos ? in.size() : comma + 1;
    // split s on ':' but the addr field may itself contain no ':'
    base::Vector<base::String> f;
    size_t j = 0;
    while (j <= s.size()) {
      size_t c = s.find(':', j);
      f.push_back(s.substr(j, c == base::String::npos ? c : c - j));
      if (c == base::String::npos)
        break;
      j = c + 1;
    }
    if (f.size() < 3)
      continue;
    PokeSpec p;
    base::String addr = f[0];
    if (!addr.empty() && addr[0] == '*') {
      p.indirect = true;
      size_t plus = addr.find('+');
      p.ptr_addr = std::strtoull(addr.c_str() + 1, nullptr, 0);
      p.off = plus == base::String::npos
                  ? 0
                  : std::strtoull(addr.c_str() + plus + 1, nullptr, 0);
    } else {
      p.off = std::strtoull(addr.c_str(), nullptr, 0);
    }
    p.width = std::atoi(f[1].c_str());
    p.value = std::strtoull(f[2].c_str(), nullptr, 0);
    if (f.size() >= 4)
      p.delay_ms = std::strtoull(f[3].c_str(), nullptr, 0);
    if (p.width == 1 || p.width == 2 || p.width == 4 || p.width == 8)
      specs.push_back(p);
  }
  if (specs.empty())
    return;
  base::SpawnDetachedThread("libScePad", [specs] {
    const auto t0 = base::TimeTicks::Now();
    base::Vector<bool> announced(specs.size(), false);
    for (;;) {
      double ms = (base::TimeTicks::Now() - t0).InSecondsF() * 1000.0;
      for (size_t k = 0; k < specs.size(); k++) {
        const auto& p = specs[k];
        if (ms < (double)p.delay_ms)
          continue;
        u64 target;
        if (p.indirect) {
          u64 obj = *reinterpret_cast<volatile u64*>(p.ptr_addr);
          if (obj < 0x1000)
            continue;  // singleton not constructed yet
          target = obj + p.off;
        } else {
          target = p.off;
        }
        if (target < 0x1000)
          continue;
        switch (p.width) {
          case 1:
            *reinterpret_cast<volatile u8*>(target) = (u8)p.value;
            break;
          case 2:
            *reinterpret_cast<volatile u16*>(target) = (u16)p.value;
            break;
          case 4:
            *reinterpret_cast<volatile u32*>(target) = (u32)p.value;
            break;
          case 8:
            *reinterpret_cast<volatile u64*>(target) = p.value;
            break;
        }
        if (!announced[k]) {
          announced[k] = true;
          BASE_LOGI("mempoke", "t={:.0f}ms first write {:#x} <- {:#x} (w{})",
                    ms, (unsigned long long)target, (unsigned long long)p.value,
                    p.width);
        }
      }
      base::SleepForMilliseconds(200);
    }
  });
}

// Auto-skip pulse for headless verification runs. NEVER pulse Circle with
// Cross: confirm then immediately backs out, so menus never advance. Sequence:
// Options first (title -> main menu), then Cross to confirm New
// Run/save-slot/char-select with the occasional Down; gaps between pulses give
// clean press edges. DELTA_PAD_SCRIPT="<name>:<reads>[,...]": replay one exact
// button sequence (names = the constants below plus "none" for a gap), then
// hold neutral.
u32 ScriptButtons(bool& active) {
  struct Step {
    u32 mask;
    u64 reads;
  };
  static const base::Vector<Step> kSteps = [] {
    base::Vector<Step> out;
    const char* e = kPadScript;
    if (!e)
      return out;
    static const struct {
      const char* name;
      u32 mask;
    } kNames[] = {
        {"none", 0},
        {"cross", kCross},
        {"circle", kCircle},
        {"square", kSquare},
        {"triangle", kTriangle},
        {"up", kUp},
        {"down", kDown},
        {"left", kLeft},
        {"right", kRight},
        {"options", kOptions},
        {"l1", kL1},
        {"r1", kR1},
        {"touchpad", kTouchPad},
    };
    for (base::String spec(e), tok; !spec.empty();) {
      const size_t comma = spec.find(',');
      tok = spec.substr(0, comma);
      spec =
          comma == base::String::npos ? base::String() : spec.substr(comma + 1);
      const size_t colon = tok.find(':');
      const base::String name = tok.substr(0, colon);
      const u64 reads =
          colon == base::String::npos
              ? 30
              : std::strtoull(tok.c_str() + colon + 1, nullptr, 10);
      for (const auto& n : kNames)
        if (name == n.name) {
          out.push_back({n.mask, reads});
          break;
        }
    }
    if (!out.empty())
      BASE_LOGI("pad", "script: {} steps", out.size());
    return out;
  }();
  active = !kSteps.empty();
  if (!active)
    return 0;
  u64 at = g_read_seq;
  for (size_t i = 0; i < kSteps.size(); i++) {
    if (at < kSteps[i].reads) {
      static size_t last = ~size_t(0);
      if (last != i) {
        last = i;
        BASE_LOGI("pad", "script step {} mask={:#x} for {} reads", i,
                  kSteps[i].mask, (unsigned long long)kSteps[i].reads);
      }
      return kSteps[i].mask;
    }
    at -= kSteps[i].reads;
  }
  return 0;  // sequence done: hold neutral
}

u32 AutoSkipButtons() {
  {
    bool scripted = false;
    const u32 m = ScriptButtons(scripted);
    if (scripted)
      return m;
  }
  // Drive intro -> title -> menu -> a started run, then stop opening menus so
  // we stay in gameplay: drop Options once a run is likely underway (it would
  // open the pause menu) and keep only an occasional Cross to dismiss popups.
  // Never Circle/ Down so nothing cancels or moves off the default path. The
  // gameplay signal latches, so a brief pause flash won't restart the mashing.
  if (host::InGameplay())
    return 0;
  // DELTA_PAD_AUTOSKIP_STOP=N: stop after N reads; an idle-triggered attract
  // DEMO (Doom64) only plays once input goes quiet, so login passes then the
  // title idles.
  if (kAutoskipStop && g_read_seq > kAutoskipStop)
    return 0;
  // DELTA_PAD_AUTOSKIP_START=N: hold neutral for the first N reads. Some
  // engines (FOX/PT) lazily construct subsystem singletons on worker threads; a
  // progression input before that completes derefs a null component. Value is
  // title-timing dependent (PT needs a few thousand reads).
  if (g_read_seq < kAutoskipStart)
    return 0;
  // DELTA_PAD_AUTOSKIP_NOOPT: Cross only (Options derails Z-to-advance titles
  // like Undertale; Cross = Z drives straight in). DELTA_PAD_AUTOSKIP_NAV:
  // Options then Down+Cross (titles whose default cursor isn't "New Game", e.g.
  // Doom64). DELTA_PAD_AUTOSKIP_SWEEP: cycle every button ~40 reads each to
  // discover which input advances an unseen menu; the [pad] log correlates draw
  // jumps to buttons.
  if (kPadAutoskipSweep) {
    static const u32 kBtns[] = {kCross,  kOptions, kCircle, kTriangle,
                                kSquare, kDown,    kUp,     kLeft,
                                kRight,  kL1,      kR1,     kTouchPad};
    static const char* names[] = {"Cross",  "Options", "Circle", "Triangle",
                                  "Square", "Down",    "Up",     "Left",
                                  "Right",  "L1",      "R1",     "TouchPad"};
    u32 n = sizeof(kBtns) / sizeof(kBtns[0]);
    u32 slot = (u32)((g_read_seq / 60) % n);
    static u32 last_slot = 0xffffffff;
    if (slot != last_slot) {
      last_slot = slot;
      BASE_LOGI("sweep", "readSeq={} now pressing {}",
                (unsigned long long)g_read_seq, names[slot]);
    }
    u32 ph = g_read_seq % 60;
    return (ph < 30) ? kBtns[slot] : 0;  // hold ~30 reads, release ~30
  }
  // DELTA_PAD_AUTOSKIP_DOOM: Options a few times to leave the title, then Cross
  // only (re-pressing Options backs out to the title; Down is safe, cursor
  // defaults to New Game). Drives title -> New Game -> skill -> load.
  if (kPadAutoskipDoom) {
    u32 ph = g_read_seq % 30;
    if (g_read_seq < 240)
      return (ph < 8) ? kOptions : 0;  // leave the title
    return (ph < 8) ? kCross : 0;      // confirm down the menu
  }
  u32 phase = g_read_seq % 24;
  if (kPadAutoskipNav) {
    if (phase < 3)
      return kOptions;  // pass the "press start" title
    if (phase >= 6 && phase < 8)
      return kDown;  // move cursor down
    if (phase >= 11 && phase < 13)
      return kCross;
    if (phase >= 16 && phase < 18)
      return kDown;
    if (phase >= 21 && phase < 23)
      return kCross;
    return 0;
  }
  if (phase < 3)
    return kPadAutoskipNoopt ? kCross : kOptions;
  if (phase >= 8 && phase < 11)
    return kCross;
  if (phase >= 16 && phase < 19)
    return kCross;
  return 0;
}

// Adapter from the host pad (maps the SDL window keyboard; the Android app maps
// the on-screen touch gamepad). On by default for interactive play; set
// DELTA_PAD_KEYBOARD=0 to disable. DELTA_PAD_AUTOSKIP overrides it.
#if defined(DELTA_ANDROID_APP)
static const bool g_keyboard = true;
#else
#endif

// Symbolic name <-> Orbis bitmask table, shared by the script parser and
// tracer.
struct BtnName {
  u32 mask;
  const char* name;
};
constexpr BtnName kBtnNames[] = {
    {kCross, "cross"},     {kCircle, "circle"},
    {kSquare, "square"},   {kTriangle, "triangle"},
    {kOptions, "options"}, {kUp, "up"},
    {kDown, "down"},       {kLeft, "left"},
    {kRight, "right"},     {kL1, "l1"},
    {kR1, "r1"},           {kL2, "l2"},
    {kR2, "r2"},           {kL3, "l3"},
    {kR3, "r3"},           {kTouchPad, "touchpad"},
};

// Symbolic analog deflections for the script table: buttons alone cannot pass a
// first-person door (walking is the left stick); 0/255 are the uint8 extremes,
// 128 neutral, `up` = 0 on the PS4 y axis.
struct AxisName {
  const char* name;
  int lx, ly, rx, ry;  // -1: this step leaves that axis alone
};
constexpr AxisName kAxisNames[] = {
    {"lsup", -1, 0, -1, -1},   {"lsdown", -1, 255, -1, -1},
    {"lsleft", 0, -1, -1, -1}, {"lsright", 255, -1, -1, -1},
    {"rsup", -1, -1, -1, 0},   {"rsdown", -1, -1, -1, 255},
    {"rsleft", -1, -1, 0, -1}, {"rsright", -1, -1, 255, -1},
};

const AxisName* AxisByName(const base::String& name) {
  for (const auto& a : kAxisNames)
    if (name == a.name)
      return &a;
  return nullptr;
}

u32 ButtonMask(const base::String& name) {
  for (const auto& b : kBtnNames)
    if (name == b.name)
      return b.mask;
  if (AxisByName(name))
    return 0;  // an axis, reported by the caller instead
  BASE_LOGI("padscript", "unknown button '{}'", name.c_str());
  return 0;
}

base::String ButtonNames(u32 buttons) {
  base::String out;
  for (const auto& b : kBtnNames)
    if (buttons & b.mask) {
      if (!out.empty())
        out += '+';
      out += b.name;
    }
  return out.empty() ? "none" : out;
}

// DELTA_PAD_SCRIPT="<time>:<buttons>[:<holdMs>],...": press at scripted seconds
// after the first read (e.g. "12:cross,15:down+cross:200", holdMs default 150),
// OR'd into the read path's state. Stick deflections use the same syntax, so
// "20:lsup:3000" walks forward 3s (holdMs = travel/turn, deflection is full).
// An axis a step doesn't name keeps whatever the rest of the read path
// produced.
struct ScriptStep {
  double start, end;
  u32 buttons;
  int lx, ly, rx, ry;  // -1: leave alone
};

base::Vector<ScriptStep> ParseScript(const char* s) {
  base::Vector<ScriptStep> steps;
  const base::String in(s);
  // Two script formats share this env: time-keyed here, read-count-keyed in
  // scriptButtons(). Distinguish by what precedes the first colon (number vs
  // button name); feeding a count script here spammed unknown-button
  // complaints.
  size_t colon = in.find(':');
  if (colon == base::String::npos)
    return steps;
  for (size_t k = 0; k < colon; k++)
    if (!std::isdigit(static_cast<unsigned char>(in[k])) && in[k] != '.' &&
        in[k] != ' ')
      return steps;  // a name, not a time: this is the read-count format
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    base::String e =
        in.substr(i, comma == base::String::npos ? comma : comma - i);
    i = comma == base::String::npos ? in.size() : comma + 1;
    size_t c1 = e.find(':');
    if (c1 == base::String::npos)
      continue;
    double t = std::atof(e.substr(0, c1).c_str());
    size_t c2 = e.find(':', c1 + 1);
    base::String btns =
        e.substr(c1 + 1, c2 == base::String::npos ? c2 : c2 - c1 - 1);
    double hold_ms =
        c2 == base::String::npos ? 150.0 : std::atof(e.c_str() + c2 + 1);
    u32 mask = 0;
    int lx = -1, ly = -1, rx = -1, ry = -1;
    size_t j = 0;
    while (j < btns.size()) {
      size_t plus = btns.find('+', j);
      const base::String tok =
          btns.substr(j, plus == base::String::npos ? plus : plus - j);
      if (const AxisName* a = AxisByName(tok)) {
        if (a->lx >= 0)
          lx = a->lx;
        if (a->ly >= 0)
          ly = a->ly;
        if (a->rx >= 0)
          rx = a->rx;
        if (a->ry >= 0)
          ry = a->ry;
      } else {
        mask |= ButtonMask(tok);
      }
      j = plus == base::String::npos ? btns.size() : plus + 1;
    }
    // A stick-only step carries no button bits, so testing the mask alone would
    // silently drop every movement instruction.
    if (mask || lx >= 0 || ly >= 0 || rx >= 0 || ry >= 0)
      steps.push_back({t, t + hold_ms / 1000.0, mask, lx, ly, rx, ry});
  }
  return steps;
}

// The watch/poke experiments hang off the pad because that is where a title's
// per-frame heartbeat is. A title that opens a pad and then never reads it is
// exactly the case worth probing, so open counts as a start too.
void StartPadExperiments() {
  static const bool kStarted = [] {
    StartMemWatch();
    StartMemPoke();
    return true;
  }();
  (void)kStarted;
}

void FillPadState(PadData* d) {
  if (!d)
    return;
  StartPadExperiments();
  std::memset(d, 0, sizeof(*d));
  u32 buttons = 0;
  u8 lx = 128, ly = 128, rx = 128, ry = 128;
  host::PadKeys k;
  if (kPadAutoskip) {
    buttons = AutoSkipButtons();
  } else if (kPadKeyboard && host::PollKeyboardPad(k)) {
    if (k.cross)
      buttons |= kCross;
    if (k.circle)
      buttons |= kCircle;
    if (k.square)
      buttons |= kSquare;
    if (k.triangle)
      buttons |= kTriangle;
    if (k.up)
      buttons |= kUp;
    if (k.down)
      buttons |= kDown;
    if (k.left)
      buttons |= kLeft;
    if (k.right)
      buttons |= kRight;
    if (k.l1)
      buttons |= kL1;
    if (k.r1)
      buttons |= kR1;
    if (k.l2)
      buttons |= kL2;
    if (k.r2)
      buttons |= kR2;
    if (k.options)
      buttons |= kOptions;
    if (k.touchpad)
      buttons |= kTouchPad;
    lx = k.lx;
    ly = k.ly;
    rx = k.rx;
    ry = k.ry;
  }
  // Explore mode (DELTA_PAD_EXPLORE): once in gameplay, walk Isaac toward doors
  // so a headless run visits multiple rooms (to verify rendering beyond the
  // start room). Cycles direction every ~150 reads (up, right, down, left) on
  // the left stick.
  static u64 g_first_gameplay_seq = 0;
  if (kPadExplore && kPadAutoskip && host::InGameplay()) {
    if (!g_first_gameplay_seq)
      g_first_gameplay_seq = g_read_seq;
    u64 since = g_read_seq - g_first_gameplay_seq;
    // Walk up into the adjacent room and stop near its centre (a short burst),
    // then settle (hold neutral) so a clean, non-transition frame of a
    // non-start room can be captured. Tunable via DELTA_PAD_EXPLORE_READS.
    const u64 walk = kExploreReads;
    // Default walk right (the start room's exits are the side doors; up is the
    // hatch/wall). DELTA_PAD_EXPLORE_DIR: 0=right 1=left 2=up 3=down.
    const int dir = kExploreDir;
    // Continuous mode (DELTA_PAD_EXPLORE_CONT): keep moving (circle) so Isaac
    // dodges and survives in a hostile room long enough to capture a settled
    // non-start room.
    if (kPadExploreCont) {
      // Longer bursts (default 200 reads/dir) so Isaac actually crosses the
      // room and transits a door, not just jitter in place. Tunable via the
      // same DELTA_PAD_EXPLORE_READS knob.
      u64 burst = walk ? walk : 200ull;
      u64 ph = (since / burst) % 4;  // right, down, left, up
      if (ph == 0)
        lx = 255;
      else if (ph == 1)
        ly = 255;
      else if (ph == 2)
        lx = 0;
      else
        ly = 0;
    } else if (since < walk) {
      if (dir == 0)
        lx = 255;
      else if (dir == 1)
        lx = 0;
      else if (dir == 2)
        ly = 0;
      else
        ly = 255;
    }
  }
  static const base::Vector<ScriptStep> kScript =
      kPadScript ? ParseScript(kPadScript) : base::Vector<ScriptStep>{};
  static const auto kScriptT0 = base::TimeTicks::Now();
  if (!kScript.empty()) {
    double t = (base::TimeTicks::Now() - kScriptT0).InSecondsF();
    for (const auto& st : kScript)
      if (t >= st.start && t < st.end) {
        buttons |= st.buttons;
        // A named axis REPLACES the neutral the read path filled in; OR-ing
        // would be meaningless on a 0..255 deflection where 128 is centre.
        if (st.lx >= 0)
          lx = static_cast<u8>(st.lx);
        if (st.ly >= 0)
          ly = static_cast<u8>(st.ly);
        if (st.rx >= 0)
          rx = static_cast<u8>(st.rx);
        if (st.ry >= 0)
          ry = static_cast<u8>(st.ry);
      }
  }

  if (kPadTrace) {
    static u32 last_traced = 0;
    static u32 last_sticks = ~0u;
    const u32 sticks = u32(lx) | u32(ly) << 8 | u32(rx) << 16 | u32(ry) << 24;
    static bool first = true;
    if (first || buttons != last_traced || sticks != last_sticks) {
      first = false;
      last_traced = buttons;
      last_sticks = sticks;
      BASE_LOGI("padtrace", "readSeq={} buttons={:#x} {} ls=({},{}) rs=({},{})",
                (unsigned long long)g_read_seq, buttons,
                ButtonNames(buttons).c_str(), lx, ly, rx, ry);
    }
  }

  d->buttons = buttons;
  d->left_stick = {lx, ly};
  d->right_stick = {rx, ry};
  d->analog_buttons = {static_cast<u8>((buttons & kL2) ? 255 : 0),
                       static_cast<u8>((buttons & kR2) ? 255 : 0)};
  d->orientation = {0, 0, 0, 1};
  d->connected = true;
  d->connected_count = 1;
  d->timestamp =
      (base::TimeTicks::Now() - base::TimeTicks()).InMicroseconds();
  ++g_read_seq;
  if (kPadAutoskip && (g_read_seq % 600 == 1))
    BASE_LOGI("pad", "readSeq={} buttons={:#x}", (unsigned long long)g_read_seq,
              buttons);
}

}  // namespace

int scePadClose() {
  // Single fixed handle with no per-open state; nothing to tear down.
  return 0;
}

int scePadConnectPort() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDeviceClassGetExtendedInformation() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDeviceClassParseData() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDeviceOpen() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDisableVibration() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDisconnectDevice() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadDisconnectPort() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadEnableAutoDetect() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadEnableUsbConnection() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetCapability() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetControllerInformation(int handle, void* p_info) {
  if (auto* info = static_cast<PadControllerInformation*>(p_info)) {
    std::memset(info, 0, sizeof(*info));
    info->touchpad_density = 44.86f;
    info->touch_resolution_x = 1920;
    info->touch_resolution_y = 942;
    info->stick_dead_zone_left = 12;
    info->stick_dead_zone_right = 12;
    info->connection_type = 0;  // local
    info->connected_count = 1;
    info->connected = true;
    info->device_class = 0;  // STANDARD (DualShock4)
  }
  return 0;
}

int scePadGetDataInternal() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetDeviceInfo() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetHandle(int user_id, int type, int index) {
  return 1;  // single fixed handle
}

int scePadGetVersionInfo() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadInit() {
  // No device to bring up: the HLE pad is always available. Accept silently
  // (titles call this once at boot; the unimplemented log was misleading since
  // the pad is fully serviced through the read/open path below).
  return 0;
}

int scePadIsLightBarBaseBrightnessControllable() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadMbusInit() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadOpen(int user_id, int type, int index, const void* param) {
  // Worth tracing on its own: a title that never opens the pad is stuck before
  // its input path, which the read trace below cannot tell apart from a title
  // that opened one and is ignoring it.
  if (kPadTrace)
    BASE_LOGI("padtrace", "open user={} type={} index={}", user_id, type,
              index);
  StartPadExperiments();
  return 1;  // positive handle = success
}

int scePadRead(int handle, void* data, int num) {
  if (num <= 0)
    return 0;
  auto* d = static_cast<PadData*>(data);
  // Return one fresh sample (we don't keep history); games read [0].
  FillPadState(&d[0]);
  return 1;  // number of samples read
}

int scePadReadState(int handle, void* data) {
  // Whether a title polls the pad at all, and how often. A title sitting on a
  // screen that renders nothing is either waiting for input or not asking for
  // it, and those want opposite fixes.
  static base::Atomic<u64> reads{0};
  const u64 n = reads.fetch_add(1);
  if (n < 2 || (n % 3000) == 0)
    BASE_LOGI("pad", "readState #{}", (unsigned long long)n);
  FillPadState(static_cast<PadData*>(data));
  return 0;
}

int scePadResetLightBar() {
  // No light bar to drive; accept silently (mirrors scePadSetLightBar).
  return 0;
}

int scePadResetOrientation() {
  // Orientation is reported as identity every read, so a reset is a no-op.
  return 0;
}

int scePadSetAngularVelocityDeadbandState() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetAutoPowerOffCount() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetButtonRemappingInfo() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetConnection() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetForceIntercepted() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetLightBar() {
  // Accepted silently: no light bar to drive, and titles (SotC) set it every
  // frame, and the unimplemented log became per-frame spam.
  return 0;
}

int scePadSetLightBarBaseBrightness() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetLightBarBlinking() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetMotionSensorState() {
  // Motion data is synthesized (identity orientation, zero accel/gyro);
  // toggling the sensor has no backing device, so accept silently.
  return 0;
}

int scePadSetTiltCorrectionState() {
  LOG_UNIMPLEMENTED;
  return 0;
}

// ScePadVibrationParam: two 0..255 motor intensities (large = low-freq, small =
// high-freq). Drive the active controller's haptics; logging is omitted because
// games call this every frame and the spam dominated the trace.
struct ScePadVibrationParam {
  u8 large_motor;
  u8 small_motor;
};
int scePadSetVibration(int /*handle*/, const ScePadVibrationParam* param) {
  if (param)
    host::SetRumble(param->large_motor, param->small_motor);
  return 0;
}

int scePadShareOutputData() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSwitchConnection() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetProcessPrivilege() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadOutputReport() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadEnableSpecificDeviceClass() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetProcessPrivilegeOfButtonRemapping() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadVirtualDeviceInsertData() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadVirtualDeviceGetRemoteSetting() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadVirtualDeviceAddDevice() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadVirtualDeviceDeleteDevice() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetFeatureReport() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadReadExt(int handle, void* data, int num) {
  if (num <= 0)
    return 0;
  FillPadState(static_cast<PadData*>(data));
  return 1;
}

int scePadGetBluetoothAddress() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int unk_UeUUvNOgXKU() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadOpenExt() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetMotionSensorPosition() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadIsBlasterConnected() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetExtensionReport() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetSphereRadius() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetProcessFocus() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadReadBlasterForTracker() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadStopRecording() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetDeviceId() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetExtControllerInformation() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetLightBarForTracker() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int unk_ickjfjk9okM() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadResetOrientationForTracker() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetIdleCount() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetMotionTimerUnit() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadIsDS4Connected() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetLoginUserNumber() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadIsValidHandle(int handle) {
  return handle > 0 ? 1 : 0;
}

int scePadMbusTerm() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetLicenseControllerInformation() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetFeatureReport() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetUserColor() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadVertualDeviceAddDevice() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetExtensionUnitInfo() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetInfo() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadReadForTracker() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadReadHistory() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadGetInfoByPortType() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadIsMoveConnected() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadReadStateExt(int handle, void* data) {
  FillPadState(static_cast<PadData*>(data));
  return 0;
}

int unk_7xA_hFtvBCA() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadOpenExt2() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetVibrationForce() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadStartRecording() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadIsMoveReproductionModel() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadResetLightBarAllByPortType() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadEnableExtensionPort() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadResetLightBarAll() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetVrTrackingMode() {
  LOG_UNIMPLEMENTED;
  return 0;
}
