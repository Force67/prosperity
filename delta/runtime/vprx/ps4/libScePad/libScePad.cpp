
// Copyright (C) Force67

// This file was generated on 10/12/2019

#include "../../vprx.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gfx/gfx.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kMemWatch, "DELTA_MEMWATCH", nullptr);
DELTA_OPTION(const char *, kMemPoke, "DELTA_MEMPOKE", nullptr);
DELTA_OPTION(const char *, kPadScript, "DELTA_PAD_SCRIPT", nullptr);
DELTA_OPTION(uint64_t, kAutoskipStop, "DELTA_PAD_AUTOSKIP_STOP", 0);
DELTA_OPTION(uint64_t, kAutoskipStart, "DELTA_PAD_AUTOSKIP_START", 0);
DELTA_OPTION(bool, kPadKeyboard, "DELTA_PAD_KEYBOARD", true);
DELTA_OPTION(uint64_t, kExploreReads, "DELTA_PAD_EXPLORE_READS", 130);
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

// HLE controller. We report a single connected DS4 on the open handle and feed
// the game a neutral pad state (sticks centered, no buttons). With
// DELTA_PAD_AUTOSKIP=1 we pulse the confirm/back/start buttons periodically so a
// headless run can advance past the intro/title into the menu for verification.
namespace {

// Orbis button bitmasks (ScePadButtonDataOffset).
enum : uint32_t {
  kL3 = 0x0002, kR3 = 0x0004, kOptions = 0x0008,
  kUp = 0x0010, kRight = 0x0020, kDown = 0x0040, kLeft = 0x0080,
  kL2 = 0x0100, kR2 = 0x0200, kL1 = 0x0400, kR1 = 0x0800,
  kTriangle = 0x1000, kCircle = 0x2000, kCross = 0x4000, kSquare = 0x8000,
  kTouchPad = 0x100000,
};

struct AnalogStick { uint8_t x, y; };
struct AnalogButtons { uint8_t l2, r2; };
struct FQuaternion { float x, y, z, w; };
struct FVector3 { float x, y, z; };
struct PadTouch { uint16_t x, y; uint8_t id; uint8_t reserve[3]; };
struct PadTouchData {
  uint8_t touchNum; uint8_t reserve[3]; uint32_t reserve1; PadTouch touch[2];
};
struct PadExtUnitData { uint32_t id; uint8_t reserve; uint8_t dataLen; uint8_t data[10]; };

// ScePadData: offsets verified against the orbis layout (connected@0x4C,
// timestamp@0x50). Written into the game's buffer on read.
struct PadData {
  uint32_t buttons;             // 0x00
  AnalogStick leftStick;        // 0x04
  AnalogStick rightStick;       // 0x06
  AnalogButtons analogButtons;  // 0x08
  uint8_t pad0[2];              // 0x0A
  FQuaternion orientation;      // 0x0C
  FVector3 acceleration;        // 0x1C
  FVector3 angularVelocity;     // 0x28
  PadTouchData touchData;       // 0x34
  bool connected;               // 0x4C
  uint8_t pad1[3];
  uint64_t timestamp;           // 0x50
  PadExtUnitData extUnit;       // 0x58
  uint8_t connectedCount;       // 0x68
  uint8_t reserve[2];
  uint8_t deviceUniqueDataLen;  // 0x6B
  uint8_t deviceUniqueData[12]; // 0x6C
};
static_assert(sizeof(PadData) >= 0x78, "PadData layout");

struct PadControllerInformation {
  float touchpadDensity;        // 0x00
  uint16_t touchResolutionX;    // 0x04
  uint16_t touchResolutionY;    // 0x06
  uint8_t stickDeadZoneLeft;    // 0x08
  uint8_t stickDeadZoneRight;   // 0x09
  uint8_t connectionType;       // 0x0A
  uint8_t connectedCount;       // 0x0B
  bool connected;               // 0x0C
  uint8_t deviceClass;          // 0x0D (ORBIS_PAD_DEVICE_CLASS_STANDARD = 0)
  uint8_t reserve[8];
};

uint64_t g_readSeq = 0;

// DELTA_MEMWATCH=addr[,addr...]: spawn a background thread that prints the qword
// at each guest VA every ~250ms with a wall-clock delta, so the lifecycle of a
// guest global (e.g. a lazily-constructed singleton pointer transitioning
// null->non-null) can be correlated with boot/crash timing. Generic diagnostic;
// guest memory is identity-mapped so a fixed VA reads as a host pointer. Started
// once from the first pad read (guaranteed to run for any title that polls input).
void startMemWatch() {
  const char *e = kMemWatch;
  if (!e) return;
  std::vector<uint64_t> addrs;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ') p++;
    char *end = nullptr;
    uint64_t v = std::strtoull(p, &end, 0);
    if (end == p) break;
    if (v >= 0x1000) addrs.push_back(v);
    p = end;
  }
  if (addrs.empty()) return;
  std::thread([addrs] {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> last(addrs.size(), 0xdeadbeefdeadbeefull);
    for (;;) {
      double t = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count();
      bool any = false;
      for (size_t i = 0; i < addrs.size(); i++) {
        uint64_t cur = *reinterpret_cast<volatile uint64_t *>(addrs[i]);
        if (cur != last[i]) {
          std::fprintf(stderr, "[memwatch] t=%.2f  *%#llx: %016llx -> %016llx\n",
                       t, (unsigned long long)addrs[i],
                       (unsigned long long)last[i], (unsigned long long)cur);
          last[i] = cur;
          any = true;
        }
      }
      (void)any;
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }).detach();
}

// DELTA_MEMPOKE=<spec>[,<spec>...]: runtime guest-memory patch experiment. Each
// spec = addr:width:value[:delayMs], colon-separated:
//   addr   literal hex VA (e.g. 0x201402ee2d00), OR *PTR+OFF meaning "read the
//          u64 at PTR, add OFF" -- resolves a singleton object every apply (so a
//          pointer that is null until constructed is followed once it is live).
//   width  1/2/4/8 bytes.
//   value  hex (0x..) or decimal.
//   delayMs optional; hold neutral this long after the first pad read before the
//          first write (default 0).
// Writes are RE-APPLIED every 200ms so the value is HELD against the guest's own
// updates -- an experiment can pin a guest global (e.g. force a load counter to
// 0, or set a completion event word) to probe what a value change triggers.
// Guest memory is identity-mapped, so a resolved VA is written as a host pointer.
struct PokeSpec {
  bool indirect = false;
  uint64_t ptrAddr = 0;   // for indirect: address holding the object pointer
  uint64_t off = 0;       // offset added to literal addr or to *ptrAddr
  int width = 8;
  uint64_t value = 0;
  uint64_t delayMs = 0;
};

void startMemPoke() {
  const char *e = kMemPoke;
  if (!e) return;
  std::vector<PokeSpec> specs;
  const std::string in(e);
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    std::string s = in.substr(i, comma == std::string::npos ? comma : comma - i);
    i = comma == std::string::npos ? in.size() : comma + 1;
    // split s on ':' but the addr field may itself contain no ':'
    std::vector<std::string> f;
    size_t j = 0;
    while (j <= s.size()) {
      size_t c = s.find(':', j);
      f.push_back(s.substr(j, c == std::string::npos ? c : c - j));
      if (c == std::string::npos) break;
      j = c + 1;
    }
    if (f.size() < 3) continue;
    PokeSpec p;
    std::string addr = f[0];
    if (!addr.empty() && addr[0] == '*') {
      p.indirect = true;
      size_t plus = addr.find('+');
      p.ptrAddr = std::strtoull(addr.c_str() + 1, nullptr, 0);
      p.off = plus == std::string::npos ? 0 : std::strtoull(addr.c_str() + plus + 1, nullptr, 0);
    } else {
      p.off = std::strtoull(addr.c_str(), nullptr, 0);
    }
    p.width = std::atoi(f[1].c_str());
    p.value = std::strtoull(f[2].c_str(), nullptr, 0);
    if (f.size() >= 4) p.delayMs = std::strtoull(f[3].c_str(), nullptr, 0);
    if (p.width == 1 || p.width == 2 || p.width == 4 || p.width == 8)
      specs.push_back(p);
  }
  if (specs.empty()) return;
  std::thread([specs] {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<bool> announced(specs.size(), false);
    for (;;) {
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
      for (size_t k = 0; k < specs.size(); k++) {
        const auto &p = specs[k];
        if (ms < (double)p.delayMs) continue;
        uint64_t target;
        if (p.indirect) {
          uint64_t obj = *reinterpret_cast<volatile uint64_t *>(p.ptrAddr);
          if (obj < 0x1000) continue;  // singleton not constructed yet
          target = obj + p.off;
        } else {
          target = p.off;
        }
        if (target < 0x1000) continue;
        switch (p.width) {
        case 1: *reinterpret_cast<volatile uint8_t *>(target)  = (uint8_t)p.value; break;
        case 2: *reinterpret_cast<volatile uint16_t *>(target) = (uint16_t)p.value; break;
        case 4: *reinterpret_cast<volatile uint32_t *>(target) = (uint32_t)p.value; break;
        case 8: *reinterpret_cast<volatile uint64_t *>(target) = p.value; break;
        }
        if (!announced[k]) {
          announced[k] = true;
          std::fprintf(stderr, "[mempoke] t=%.0fms first write %#llx <- %#llx (w%d)\n",
                       ms, (unsigned long long)target,
                       (unsigned long long)p.value, p.width);
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }).detach();
}

// Auto-skip pulse: advance the intro/title/menus into actual gameplay for a
// headless verification run. Gated by g_autoskip at the call site, where it
// takes precedence over the keyboard. NEVER pulse Circle (back/cancel) together
// with Cross (confirm): pressing both each cycle confirms then immediately backs
// out, so menus never advance (this kept the headless run stuck on the title).
// Sequence: Options first (title "PRESS OPTIONS" -> main menu), then Cross to
// confirm "New Run"/save-slot/character-select, with the occasional Down to move
// the menu cursor. Buttons pulse with gaps so menus see clean press edges.
// DELTA_PAD_SCRIPT="<name>:<reads>[,<name>:<reads>]...": replay one exact button
// sequence, then hold neutral. The heuristic pulses below can only reach menus
// whose default entry is the one we want; a scripted run drives an arbitrary
// path (e.g. title -> Play -> world list -> load) for headless verification.
// Names are the button constants below plus "none" for a gap.
uint32_t scriptButtons(bool &active) {
  struct Step { uint32_t mask; uint64_t reads; };
  static const std::vector<Step> steps = [] {
    std::vector<Step> out;
    const char *e = kPadScript;
    if (!e)
      return out;
    static const struct { const char *name; uint32_t mask; } kNames[] = {
        {"none", 0},        {"cross", kCross},   {"circle", kCircle},
        {"square", kSquare}, {"triangle", kTriangle}, {"up", kUp},
        {"down", kDown},    {"left", kLeft},     {"right", kRight},
        {"options", kOptions}, {"l1", kL1},      {"r1", kR1},
        {"touchpad", kTouchPad},
    };
    for (std::string spec(e), tok; !spec.empty();) {
      const size_t comma = spec.find(',');
      tok = spec.substr(0, comma);
      spec = comma == std::string::npos ? std::string() : spec.substr(comma + 1);
      const size_t colon = tok.find(':');
      const std::string name = tok.substr(0, colon);
      const uint64_t reads =
          colon == std::string::npos ? 30 : std::strtoull(tok.c_str() + colon + 1, nullptr, 10);
      for (const auto &n : kNames)
        if (name == n.name) {
          out.push_back({n.mask, reads});
          break;
        }
    }
    if (!out.empty())
      std::fprintf(stderr, "[pad] script: %zu steps\n", out.size());
    return out;
  }();
  active = !steps.empty();
  if (!active)
    return 0;
  uint64_t at = g_readSeq;
  for (size_t i = 0; i < steps.size(); i++) {
    if (at < steps[i].reads) {
      static size_t last = ~size_t(0);
      if (last != i) {
        last = i;
        std::fprintf(stderr, "[pad] script step %zu mask=%#x for %llu reads\n", i,
                     steps[i].mask, (unsigned long long)steps[i].reads);
      }
      return steps[i].mask;
    }
    at -= steps[i].reads;
  }
  return 0;  // sequence done: hold neutral
}

uint32_t autoSkipButtons() {
  {
    bool scripted = false;
    const uint32_t m = scriptButtons(scripted);
    if (scripted)
      return m;
  }
  // Drive intro -> title -> menu -> a started run, then STOP opening menus so we
  // stay in gameplay (for headless verification). Options opens the menu from the
  // "PRESS OPTIONS" title; Cross confirms New Run / save-slot / character-select
  // (default entries pre-highlighted). Once a run is likely underway we drop
  // Options (it would open the pause menu and Cross would navigate us back out),
  // keeping only an occasional Cross to dismiss incidental item/pickup popups.
  // Never Circle/Down so nothing cancels or moves off the default path.
  // Once the GPU renderer reports sustained gameplay, stop opening menus (Options
  // would pause and Cross would navigate us back out); just hold neutral so we
  // stay in the run. The signal latches, so a brief pause flash won't restart the
  // menu mashing.
  if (gfx::inGameplay())
    return 0;
  // DELTA_PAD_AUTOSKIP_STOP=N: stop pulsing after N reads and hold neutral. Some
  // titles (Doom64) need a few button presses to pass the login/title, but then an
  // idle-triggered attract DEMO only plays if input goes quiet -- continuous pulsing
  // suppresses it. Stop after N so login passes, then the title idles into the demo.
  if (kAutoskipStop && g_readSeq > kAutoskipStop)
    return 0;
  // DELTA_PAD_AUTOSKIP_START=N: hold neutral for the first N pad reads before any
  // pulsing. Some engines (FOX/PT) lazily construct subsystem singletons on a
  // worker/job thread during boot; feeding a progression input (which triggers a
  // fade/scene transition) before that construction completes dereferences a null
  // component pointer. A warm-up delay lets init finish so the first triggered
  // transition finds a live object. Generic; the value is title/hardware-timing
  // dependent (PT: the renderer needs a few thousand reads to spin up).
  if (g_readSeq < kAutoskipStart)
    return 0;
  // DELTA_PAD_AUTOSKIP_NOOPT: don't pulse Options, only Cross. The Options pulse is
  // for Isaac's "PRESS OPTIONS" title; in Undertale (and other Z-to-advance titles)
  // Options opens the config menu and derails progression, so Cross-only (= Z, the
  // confirm/advance button) drives the intro/title/dialogue straight into the game.
  // DELTA_PAD_AUTOSKIP_NAV: vertical menu navigation. Pulse Options (pass a title),
  // then Down (move the cursor down a list) interleaved with Cross (confirm). Needed
  // for titles whose default cursor is not on "New Game" (e.g. Doom64's KEX menu),
  // where Isaac's Cross-only never reaches the start entry.
  // DELTA_PAD_AUTOSKIP_SWEEP: cycle through every button (one at a time, ~40 reads
  // each with gaps for clean press edges) to discover which input advances a title
  // whose menu we cannot see. The [pad] log prints the current button so a draw-
  // count jump (level load) can be correlated to the button that caused it.
  if (kPadAutoskipSweep) {
    static const uint32_t btns[] = {kCross, kOptions, kCircle, kTriangle, kSquare,
                                    kDown, kUp, kLeft, kRight, kL1, kR1, kTouchPad};
    static const char *names[] = {"Cross", "Options", "Circle", "Triangle", "Square",
                                   "Down", "Up", "Left", "Right", "L1", "R1", "TouchPad"};
    uint32_t n = sizeof(btns) / sizeof(btns[0]);
    uint32_t slot = (uint32_t)((g_readSeq / 60) % n);
    static uint32_t lastSlot = 0xffffffff;
    if (slot != lastSlot) {
      lastSlot = slot;
      std::fprintf(stderr, "[sweep] readSeq=%llu now pressing %s\n",
                   (unsigned long long)g_readSeq, names[slot]);
    }
    uint32_t ph = g_readSeq % 60;
    return (ph < 30) ? btns[slot] : 0;  // hold ~30 reads, release ~30
  }
  // DELTA_PAD_AUTOSKIP_DOOM: Doom64's title/menu flow. Press Options ("START")
  // a few times to leave the title screen, then ONLY Cross (confirm) from then
  // on. Crucially it must never re-press Options once in the menu (that backs
  // out, looping at the title) and never Down (the cursor defaults to "New
  // Game"). Drives title -> New Game -> skill -> level load.
  if (kPadAutoskipDoom) {
    uint32_t ph = g_readSeq % 30;
    if (g_readSeq < 240) return (ph < 8) ? kOptions : 0;  // leave the title
    return (ph < 8) ? kCross : 0;                          // confirm down the menu
  }
  uint32_t phase = g_readSeq % 24;
  if (kPadAutoskipNav) {
    if (phase < 3) return kOptions;            // pass the "press start" title
    if (phase >= 6 && phase < 8) return kDown;  // move cursor down
    if (phase >= 11 && phase < 13) return kCross;
    if (phase >= 16 && phase < 18) return kDown;
    if (phase >= 21 && phase < 23) return kCross;
    return 0;
  }
  if (phase < 3) return kPadAutoskipNoopt ? kCross : kOptions;
  if (phase >= 8 && phase < 11) return kCross;
  if (phase >= 16 && phase < 19) return kCross;
  return 0;
}

// Adapter from the gfx pad (maps the SDL window keyboard; the Android app maps
// the on-screen touch gamepad). On by default for interactive play; set
// DELTA_PAD_KEYBOARD=0 to disable. DELTA_PAD_AUTOSKIP overrides it.
#if defined(DELTA_ANDROID_APP)
static const bool g_keyboard = true;
#else
#endif

// Symbolic name <-> Orbis bitmask table, shared by the script parser and tracer.
struct BtnName { uint32_t mask; const char *name; };
constexpr BtnName kBtnNames[] = {
    {kCross, "cross"},    {kCircle, "circle"}, {kSquare, "square"},
    {kTriangle, "triangle"}, {kOptions, "options"}, {kUp, "up"},
    {kDown, "down"},      {kLeft, "left"},     {kRight, "right"},
    {kL1, "l1"},          {kR1, "r1"},         {kL2, "l2"},
    {kR2, "r2"},          {kL3, "l3"},         {kR3, "r3"},
    {kTouchPad, "touchpad"},
};

uint32_t buttonMask(const std::string &name) {
  for (const auto &b : kBtnNames)
    if (name == b.name) return b.mask;
  std::fprintf(stderr, "[padscript] unknown button '%s'\n", name.c_str());
  return 0;
}

std::string buttonNames(uint32_t buttons) {
  std::string out;
  for (const auto &b : kBtnNames)
    if (buttons & b.mask) { if (!out.empty()) out += '+'; out += b.name; }
  return out.empty() ? "none" : out;
}

// DELTA_PAD_SCRIPT="<time>:<buttons>[:<holdMs>],..." presses buttons at scripted
// times (seconds after the first pad read) so a headless run can be driven past
// intros/menus into gameplay. Buttons are symbolic (see kBtnNames) and combine
// with '+', e.g. "12:cross,15:down+cross:200". holdMs defaults to 150. The
// scripted buttons are OR'd into whatever state the read path produced.
struct ScriptStep { double start, end; uint32_t buttons; };

std::vector<ScriptStep> parseScript(const char *s) {
  std::vector<ScriptStep> steps;
  const std::string in(s);
  size_t i = 0;
  while (i < in.size()) {
    size_t comma = in.find(',', i);
    std::string e = in.substr(i, comma == std::string::npos ? comma : comma - i);
    i = comma == std::string::npos ? in.size() : comma + 1;
    size_t c1 = e.find(':');
    if (c1 == std::string::npos) continue;
    double t = std::atof(e.substr(0, c1).c_str());
    size_t c2 = e.find(':', c1 + 1);
    std::string btns =
        e.substr(c1 + 1, c2 == std::string::npos ? c2 : c2 - c1 - 1);
    double holdMs = c2 == std::string::npos ? 150.0 : std::atof(e.c_str() + c2 + 1);
    uint32_t mask = 0;
    size_t j = 0;
    while (j < btns.size()) {
      size_t plus = btns.find('+', j);
      mask |= buttonMask(
          btns.substr(j, plus == std::string::npos ? plus : plus - j));
      j = plus == std::string::npos ? btns.size() : plus + 1;
    }
    if (mask) steps.push_back({t, t + holdMs / 1000.0, mask});
  }
  return steps;
}

// The watch/poke experiments hang off the pad because that is where a title's
// per-frame heartbeat is. A title that opens a pad and then never reads it is
// exactly the case worth probing, so open counts as a start too.
void startPadExperiments() {
  static const bool started = [] { startMemWatch(); startMemPoke(); return true; }();
  (void)started;
}

void fillPadState(PadData *d) {
  if (!d) return;
  startPadExperiments();
  std::memset(d, 0, sizeof(*d));
  uint32_t buttons = 0;
  uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
  gfx::PadKeys k;
  if (kPadAutoskip) {
    buttons = autoSkipButtons();
  } else if (kPadKeyboard && gfx::pollKeyboardPad(k)) {
    if (k.cross) buttons |= kCross;
    if (k.circle) buttons |= kCircle;
    if (k.square) buttons |= kSquare;
    if (k.triangle) buttons |= kTriangle;
    if (k.up) buttons |= kUp;
    if (k.down) buttons |= kDown;
    if (k.left) buttons |= kLeft;
    if (k.right) buttons |= kRight;
    if (k.l1) buttons |= kL1;
    if (k.r1) buttons |= kR1;
    if (k.l2) buttons |= kL2;
    if (k.r2) buttons |= kR2;
    if (k.options) buttons |= kOptions;
    if (k.touchpad) buttons |= kTouchPad;
    lx = k.lx; ly = k.ly; rx = k.rx; ry = k.ry;
  }
  // Explore mode (DELTA_PAD_EXPLORE): once in gameplay, walk Isaac toward doors so a
  // headless run visits multiple rooms (to verify rendering beyond the start room).
  // Cycles direction every ~150 reads (up, right, down, left) on the left stick.
  static uint64_t g_firstGameplaySeq = 0;
  if (kPadExplore && kPadAutoskip && gfx::inGameplay()) {
    if (!g_firstGameplaySeq) g_firstGameplaySeq = g_readSeq;
    uint64_t since = g_readSeq - g_firstGameplaySeq;
    // Walk up into the adjacent room and stop near its centre (a short burst), then
    // settle (hold neutral) so a clean, non-transition frame of a non-start room can
    // be captured. Tunable via DELTA_PAD_EXPLORE_READS.
    const uint64_t walk = kExploreReads;
    // Default walk right (the start room's exits are the side doors; up is the
    // hatch/wall). DELTA_PAD_EXPLORE_DIR: 0=right 1=left 2=up 3=down.
    const int dir = kExploreDir;
    // Continuous mode (DELTA_PAD_EXPLORE_CONT): keep moving (circle) so Isaac dodges
    // and survives in a hostile room long enough to capture a settled non-start room.
    if (kPadExploreCont) {
      // Longer bursts (default 200 reads/dir) so Isaac actually crosses the room
      // and transits a door, not just jitter in place. Tunable via the same
      // DELTA_PAD_EXPLORE_READS knob.
      uint64_t burst = walk ? walk : 200ull;
      uint64_t ph = (since / burst) % 4;  // right, down, left, up
      if (ph==0) lx=255; else if (ph==1) ly=255; else if (ph==2) lx=0; else ly=0;
    } else if (since < walk) {
      if (dir==0) lx=255; else if (dir==1) lx=0; else if (dir==2) ly=0; else ly=255;
    }
  }
  static const std::vector<ScriptStep> g_script =
      kPadScript ? parseScript(kPadScript) : std::vector<ScriptStep>{};
  static const auto g_scriptT0 = std::chrono::steady_clock::now();
  if (!g_script.empty()) {
    double t = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - g_scriptT0).count();
    for (const auto &st : g_script)
      if (t >= st.start && t < st.end) buttons |= st.buttons;
  }

  if (kPadTrace) {
    static uint32_t lastTraced = 0;
    static bool first = true;
    if (first || buttons != lastTraced) {
      first = false;
      lastTraced = buttons;
      std::fprintf(stderr, "[padtrace] readSeq=%llu buttons=%#x %s\n",
                   (unsigned long long)g_readSeq, buttons,
                   buttonNames(buttons).c_str());
    }
  }

  d->buttons = buttons;
  d->leftStick = {lx, ly};
  d->rightStick = {rx, ry};
  d->analogButtons = {static_cast<uint8_t>((buttons & kL2) ? 255 : 0),
                      static_cast<uint8_t>((buttons & kR2) ? 255 : 0)};
  d->orientation = {0, 0, 0, 1};
  d->connected = true;
  d->connectedCount = 1;
  d->timestamp = ++g_readSeq;
  if (kPadAutoskip && (g_readSeq % 600 == 1))
    std::fprintf(stderr, "[pad] readSeq=%llu buttons=%#x\n",
                 (unsigned long long)g_readSeq, buttons);
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

int scePadGetControllerInformation(int handle, void *pInfo) {
  if (auto *info = static_cast<PadControllerInformation *>(pInfo)) {
    std::memset(info, 0, sizeof(*info));
    info->touchpadDensity = 44.86f;
    info->touchResolutionX = 1920;
    info->touchResolutionY = 942;
    info->stickDeadZoneLeft = 0;
    info->stickDeadZoneRight = 0;
    info->connectionType = 0;  // local
    info->connectedCount = 1;
    info->connected = true;
    info->deviceClass = 0;  // STANDARD (DualShock4)
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

int scePadGetHandle(int userId, int type, int index) {
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

int scePadOpen(int userId, int type, int index, const void *param) {
  // Worth tracing on its own: a title that never opens the pad is stuck before
  // its input path, which the read trace below cannot tell apart from a title
  // that opened one and is ignoring it.
  if (kPadTrace)
    std::fprintf(stderr, "[padtrace] open user=%d type=%d index=%d\n", userId,
                 type, index);
  startPadExperiments();
  return 1;  // positive handle = success
}

int scePadRead(int handle, void *data, int num) {
  if (num <= 0) return 0;
  auto *d = static_cast<PadData *>(data);
  // Return one fresh sample (we don't keep history); games read [0].
  fillPadState(&d[0]);
  return 1;  // number of samples read
}

int scePadReadState(int handle, void *data) {
  fillPadState(static_cast<PadData *>(data));
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
  // frame -- the unimplemented log became per-frame spam.
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
  // Motion data is synthesized (identity orientation, zero accel/gyro); toggling
  // the sensor has no backing device, so accept silently.
  return 0;
}

int scePadSetTiltCorrectionState() {
  LOG_UNIMPLEMENTED;
  return 0;
}

// ScePadVibrationParam: two 0..255 motor intensities (large = low-freq, small =
// high-freq). Drive the active controller's haptics; logging is omitted because
// games call this every frame and the spam dominated the trace.
struct ScePadVibrationParam { uint8_t largeMotor; uint8_t smallMotor; };
int scePadSetVibration(int /*handle*/, const ScePadVibrationParam *param) {
  if (param)
    gfx::setRumble(param->largeMotor, param->smallMotor);
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

int scePadReadExt(int handle, void *data, int num) {
  if (num <= 0) return 0;
  fillPadState(static_cast<PadData *>(data));
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

int scePadReadStateExt(int handle, void *data) {
  fillPadState(static_cast<PadData *>(data));
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
