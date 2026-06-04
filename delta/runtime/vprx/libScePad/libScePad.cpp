
// Copyright (C) Force67

// This file was generated on 10/12/2019

#include "../vprx.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gfx/gfx.h"

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

// Auto-skip pulse: advance the intro/title/menus into actual gameplay for a
// headless verification run. Gated by g_autoskip at the call site, where it
// takes precedence over the keyboard. NEVER pulse Circle (back/cancel) together
// with Cross (confirm): pressing both each cycle confirms then immediately backs
// out, so menus never advance (this kept the headless run stuck on the title).
// Sequence: Options first (title "PRESS OPTIONS" -> main menu), then Cross to
// confirm "New Run"/save-slot/character-select, with the occasional Down to move
// the menu cursor. Buttons pulse with gaps so menus see clean press edges.
uint32_t autoSkipButtons() {
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
  uint32_t phase = g_readSeq % 24;
  if (phase < 3) return kOptions;
  if (phase >= 8 && phase < 11) return kCross;
  if (phase >= 16 && phase < 19) return kCross;
  return 0;
}
static const bool g_autoskip = std::getenv("DELTA_PAD_AUTOSKIP") != nullptr;

// Adapter from the gfx pad (maps the SDL window keyboard; the Android app maps
// the on-screen touch gamepad). On by default for interactive play; set
// DELTA_PAD_KEYBOARD=0 to disable. DELTA_PAD_AUTOSKIP overrides it.
#if defined(DELTA_ANDROID_APP)
static const bool g_keyboard = true;
#else
static const bool g_keyboard = [] {
  const char *e = std::getenv("DELTA_PAD_KEYBOARD");
  return !e || std::strcmp(e, "0") != 0;
}();
#endif

void fillPadState(PadData *d) {
  if (!d) return;
  std::memset(d, 0, sizeof(*d));
  uint32_t buttons = 0;
  uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
  gfx::PadKeys k;
  if (g_autoskip) {
    buttons = autoSkipButtons();
  } else if (g_keyboard && gfx::pollKeyboardPad(k)) {
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
  d->buttons = buttons;
  d->leftStick = {lx, ly};
  d->rightStick = {rx, ry};
  d->analogButtons = {static_cast<uint8_t>((buttons & kL2) ? 255 : 0),
                      static_cast<uint8_t>((buttons & kR2) ? 255 : 0)};
  d->orientation = {0, 0, 0, 1};
  d->connected = true;
  d->connectedCount = 1;
  d->timestamp = ++g_readSeq;
  if (g_autoskip && (g_readSeq % 600 == 1))
    std::fprintf(stderr, "[pad] readSeq=%llu buttons=%#x\n",
                 (unsigned long long)g_readSeq, buttons);
}

}  // namespace

int scePadClose() {
  LOG_UNIMPLEMENTED;
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
  LOG_UNIMPLEMENTED;
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
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadResetOrientation() {
  LOG_UNIMPLEMENTED;
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
  LOG_UNIMPLEMENTED;
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
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetTiltCorrectionState() {
  LOG_UNIMPLEMENTED;
  return 0;
}

int scePadSetVibration() {
  LOG_UNIMPLEMENTED;
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
