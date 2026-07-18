#include "libSceUsbd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gfx/gfx.h"

// A single virtual DualShock4 presented through the libusb-style sceUsbd API.
// Opaque tokens are handed to the game and validated on the way back.
namespace {
int g_dev = 0xD54DE0;            // device token
int g_handle = 0xD54AB1E;        // open-handle token
void *g_devList[2] = {&g_dev, nullptr};  // NULL-terminated device list

constexpr uint16_t kVid = 0x054C;  // Sony
constexpr uint16_t kPid = 0x05C4;  // DualShock4 (CUH-ZCT1)

// USB device descriptor (matches libusb_device_descriptor layout).
const uint8_t kDeviceDesc[18] = {
    0x12, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x40,
    0x4C, 0x05, 0xC4, 0x05, 0x00, 0x01, 0x01, 0x02, 0x00, 0x01};

void traceOnce(const char *fn) {
  static const char *seen[64] = {};
  for (auto *&s : seen) {
    if (!s) { s = fn; std::fprintf(stderr, "[usbd] %s\n", fn); return; }
    if (s == fn) return;
  }
}

// Build a 64-byte DS4 USB HID input report from the keyboard adapter (or
// neutral). Layout: [0]=reportId, [1..4]=sticks, [5]=dpad+face, [6]=shoulder/
// system, [7]=counter+ps/touch, [8..9]=triggers.
void buildReport(uint8_t *r, int len) {
  if (!r || len < 10) { if (r && len > 0) std::memset(r, 0, len); return; }
  std::memset(r, 0, len);
  uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
  gfx::PadKeys k;
  bool sq = false, cr = false, ci = false, tr = false, l1 = false, r1 = false,
       l2 = false, r2 = false, opt = false, tpad = false;
  int dpad = 8;  // 8 = released
  if (gfx::pollKeyboardPad(k)) {
    lx = k.lx; ly = k.ly; rx = k.rx; ry = k.ry;
    sq = k.square; cr = k.cross; ci = k.circle; tr = k.triangle;
    l1 = k.l1; r1 = k.r1; l2 = k.l2; r2 = k.r2; opt = k.options; tpad = k.touchpad;
    if (k.up && k.right) dpad = 1; else if (k.right && k.down) dpad = 3;
    else if (k.down && k.left) dpad = 5; else if (k.left && k.up) dpad = 7;
    else if (k.up) dpad = 0; else if (k.right) dpad = 2; else if (k.down) dpad = 4;
    else if (k.left) dpad = 6; else dpad = 8;
  }
  r[0] = 0x01;
  r[1] = lx; r[2] = ly; r[3] = rx; r[4] = ry;
  r[5] = (uint8_t)((dpad & 0x0F) | (sq ? 0x10 : 0) | (cr ? 0x20 : 0) |
                   (ci ? 0x40 : 0) | (tr ? 0x80 : 0));
  r[6] = (uint8_t)((l1 ? 0x01 : 0) | (r1 ? 0x02 : 0) | (l2 ? 0x04 : 0) |
                   (r2 ? 0x08 : 0) | (opt ? 0x20 : 0));
  static uint8_t counter = 0;
  r[7] = (uint8_t)((tpad ? 0x02 : 0) | ((counter++ & 0x3F) << 2));
  r[8] = l2 ? 255 : 0;
  r[9] = r2 ? 255 : 0;
}

// Async transfer record (libusb-style). We track IN-endpoint transfers so
// HandleEvents can complete them with a fresh report (+ fire the callback).
struct Transfer {
  uint8_t endpoint = 0;
  uint8_t *buf = nullptr;
  int length = 0;
  void (*cb)(void *) = nullptr;
  void *self = nullptr;
  bool submitted = false;
  bool used = false;
};
constexpr int kMaxXfer = 64;
Transfer g_xfers[kMaxXfer];
}  // namespace

int PS4ABI sceUsbdInit() { traceOnce("Init"); return 0; }
void PS4ABI sceUsbdExit() {}

int PS4ABI sceUsbdGetDeviceList(void ***list) {
  traceOnce("GetDeviceList");
  if (list) *list = g_devList;  // NULL-terminated [&g_dev, nullptr]
  return 1;  // one device
}
void PS4ABI sceUsbdFreeDeviceList(void **list, int unref) {
  static int c = 0; if (c < 3) { c++;
    std::fprintf(stderr, "[usbd] FreeDeviceList list=%p unref=%d (mine=%p)\n",
                 (void *)list, unref, (void *)g_devList); }
}

int PS4ABI sceUsbdGetDeviceDescriptor(void *dev, void *desc) {
  traceOnce("GetDeviceDescriptor");
  if (desc) std::memcpy(desc, kDeviceDesc, sizeof(kDeviceDesc));
  return 0;
}

int PS4ABI sceUsbdGetActiveConfigDescriptor(void *dev, void **config) {
  traceOnce("GetActiveConfigDescriptor");
  return -99;  // observe whether the game needs the parsed config
}
int PS4ABI sceUsbdGetConfigDescriptor(void *dev, uint8_t idx, void **config) {
  traceOnce("GetConfigDescriptor");
  return -99;
}

int PS4ABI sceUsbdOpen(void *dev, void **handle) {
  traceOnce("Open");
  if (handle) *handle = &g_handle;
  return 0;
}
void PS4ABI sceUsbdClose(void *handle) {}
void *PS4ABI sceUsbdOpenDeviceWithVidPid(void *ctx, uint16_t vid, uint16_t pid) {
  traceOnce("OpenDeviceWithVidPid");
  return (vid == kVid && pid == kPid) ? &g_handle : nullptr;
}

int PS4ABI sceUsbdGetConfiguration(void *handle, int *config) {
  if (config) *config = 1;
  return 0;
}
int PS4ABI sceUsbdSetConfiguration(void *handle, int config) { return 0; }
int PS4ABI sceUsbdClaimInterface(void *handle, int iface) {
  traceOnce("ClaimInterface");
  return 0;
}
int PS4ABI sceUsbdReleaseInterface(void *handle, int iface) { return 0; }

int PS4ABI sceUsbdControlTransfer(void *handle, uint8_t reqType, uint8_t req,
                                  uint16_t value, uint16_t index, void *data,
                                  uint16_t length, uint32_t timeout) {
  traceOnce("ControlTransfer");
  if ((reqType & 0x80) && data && length) {  // IN: descriptor request
    std::memset(data, 0, length);
    if ((value >> 8) == 0x01)
      std::memcpy(data, kDeviceDesc,
                  length < sizeof(kDeviceDesc) ? length : sizeof(kDeviceDesc));
    return length;
  }
  return 0;
}

int PS4ABI sceUsbdInterruptTransfer(void *handle, uint8_t endpoint, void *data,
                                    int length, int *transferred, uint32_t timeout) {
  traceOnce("InterruptTransfer");
  if ((endpoint & 0x80) && data)
    buildReport(static_cast<uint8_t *>(data), length);
  if (transferred) *transferred = length;
  return 0;
}
int PS4ABI sceUsbdBulkTransfer(void *handle, uint8_t endpoint, void *data,
                               int length, int *transferred, uint32_t timeout) {
  if (transferred) *transferred = 0;
  return 0;
}

void *PS4ABI sceUsbdGetDevice(void *handle) { return &g_dev; }
void *PS4ABI sceUsbdRefDevice(void *dev) { return dev; }
void PS4ABI sceUsbdUnrefDevice(void *dev) {}

void *PS4ABI sceUsbdAllocTransfer(int isoPackets) {
  traceOnce("AllocTransfer");
  for (auto &x : g_xfers)
    if (!x.used) { x = Transfer{}; x.used = true; x.self = &x; return &x; }
  return nullptr;
}
void PS4ABI sceUsbdFreeTransfer(void *transfer) {
  for (auto &x : g_xfers)
    if (transfer && transfer == x.self) { x = Transfer{}; return; }
}
void PS4ABI sceUsbdFillInterruptTransfer(void *transfer, void *handle,
                                         uint8_t endpoint, void *buf, int length,
                                         void *cb, void *user, uint32_t timeout) {
  traceOnce("FillInterruptTransfer");
  for (auto &x : g_xfers)
    if (transfer == x.self) {
      x.endpoint = endpoint; x.buf = static_cast<uint8_t *>(buf);
      x.length = length; x.cb = reinterpret_cast<void (*)(void *)>(cb);
      return;
    }
}
void PS4ABI sceUsbdFillControlTransfer(void *transfer, void *handle, void *buf,
                                       void *cb, void *user, uint32_t timeout) {}
void PS4ABI sceUsbdFillBulkTransfer(void *transfer, void *handle, uint8_t endpoint,
                                    void *buf, int length, void *cb, void *user,
                                    uint32_t timeout) {}
int PS4ABI sceUsbdSubmitTransfer(void *transfer) {
  traceOnce("SubmitTransfer");
  for (auto &x : g_xfers)
    if (transfer == x.self) { x.submitted = true; return 0; }
  return 0;
}
int PS4ABI sceUsbdCancelTransfer(void *transfer) {
  for (auto &x : g_xfers)
    if (transfer == x.self) { x.submitted = false; return 0; }
  return 0;
}

int PS4ABI sceUsbdHandleEvents() {
  for (auto &x : g_xfers)
    if (x.submitted && x.buf && (x.endpoint & 0x80)) {
      buildReport(x.buf, x.length);
      x.submitted = false;
      if (x.cb) x.cb(x.self);
    }
  return 0;
}
int PS4ABI sceUsbdHandleEventsTimeout(void *tv) { return sceUsbdHandleEvents(); }

int PS4ABI sceUsbdGetStringDescriptorAscii(void *handle, uint8_t idx, void *data,
                                           int length) {
  if (data && length) std::strncpy(static_cast<char *>(data), "Wireless Controller", length - 1);
  return data ? (int)std::strlen(static_cast<char *>(data)) : 0;
}
int PS4ABI sceUsbdSetInterfaceAltSetting(void *handle, int iface, int alt) { return 0; }
int PS4ABI sceUsbdResetDevice(void *handle) { return 0; }
int PS4ABI sceUsbdKernelDriverActive(void *handle, int iface) { return 0; }
int PS4ABI sceUsbdDetachKernelDriver(void *handle, int iface) { return 0; }
int PS4ABI sceUsbdGetBusNumber(void *dev) { return 1; }
int PS4ABI sceUsbdGetDeviceAddress(void *dev) { return 1; }
int PS4ABI sceUsbdGetDeviceSpeed(void *dev) { return 3; }
int PS4ABI sceUsbdCheckConnected(void *handle) { return 0; }
int PS4ABI sceUsbdEventHandlingOk(void *ctx) { return 1; }
