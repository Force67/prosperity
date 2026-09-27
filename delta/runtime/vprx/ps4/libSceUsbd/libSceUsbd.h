#pragma once
#include "guest_abi.h"
#include "../../vprx.h"
#include "base/arch.h"

// HLE of libSceUsbd (the PS4's libusb-1.0 port). Presents a single virtual
// DualShock4 so the title's USB controller enumeration succeeds and it leaves
// the "no controller" pause into active gameplay. HID input reports are built
// from the keyboard adapter (gfx::pollKeyboardPad) / neutral state.
int PS4ABI sceUsbdInit();
void PS4ABI sceUsbdExit();
int PS4ABI sceUsbdGetDeviceList(void ***list);
void PS4ABI sceUsbdFreeDeviceList(void **list, int unrefDevices);
int PS4ABI sceUsbdGetDeviceDescriptor(void *dev, void *desc);
int PS4ABI sceUsbdGetActiveConfigDescriptor(void *dev, void **config);
int PS4ABI sceUsbdGetConfigDescriptor(void *dev, u8 idx, void **config);
int PS4ABI sceUsbdOpen(void *dev, void **handle);
void PS4ABI sceUsbdClose(void *handle);
void *PS4ABI sceUsbdOpenDeviceWithVidPid(void *ctx, u16 vid, u16 pid);
int PS4ABI sceUsbdGetConfiguration(void *handle, int *config);
int PS4ABI sceUsbdSetConfiguration(void *handle, int config);
int PS4ABI sceUsbdClaimInterface(void *handle, int iface);
int PS4ABI sceUsbdReleaseInterface(void *handle, int iface);
int PS4ABI sceUsbdControlTransfer(void *handle, u8 reqType, u8 req,
                                  u16 value, u16 index, void *data,
                                  u16 length, u32 timeout);
int PS4ABI sceUsbdInterruptTransfer(void *handle, u8 endpoint, void *data,
                                    int length, int *transferred, u32 timeout);
int PS4ABI sceUsbdBulkTransfer(void *handle, u8 endpoint, void *data,
                               int length, int *transferred, u32 timeout);
void *PS4ABI sceUsbdGetDevice(void *handle);
void *PS4ABI sceUsbdRefDevice(void *dev);
void PS4ABI sceUsbdUnrefDevice(void *dev);
void *PS4ABI sceUsbdAllocTransfer(int isoPackets);
int PS4ABI sceUsbdSubmitTransfer(void *transfer);
void PS4ABI sceUsbdFreeTransfer(void *transfer);
int PS4ABI sceUsbdCancelTransfer(void *transfer);
void PS4ABI sceUsbdFillControlTransfer(void *transfer, void *handle, void *buf,
                                       void *cb, void *user, u32 timeout);
void PS4ABI sceUsbdFillInterruptTransfer(void *transfer, void *handle,
                                         u8 endpoint, void *buf, int length,
                                         void *cb, void *user, u32 timeout);
void PS4ABI sceUsbdFillBulkTransfer(void *transfer, void *handle, u8 endpoint,
                                    void *buf, int length, void *cb, void *user,
                                    u32 timeout);
int PS4ABI sceUsbdHandleEvents();
int PS4ABI sceUsbdHandleEventsTimeout(void *tv);
int PS4ABI sceUsbdGetStringDescriptorAscii(void *handle, u8 idx, void *data,
                                           int length);
int PS4ABI sceUsbdSetInterfaceAltSetting(void *handle, int iface, int alt);
int PS4ABI sceUsbdResetDevice(void *handle);
int PS4ABI sceUsbdKernelDriverActive(void *handle, int iface);
int PS4ABI sceUsbdDetachKernelDriver(void *handle, int iface);
int PS4ABI sceUsbdGetBusNumber(void *dev);
int PS4ABI sceUsbdGetDeviceAddress(void *dev);
int PS4ABI sceUsbdGetDeviceSpeed(void *dev);
int PS4ABI sceUsbdCheckConnected(void *handle);
int PS4ABI sceUsbdEventHandlingOk(void *ctx);
