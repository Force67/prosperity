#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceVideoOut. The real Sony module loads fine but its internal device
 * handler table lives in .bss and is populated by a kernel/display-service
 * registration we don't emulate, so its sceVideoOutOpen returns 0x802900ff and
 * the whole display-init cascade fails. We override the library here: open a
 * real SDL3/Vulkan window, track registered scanout buffers, and on SubmitFlip
 * present the scanout to the window and deliver the flip event.
 */

#include "../vprx.h"

#include <cstdint>

namespace gfx {}  // fwd: present/init live in delta/gfx/gfx.h

extern "C" {

// --- core ---
int PS4ABI sceVideoOutOpen(int userId, int busType, int index, const void *param);
int PS4ABI sceVideoOutClose(int handle);
int PS4ABI sceVideoOutGetResolutionStatus(int handle, void *status);
int PS4ABI sceVideoOutSetBufferAttribute(void *attribute, uint32_t pixelFormat,
                                         uint32_t tilingMode, uint32_t aspectRatio,
                                         uint32_t width, uint32_t height,
                                         uint32_t pitchInPixel);
int PS4ABI sceVideoOutRegisterBuffers(int handle, int startIndex,
                                     void *const *addresses, int bufferNum,
                                     const void *attribute);
int PS4ABI sceVideoOutUnregisterBuffers(int handle, int attributeIndex);
int PS4ABI sceVideoOutSetFlipRate(int handle, int rate);

// --- events ---
int PS4ABI sceVideoOutAddFlipEvent(int eqHandle, int handle, void *udata);
int PS4ABI sceVideoOutDeleteFlipEvent(int eqHandle, int handle);
int PS4ABI sceVideoOutAddVblankEvent(int eqHandle, int handle, void *udata);
int PS4ABI sceVideoOutGetEventCount(const void *event);
int PS4ABI sceVideoOutGetEventId(const void *event);
int PS4ABI sceVideoOutGetEventData(const void *event, int64_t *data);

// --- flip / vblank ---
int PS4ABI sceVideoOutSubmitFlip(int handle, int bufferIndex, int flipMode,
                                int64_t flipArg);
int PS4ABI sceVideoOutGetFlipStatus(int handle, void *status);
int PS4ABI sceVideoOutIsFlipPending(int handle);
int PS4ABI sceVideoOutGetVblankStatus(int handle, void *status);
int PS4ABI sceVideoOutWaitVblank(int handle);

// --- misc ---
int PS4ABI sceVideoOutGetBufferLabelAddress(int handle, uintptr_t *label);
int PS4ABI sceVideoOutSetWindowModeMargins(int handle, int top, int bottom);
int PS4ABI sceVideoOutColorSettingsSetGamma_(void *settings, float gamma);
int PS4ABI sceVideoOutModeSetAny_(int handle, void *arg);

}  // extern "C"
