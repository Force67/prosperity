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

#include "guest_abi.h"
#include "../../vprx.h"
#include "base/arch.h"

#include <cstdint>

namespace host {}  // fwd: present/init live in delta/host/window.h

extern "C" {

// --- core ---
int PS4ABI sceVideoOutOpen(int userId, int busType, int index, const void *param);
int PS4ABI sceVideoOutClose(int handle);
int PS4ABI sceVideoOutGetResolutionStatus(int handle, void *status);
int PS4ABI sceVideoOutSetBufferAttribute(void *attribute, u32 pixelFormat,
                                         u32 tilingMode, u32 aspectRatio,
                                         u32 width, u32 height,
                                         u32 pitchInPixel);
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
int PS4ABI sceVideoOutGetEventData(const void *event, i64 *data);

// --- flip / vblank ---
int PS4ABI sceVideoOutSubmitFlip(int handle, int bufferIndex, int flipMode,
                                i64 flipArg);
// EOP-label variant of SubmitFlip (NID j8xl+92A0q4). The real libSceGnmDriver
// calls this on its LLE flip path; same as SubmitFlip plus a GPU completion-label
// pointer we don't need (we present synchronously).
int PS4ABI sceVideoOutSubmitFlipEop(int handle, int bufferIndex, int flipMode,
                                    i64 flipArg, void *eopLabel);
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
