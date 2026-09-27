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

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"

#include <cstdint>

namespace host {}  // namespace host

extern "C" {

// --- core ---
int PS4ABI sceVideoOutOpen(int user_id,
                           int bus_type,
                           int index,
                           const void* param);
int PS4ABI sceVideoOutClose(int handle);
int PS4ABI sceVideoOutGetResolutionStatus(int handle, void* status);
int PS4ABI sceVideoOutSetBufferAttribute(void* attribute,
                                         u32 pixel_format,
                                         u32 tiling_mode,
                                         u32 aspect_ratio,
                                         u32 width,
                                         u32 height,
                                         u32 pitch_in_pixel);
int PS4ABI sceVideoOutRegisterBuffers(int handle,
                                      int start_index,
                                      void* const* addresses,
                                      int buffer_num,
                                      const void* attribute);
int PS4ABI sceVideoOutUnregisterBuffers(int handle, int attribute_index);
int PS4ABI sceVideoOutSetFlipRate(int handle, int rate);

// --- events ---
int PS4ABI sceVideoOutAddFlipEvent(int eq_handle, int handle, void* udata);
int PS4ABI sceVideoOutDeleteFlipEvent(int eq_handle, int handle);
int PS4ABI sceVideoOutAddVblankEvent(int eq_handle, int handle, void* udata);
int PS4ABI sceVideoOutGetEventCount(const void* event);
int PS4ABI sceVideoOutGetEventId(const void* event);
int PS4ABI sceVideoOutGetEventData(const void* event, i64* data);

// --- flip / vblank ---
int PS4ABI sceVideoOutSubmitFlip(int handle,
                                 int buffer_index,
                                 int flip_mode,
                                 i64 flip_arg);
// EOP-label variant of SubmitFlip (NID j8xl+92A0q4). The real libSceGnmDriver
// calls this on its LLE flip path; same as SubmitFlip plus a GPU
// completion-label pointer we don't need (we present synchronously).
int PS4ABI sceVideoOutSubmitFlipEop(int handle,
                                    int buffer_index,
                                    int flip_mode,
                                    i64 flip_arg,
                                    void* eop_label);
int PS4ABI sceVideoOutGetFlipStatus(int handle, void* status);
int PS4ABI sceVideoOutIsFlipPending(int handle);
int PS4ABI sceVideoOutGetVblankStatus(int handle, void* status);
int PS4ABI sceVideoOutWaitVblank(int handle);

// --- misc ---
int PS4ABI sceVideoOutGetBufferLabelAddress(int handle, uintptr_t* label);
int PS4ABI sceVideoOutSetWindowModeMargins(int handle, int top, int bottom);
int PS4ABI sceVideoOutColorSettingsSetGamma_(void* settings, float gamma);
int PS4ABI sceVideoOutModeSetAny_(int handle, void* arg);

}  // extern "C"
