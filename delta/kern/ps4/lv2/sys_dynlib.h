#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

namespace krnl {
struct proc_param {
  uint64_t length;
  uint32_t magic;
  uint32_t unk;
  uint32_t kvers;
};

struct segment_info {
  uintptr_t addr;
  uint32_t size;
  uint32_t flags;
};

// SceKernelModuleInfoEx (0x1A8 / 424 bytes). The guest passes size=424 in the
// first field and the kernel fills the rest. Segment flags: R=1, W=2, X=4, so
// text is 5 (R|X) and data is 3 (R|W). The kernel fills exactly two segments
// and reports seg_count=2.
struct dynlib_info_ex {
  size_t size;            // +0x000
  char name[256];         // +0x008  module basename
  int32_t handle;         // +0x108
  uint16_t tls_index;     // +0x10C
  uint16_t pad0;          // +0x10E
  uintptr_t tls_init_addr;// +0x110
  uint32_t tls_init_size; // +0x118
  uint32_t tls_size;      // +0x11C
  uint32_t tls_offset;    // +0x120
  uint32_t tls_align;     // +0x124
  uintptr_t init_proc_addr;// +0x128
  uintptr_t fini_proc_addr;// +0x130
  uint64_t reserved1;     // +0x138
  uint64_t reserved2;     // +0x140
  uintptr_t eh_frame_hdr_addr; // +0x148
  uintptr_t eh_frame_addr;     // +0x150
  uint32_t eh_frame_hdr_size;  // +0x158
  uint32_t eh_frame_size;      // +0x15C
  segment_info segs[4];        // +0x160
  uint32_t seg_count;          // +0x1A0
  uint32_t ref_count;          // +0x1A4
};

struct dynlib_info {
  size_t size;
  char name[256];
  segment_info segs[4];
  uint32_t seg_count;
  uint8_t fingerprint[20];
};

static_assert(sizeof(dynlib_info_ex) == 424);
static_assert(sizeof(dynlib_info) == 352);

int PS4ABI sys_dynlib_dlopen(const char *);
int PS4ABI sys_dynlib_get_info(uint32_t handle, dynlib_info *);
int PS4ABI sys_dynlib_get_info_ex(uint32_t handle, int32_t ukn /*always 1*/,
                                  dynlib_info_ex *dyn_info);
int PS4ABI sys_dynlib_get_proc_param(void **data, size_t *size);
int PS4ABI sys_dynlib_get_list(uint32_t *handles, size_t maxCount,
                               size_t *count);
int PS4ABI sys_dynlib_dlsym(uint32_t handle, const char *cname, void **sym);
int PS4ABI sys_dynlib_get_obj_member(uint32_t handle, uint8_t index,
                                     void **value);
int PS4ABI sys_dynlib_process_needed_and_relocate();
int PS4ABI sys_dynlib_load_prx(const char *path, uint64_t arg2, int *pHandle,
                               uint64_t arg4, const void *opt, int64_t *pRes);
int PS4ABI sys_dynlib_unload_prx(uint32_t handle);

// HLE __tls_get_addr; libkernel's export is patched to this. (See impl.)
struct tls_index;
void *PS4ABI guest_tls_get_addr(tls_index *ti);
}