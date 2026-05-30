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

struct dynlib_info_ex {
  size_t size;
  char name[256];
  int32_t handle;
  uint16_t tls_index;
  uint16_t pad0;
  uintptr_t tls_init_addr;
  uint32_t tls_init_size;
  uint32_t tls_size;
  uint32_t tls_offset;
  uint32_t tls_align;
  uintptr_t init_proc_addr;
  uintptr_t fini_proc_addr;
  uint64_t reserved1;
  uint64_t reserved2;
  uintptr_t eh_frame_hdr_addr;
  uintptr_t eh_frame_addr;
  uint32_t eh_frame_hdr_size;
  uint32_t eh_frame_size;
  segment_info segs[4];
  uint32_t seg_count;
  uint32_t ref_count;
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
}