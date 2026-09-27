#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {
struct proc_param {  // NOLINT(readability-identifier-naming): guest ABI name
  u64 length;
  u32 magic;
  u32 unk;
  u32 kvers;
};

struct segment_info {  // NOLINT(readability-identifier-naming): guest ABI name
  uintptr_t addr;
  u32 size;
  u32 flags;
};

// SceKernelModuleInfoEx (0x1A8 / 424 bytes). The guest passes size=424 in the
// first field and the kernel fills the rest. Segment flags: R=1, W=2, X=4, so
// text is 5 (R|X) and data is 3 (R|W). The kernel fills exactly two segments
// and reports seg_count=2.
struct dynlib_info_ex {      // NOLINT(readability-identifier-naming): guest ABI
                             // name
  size_t size;               // +0x000
  char name[256];            // +0x008  module basename
  i32 handle;                // +0x108
  u16 tls_index;             // +0x10C
  u16 pad0;                  // +0x10E
  uintptr_t tls_init_addr;   // +0x110
  u32 tls_init_size;         // +0x118
  u32 tls_size;              // +0x11C
  u32 tls_offset;            // +0x120
  u32 tls_align;             // +0x124
  uintptr_t init_proc_addr;  // +0x128
  uintptr_t fini_proc_addr;  // +0x130
  u64 reserved1;             // +0x138
  u64 reserved2;             // +0x140
  uintptr_t eh_frame_hdr_addr;  // +0x148
  uintptr_t eh_frame_addr;      // +0x150
  u32 eh_frame_hdr_size;        // +0x158
  u32 eh_frame_size;            // +0x15C
  segment_info segs[4];         // +0x160
  u32 seg_count;                // +0x1A0
  u32 ref_count;                // +0x1A4
};

struct dynlib_info {  // NOLINT(readability-identifier-naming): guest ABI name
  size_t size;
  char name[256];
  segment_info segs[4];
  u32 seg_count;
  u8 fingerprint[20];
};

static_assert(sizeof(dynlib_info_ex) == 424);
static_assert(sizeof(dynlib_info) == 352);

int PS4ABI sys_dynlib_dlopen(const char*);
int PS4ABI sys_dynlib_get_info(u32 handle, dynlib_info*);
int PS4ABI sys_dynlib_get_info_ex(u32 handle,
                                  i32 ukn /*always 1*/,
                                  dynlib_info_ex* dyn_info);
int PS4ABI sys_dynlib_get_proc_param(void** data, size_t* size);
int PS4ABI sys_dynlib_get_list(u32* handles, size_t max_count, size_t* count);
int PS4ABI sys_dynlib_dlsym(u32 handle, const char* cname, void** sym);
int PS4ABI sys_dynlib_get_obj_member(u32 handle, u8 index, void** value);
int PS4ABI sys_dynlib_process_needed_and_relocate();
int PS4ABI sys_dynlib_load_prx(const char* path,
                               u64 arg2,
                               int* p_handle,
                               u64 arg4,
                               const void* opt,
                               i64* p_res);
int PS4ABI sys_dynlib_unload_prx(u32 handle);

// HLE __tls_get_addr; libkernel's export is patched to this. (See impl.)
struct tls_index;  // NOLINT(readability-identifier-naming): guest ABI name
void* PS4ABI GuestTlsGetAddr(tls_index* ti);

int PS4ABI sys_dl_get_list();
int PS4ABI sys_dl_get_info();
int PS4ABI sys_dl_notify_event();
int PS4ABI sys_dynlib_dlclose();
int PS4ABI sys_dynlib_prepare_dlclose();
int PS4ABI sys_dl_get_metadata();
int PS4ABI sys_dynlib_get_info_for_libdbg();
int PS4ABI sys_dynlib_get_list_for_libdbg();
int PS4ABI sys_dynlib_get_list2();
int PS4ABI sys_dynlib_get_info2();

int PS4ABI sys_dynlib_do_copy_relocations();

}  // namespace kern