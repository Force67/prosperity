
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <logger/logger.h>

#include "../module.h"
#include "../proc.h"

#include "error_table.h"
#include "sys_dynlib.h"
#include <runtime/vprx/vprx.h>

namespace krnl {
int PS4ABI sys_dynlib_dlopen(const char *) {
  /*TODO: implement, however note that this function is only
  present in devkits*/
  return SysError::eNOSYS;
}

int PS4ABI sys_dynlib_get_info(uint32_t handle, dynlib_info *dyn_info) {
  if (dyn_info->size != sizeof(*dyn_info)) {
    __debugbreak();
    return SysError::eINVAL;
  }

  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return SysError::eSRCH;

  auto &info = mod->getInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info));
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  auto &text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.textSeg.addr);
  text.size = info.textSeg.size;
  text.flags = 1 | 4;

  auto &data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.dataSeg.addr);
  data.size = info.dataSeg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;

  std::memcpy(dyn_info->fingerprint, info.fingerprint, 20);
  return 0;
}

int PS4ABI sys_dynlib_get_info_ex(uint32_t handle, int32_t ukn /*always 1*/,
                                  dynlib_info_ex *dyn_info) {
  if (dyn_info->size != sizeof(*dyn_info)) {
    __debugbreak();
    return SysError::eINVAL;
  }

  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return SysError::eSRCH;

  auto &info = mod->getInfo();
  std::memset(dyn_info, 0, sizeof(dynlib_info_ex));

  dyn_info->size = sizeof(dynlib_info_ex);
  dyn_info->handle = info.handle;
  std::strncpy(dyn_info->name, info.name.c_str(), 256);

  dyn_info->tls_index = info.tlsSlot;
  dyn_info->tls_align = info.tlsalign;
  dyn_info->tls_init_size = info.tlsSizeFile;
  dyn_info->tls_size = info.tlsSizeMem;
  dyn_info->tls_init_addr = reinterpret_cast<uintptr_t>(info.tlsAddr);

  dyn_info->init_proc_addr = reinterpret_cast<uintptr_t>(info.initAddr);
  dyn_info->fini_proc_addr = reinterpret_cast<uintptr_t>(info.finiAddr);

  dyn_info->eh_frame_addr = reinterpret_cast<uintptr_t>(info.ehFrameAddr);
  dyn_info->eh_frame_hdr_addr =
      reinterpret_cast<uintptr_t>(info.ehFrameheaderAddr);
  dyn_info->eh_frame_size = info.ehFrameSize;
  dyn_info->eh_frame_hdr_size = info.ehFrameheaderSize;

  auto &text = dyn_info->segs[0];
  text.addr = reinterpret_cast<uintptr_t>(info.textSeg.addr);
  text.size = info.textSeg.size;
  text.flags = 1 | 4;

  auto &data = dyn_info->segs[1];
  data.addr = reinterpret_cast<uintptr_t>(info.dataSeg.addr);
  data.size = info.dataSeg.size;
  data.flags = 1 | 2;

  dyn_info->seg_count = 2;
  dyn_info->ref_count = 1;
  return 0;
}

int PS4ABI sys_dynlib_dlsym(uint32_t handle, const char *symName, void **sym) {
  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -1;

  std::printf("DLSYM %s!%s\n", mod->getInfo().name.c_str(), symName);

  char nameenc[12]{};
  runtime::encode_nid(symName, reinterpret_cast<uint8_t *>(&nameenc));

  auto &modName = mod->getInfo().name;
  base::String longName(nameenc);
  longName += "#";
  longName += modName;
  longName += "#";
  longName += modName;

  uintptr_t addrOut = 0;
  if (!mod->resolveObfSymbol(longName.c_str(), addrOut)) {
    *sym = nullptr;
    return -1;
  }

  *sym = reinterpret_cast<void *>(addrOut);

  return 0;
}

int PS4ABI sys_dynlib_get_obj_member(uint32_t handle, uint8_t index,
                                     void **value) {
  if (index != 1)
    return SysError::eINVAL;

  auto mod = proc::getActive()->getModule(handle);
  if (!mod)
    return -1;

  *value = mod->getInfo().initAddr;
  return 0;
}

int PS4ABI sys_dynlib_get_proc_param(void **data, size_t *size) {
  auto mod = proc::getActive()->getModuleList()[0];
  if (mod) {
    auto &info = mod->getInfo();

    *data = reinterpret_cast<void *>(info.procParam);
    *size = info.procParamSize;
    return 0;
  }

  *data = nullptr;
  *size = 0;

  return 0;
}

int PS4ABI sys_dynlib_get_list(uint32_t *handles, size_t maxCount,
                               size_t *count) {
  auto &list = proc::getActive()->getModuleList();

  int listCount = 0;
  for (auto &mod : list) {
    *(handles++) = mod->getInfo().handle;
    listCount++;
  }

  *count = listCount;
  return 0;
}

int PS4ABI sys_dynlib_process_needed_and_relocate() {
  auto &list = proc::getActive()->getModuleList();
  for (auto &mod : list) {
    LOG_ASSERT(mod);

    if (!mod->resolveImports() || !mod->applyRelocations()) {
      LOG_ERROR("failed to apply relocations for module {}",
                mod->getInfo().name.c_str());
      return -1;
    }
  }

  return 0;
}
} // namespace krnl