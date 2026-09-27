#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceMsgDialog. See lib_sce_msg_dialog.cc.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "runtime/vprx/vprx.h"

extern "C" {

int PS4ABI sceMsgDialogInitialize();
int PS4ABI sceMsgDialogTerminate();
int PS4ABI sceMsgDialogOpen(const void* param);
int PS4ABI sceMsgDialogClose();
int PS4ABI sceMsgDialogUpdateStatus();
int PS4ABI sceMsgDialogGetStatus();
int PS4ABI sceMsgDialogGetResult(void* result);
int PS4ABI sceMsgDialogProgressBarSetValue(u32 target, u32 rate);
int PS4ABI sceMsgDialogProgressBarInc(u32 target, u32 delta);
int PS4ABI sceMsgDialogProgressBarSetMsg(u32 target, const char* msg);

}  // extern "C"
