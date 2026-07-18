#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceMsgDialog. See libSceMsgDialog.cpp.
 */

#include "../../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceMsgDialogInitialize();
int PS4ABI sceMsgDialogTerminate();
int PS4ABI sceMsgDialogOpen(const void *param);
int PS4ABI sceMsgDialogClose();
int PS4ABI sceMsgDialogUpdateStatus();
int PS4ABI sceMsgDialogGetStatus();
int PS4ABI sceMsgDialogGetResult(void *result);
int PS4ABI sceMsgDialogProgressBarSetValue(uint32_t target, uint32_t rate);
int PS4ABI sceMsgDialogProgressBarInc(uint32_t target, uint32_t delta);
int PS4ABI sceMsgDialogProgressBarSetMsg(uint32_t target, const char *msg);

}  // extern "C"
