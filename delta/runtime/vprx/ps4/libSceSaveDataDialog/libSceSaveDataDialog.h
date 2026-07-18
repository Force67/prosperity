#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveDataDialog. See libSceSaveDataDialog.cpp.
 */

#include "../../vprx.h"

#include <cstdint>

extern "C" {

int PS4ABI sceSaveDataDialogInitialize();
int PS4ABI sceSaveDataDialogTerminate();
int PS4ABI sceSaveDataDialogOpen(const void *param);
int PS4ABI sceSaveDataDialogClose();
int PS4ABI sceSaveDataDialogGetStatus();
int PS4ABI sceSaveDataDialogUpdateStatus();
int PS4ABI sceSaveDataDialogGetResult(void *result);
int PS4ABI sceSaveDataDialogIsReadyToDisplay();
int PS4ABI sceSaveDataDialogProgressBarSetValue(uint32_t target, uint32_t rate);
int PS4ABI sceSaveDataDialogProgressBarInc(uint32_t target, uint32_t delta);

}  // extern "C"
