/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveDataDialog. On real hardware the save-data dialog is drawn by
 * the system UI process (SceShellUI); the game calls sceSaveDataDialogOpen and
 * then polls sceSaveDataDialogUpdateStatus until it reports FINISHED (the user
 * dismissed it / a progress-bar op completed). The LLE libSceSaveDataDialog.sprx
 * forwards over IPMI to that dialog service, which we don't host, so its status
 * never advances past RUNNING and any title that gates progression on the dialog
 * finishing (P.T.'s world-load "GameSave"/"UpdateSaveDialog" flow) hangs forever.
 *
 * We host no interactive UI and there is no user to click, so the only correct
 * emulated behaviour is to complete the dialog immediately with a default
 * accept: report FINISHED as soon as it is opened and return an OK / accepted
 * result. This mirrors the existing HLE libSceMsgDialog.
 *
 * SceCommonDialogStatus: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
 */

#include "libSceSaveDataDialog.h"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace {

enum {
  STATUS_NONE = 0,
  STATUS_INITIALIZED = 1,
  STATUS_RUNNING = 2,
  STATUS_FINISHED = 3,
};

// Whole dialog lifecycle is a single global (like the real per-process singleton;
// only one common dialog can be active at a time). Touched from the game's dialog
// pump thread and its poller, hence atomic.
std::atomic<int> g_status{STATUS_NONE};

}  // namespace

extern "C" {

int PS4ABI sceSaveDataDialogInitialize() {
  g_status.store(STATUS_INITIALIZED);
  return 0;
}

int PS4ABI sceSaveDataDialogTerminate() {
  g_status.store(STATUS_NONE);
  return 0;
}

// Open: on real HW this hands the request to the system UI and the dialog is
// RUNNING until dismissed. With no UI/user we complete it right away, so the
// game's UpdateStatus/GetStatus poll sees FINISHED on its next tick.
int PS4ABI sceSaveDataDialogOpen(const void *param) {
  (void)param;
  g_status.store(STATUS_FINISHED);
  return 0;
}

int PS4ABI sceSaveDataDialogClose() {
  g_status.store(STATUS_INITIALIZED);
  return 0;
}

int PS4ABI sceSaveDataDialogGetStatus() { return g_status.load(); }

int PS4ABI sceSaveDataDialogUpdateStatus() { return g_status.load(); }

// GetResult: OrbisSaveDataDialogResult layout (verified against the libSce port):
//   +0  u32 mode        (leave 0)
//   +4  u32 result      (0 == SCE_COMMON_DIALOG_RESULT_OK)
//   +8  u32 buttonId    (1 == OK / YES)
//   +16 ptr dirName, +24 ptr param, +32 ptr userData, +40 [32] reserved
// The caller supplies a 72-byte struct it has already zeroed; we only need to
// assert the accepted/OK outcome so the game proceeds as if the user confirmed.
int PS4ABI sceSaveDataDialogGetResult(void *result) {
  if (result) {
    auto *r = static_cast<uint8_t *>(result);
    const uint32_t ok = 0;         // result = OK
    const uint32_t buttonOk = 1;   // buttonId = OK/YES
    std::memcpy(r + 4, &ok, 4);
    std::memcpy(r + 8, &buttonOk, 4);
  }
  return 0;
}

int PS4ABI sceSaveDataDialogIsReadyToDisplay() { return 1; }

int PS4ABI sceSaveDataDialogProgressBarSetValue(uint32_t target, uint32_t rate) {
  (void)target;
  (void)rate;
  return 0;
}

int PS4ABI sceSaveDataDialogProgressBarInc(uint32_t target, uint32_t delta) {
  (void)target;
  (void)delta;
  return 0;
}

}  // extern "C"
