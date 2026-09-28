/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceSaveDataDialog. On real hardware the save-data dialog is drawn by
 * the system UI process (SceShellUI); the game calls sceSaveDataDialogOpen and
 * then polls sceSaveDataDialogUpdateStatus until it reports FINISHED (the user
 * dismissed it / a progress-bar op completed). The LLE
 * libSceSaveDataDialog.sprx forwards over IPMI to that dialog service, which we
 * don't host, so its status never advances past RUNNING and any title that
 * gates progression on the dialog finishing (P.T.'s world-load
 * "GameSave"/"UpdateSaveDialog" flow) hangs forever.
 *
 * We host no interactive UI and there is no user to click, so the only correct
 * emulated behaviour is to complete the dialog immediately with a default
 * accept: report FINISHED as soon as it is opened and return an OK / accepted
 * result. This mirrors the existing HLE libSceMsgDialog.
 *
 * SceCommonDialogStatus: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
 */

#include "runtime/vprx/ps4/lib_sce_save_data_dialog/lib_sce_save_data_dialog.h"
#include "base/arch.h"
#include "guest_abi.h"

#include <cstring>
#include "base/atomic.h"

namespace {

enum {
  kStatusNone = 0,
  kStatusInitialized = 1,
  kStatusRunning = 2,
  kStatusFinished = 3,
};

// Whole dialog lifecycle is a single global (like the real per-process
// singleton; only one common dialog can be active at a time). Touched from the
// game's dialog pump thread and its poller, hence atomic.
base::Atomic<int> g_status{kStatusNone};

// SceSaveDataDialogMode values that change how the dialog ends.
enum { kModeList = 1, kModeProgressBar = 5 };
constexpr size_t kDirNameBytes = 32;  // SceSaveDataDirName

// What the open asked for, for GetResult: a list dialog's result names the
// slot the user picked, and every result hands back the caller's userData.
u32 g_mode = 0;
u8 g_dir_name[kDirNameBytes] = {};
bool g_have_dir_name = false;
u64 g_user_data = 0;

}  // namespace

extern "C" {

int PS4ABI sceSaveDataDialogInitialize() {
  g_status.store(kStatusInitialized);
  return 0;
}

int PS4ABI sceSaveDataDialogTerminate() {
  g_status.store(kStatusNone);
  return 0;
}

// Open: on real HW this hands the request to the system UI and the dialog is
// RUNNING until dismissed. With no UI/user we complete it right away, so the
// game's UpdateStatus/GetStatus poll sees FINISHED on its next tick.
// Open: on real HW this hands the request to the system UI and the dialog is
// RUNNING until dismissed. With no UI/user, a question is answered at once:
// FINISHED on the next poll, as if the user accepted. A progress bar is not a
// question: it stays up until the title closes it, and reporting it finished
// straight away made Uncharted 2 reopen it ~900 times a second instead of
// saving its new game.
// SceSaveDataDialogParam: +0x34 mode, +0x48 items, +0x70 userData;
// SceSaveDataDialogItems: +0x10 dirName array, +0x18 dirName count.
int PS4ABI sceSaveDataDialogOpen(const void* param) {
  g_mode = 0;
  g_have_dir_name = false;
  g_user_data = 0;
  if (param) {
    const auto* p = static_cast<const u8*>(param);
    std::memcpy(&g_mode, p + 0x34, sizeof(g_mode));
    std::memcpy(&g_user_data, p + 0x70, sizeof(g_user_data));
    const u8* items = nullptr;
    std::memcpy(&items, p + 0x48, sizeof(items));
    if (items) {
      const u8* dir_names = nullptr;
      u32 count = 0;
      std::memcpy(&dir_names, items + 0x10, sizeof(dir_names));
      std::memcpy(&count, items + 0x18, sizeof(count));
      if (dir_names && count) {
        std::memcpy(g_dir_name, dir_names, kDirNameBytes);
        g_have_dir_name = true;
      }
    }
  }
  g_status.store(g_mode == kModeProgressBar ? kStatusRunning : kStatusFinished);
  return 0;
}

int PS4ABI sceSaveDataDialogClose() {
  g_status.store(kStatusFinished);
  return 0;
}

int PS4ABI sceSaveDataDialogGetStatus() {
  return g_status.load();
}

int PS4ABI sceSaveDataDialogUpdateStatus() {
  return g_status.load();
}

// GetResult: SceSaveDataDialogResult
//   +0  u32 mode, +4 i32 result (0 == OK), +8 u32 buttonId (1 == OK / YES)
//   +16 SceSaveDataDirName* dirName (the caller's buffer), +24 param,
//   +32 userData
int PS4ABI sceSaveDataDialogGetResult(void* result) {
  if (result) {
    auto* r = static_cast<u8*>(result);
    const u32 ok = 0;
    const u32 button_ok = 1;
    std::memcpy(r, &g_mode, 4);
    std::memcpy(r + 4, &ok, 4);
    std::memcpy(r + 8, &button_ok, 4);
    u8* dir_name = nullptr;
    std::memcpy(&dir_name, r + 16, sizeof(dir_name));
    if (dir_name && g_have_dir_name)
      std::memcpy(dir_name, g_dir_name, kDirNameBytes);
    std::memcpy(r + 32, &g_user_data, sizeof(g_user_data));
  }
  return 0;
}

int PS4ABI sceSaveDataDialogIsReadyToDisplay() {
  return 1;
}

int PS4ABI sceSaveDataDialogProgressBarSetValue(u32 target, u32 rate) {
  (void)target;
  (void)rate;
  return 0;
}

int PS4ABI sceSaveDataDialogProgressBarInc(u32 target, u32 delta) {
  (void)target;
  (void)delta;
  return 0;
}

}  // extern "C"
