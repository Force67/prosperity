/*
 * PS4Delta : PS4 emulation and research project
 *
 * HLE libSceMsgDialog. Like libSceVideoOut/libSceGnmDriver, the real module is a
 * boot module whose init never runs in our env, so its internal state/callbacks
 * are null and the first call (rebirth polls sceMsgDialogUpdateStatus every
 * frame) jumps through an unset pointer and crashes (then corrupts the kernel
 * object table). Override the API: report "no dialog active" so the game's
 * per-frame dialog pump is a clean no-op, and complete any dialog the game does
 * open immediately with a default OK result.
 *
 * SceCommonDialogStatus: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
 */

#include "libSceMsgDialog.h"

#include <atomic>
#include <cstdint>

namespace {
std::atomic<bool> g_initialized{false};
std::atomic<bool> g_open{false};
}  // namespace

extern "C" {

int PS4ABI sceMsgDialogInitialize() {
  g_initialized.store(true);
  return 0;
}

int PS4ABI sceMsgDialogTerminate() {
  g_initialized.store(false);
  g_open.store(false);
  return 0;
}

int PS4ABI sceMsgDialogOpen(const void *param) {
  g_open.store(true);
  return 0;
}

int PS4ABI sceMsgDialogClose() {
  g_open.store(false);
  return 0;
}

// Per-frame status pump. With no dialog open, report NONE (0) so the game's
// poll loop just continues; an opened dialog completes immediately (FINISHED).
int PS4ABI sceMsgDialogUpdateStatus() {
  return g_open.load() ? 3 /*FINISHED*/ : 0 /*NONE*/;
}

int PS4ABI sceMsgDialogGetStatus() {
  return g_open.load() ? 3 : 0;
}

int PS4ABI sceMsgDialogGetResult(void *result) {
  // Leave the caller-provided result struct as-is (games zero it first); a 0
  // return is success with the default (no button / OK) selection. Mark the
  // dialog consumed.
  g_open.store(false);
  return 0;
}

int PS4ABI sceMsgDialogProgressBarSetValue(uint32_t target, uint32_t rate) {
  return 0;
}

int PS4ABI sceMsgDialogProgressBarInc(uint32_t target, uint32_t delta) {
  return 0;
}

int PS4ABI sceMsgDialogProgressBarSetMsg(uint32_t target, const char *msg) {
  return 0;
}

}  // extern "C"
