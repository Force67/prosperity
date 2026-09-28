
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "runtime/vprx/vprx.h"
#include <cstdlib>
#include <cstring>
#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/logging.h"

#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kHleLibs, "DELTA_HLE", nullptr);
DELTA_OPTION(const char*, kLleLibs, "DELTA_LLE", nullptr);
DELTA_OPTION(const char*, kHleNidsGnm, "DELTA_HLE_NIDS_GNM", nullptr);
DELTA_OPTION(const char*, kHleNidsVo, "DELTA_HLE_NIDS_VO", nullptr);
DELTA_OPTION(const char*, kNidTrace, "DELTA_NID_TRACE", nullptr);
DELTA_OPTION(bool, kGnmHle, "DELTA_GNM_HLE", false);
DELTA_OPTION(bool, kVoHle, "DELTA_VO_HLE", false);
}  // namespace

namespace runtime::vprx {
static base::Vector<const ExportTable*> g_vprx_table;
// PS5-only NID alias tables (runtime/vprx/ps5/*). Kept separate from vprxTable
// so PS4 resolution is byte-for-byte unchanged; only LookupForced (PS5) reads
// it.
static base::Vector<const ExportTable*> g_vprx_table_ps5;

// HLE-module anchors. Each vprx HLE module's _exports.cc defines one of these;
// we reference them here so the linker keeps those archive members (otherwise
// the MODULE_INIT static initializers never run and the HLE tables stay empty).
extern "C" int g_vprx_anchor_lib_sce_video_out;
// PS5 module copies (runtime/vprx/ps5/*). Separate registry (vprxTablePs5).
extern "C" int g_vprx_anchor_ps5_lib_sce_video_out;
extern "C" int g_vprx_anchor_ps5_lib_sce_videodec2;
extern "C" int g_vprx_anchor_ps5_lib_sce_user_service;
// A few libkernel exports newer SDK libc.prx builds import that firmware
// 01.14.00 doesn't export at all; the rest of libkernel stays LLE.
extern "C" int g_vprx_anchor_ps5_libkernel;
// Same story for the AGC/Ngs2 exports newer-SDK titles import.
extern "C" int g_vprx_anchor_ps5_lib_sce_agc_driver;
extern "C" int g_vprx_anchor_ps5_lib_sce_agc;
extern "C" int g_vprx_anchor_ps5_lib_sce_ngs2;
// Forced-HLE sceImeKeyboardOpen: the LLE one needs the IME service daemon and
// fails with a code titles don't expect from it (see ps5/lib_sce_ime.cc).
extern "C" int g_vprx_anchor_ps5_lib_sce_ime;
extern "C" int g_vprx_anchor_ps5_lib_sce_app_content;
// Same abnormal-termination reporter override the PS4 HLE has.
extern "C" int g_vprx_anchor_ps5_lib_sce_system_service;
// No-op sanitizer fiber hooks; retail fw ships no TSan/ASan runtime for
// libSceFiber to import them from (see ps5/lib_sce_fiber.cc).
extern "C" int g_vprx_anchor_ps5_lib_sce_fiber;
extern "C" int g_vprx_anchor_lib_sce_gnm_driver;
extern "C" int g_vprx_anchor_lib_sce_msg_dialog;
// Pad + UserService HLE: a connected controller + one logged-in user lets the
// title advance into actual gameplay (the UserService init override avoids the
// IPMI sign-in spin). Mbus still busy-polls /dev/usbctl on a worker but that no
// longer blocks boot or rendering.
extern "C" int g_vprx_anchor_lib_sce_pad;
extern "C" int g_vprx_anchor_lib_sce_user_service;
extern "C" int g_vprx_anchor_lib_sce_usbd;
extern "C" int g_vprx_anchor_lib_sce_audio_out;
extern "C" int g_vprx_anchor_lib_sce_audio_in;
extern "C" int g_vprx_anchor_lib_sce_np_trophy;
// HLE libSceAvPlayer: stub the movie player so intro/cutscene playback is
// skipped instead of crashing the un-emulated H.264/Atrac9 decode threads.
extern "C" int g_vprx_anchor_lib_sce_av_player;
extern "C" int g_vprx_anchor_lib_sce_videodec2;
// Partial HLE override: only sceSystemServiceReportAbnormalTermination (the
// rest of libSceSystemService stays LLE). Stops the title's fatal-error
// reporter from tripping the real .sprx's NULL-arg assert.
extern "C" int g_vprx_anchor_lib_sce_system_service;
// HLE libfmod: the game's bundled FMOD .prx. Its real init needs the
// un-emulated AJM ATRAC9 decoder; stub the API to "succeed" with null audio so
// Doom64 boots.
extern "C" int g_vprx_anchor_libfmod;
// HLE libSceNetCtl: report a connected wired network (state IPOBTAINED). The
// LLE .sprx polls a non-existent system net daemon, so titles that gate boot on
// connectivity (PT) would stall 10s and then continue down a broken init path.
extern "C" int g_vprx_anchor_lib_sce_net_ctl;
// HLE libSceSaveData: PS4 saves are client/server (the LLE .sprx forwards over
// IPMI to the SceSaveData system-service process we don't host, so it blocks
// forever). Replace the library and back saves with a writable host directory.
extern "C" int g_vprx_anchor_lib_sce_save_data;
// HLE libSceSaveDataDialog: the LLE .sprx forwards the dialog to the SceShellUI
// service (over IPMI) we don't host, so its status never reaches FINISHED and a
// title that waits for the save dialog to close (PT's world-load save flow)
// hangs. Complete the dialog immediately with a default OK.
extern "C" int g_vprx_anchor_lib_sce_save_data_dialog;
static volatile int* const kVprxAnchors[] = {
    &g_vprx_anchor_lib_sce_video_out,
    &g_vprx_anchor_ps5_lib_sce_video_out,
    &g_vprx_anchor_ps5_lib_sce_videodec2,
    &g_vprx_anchor_ps5_lib_sce_user_service,
    &g_vprx_anchor_ps5_libkernel,
    &g_vprx_anchor_ps5_lib_sce_agc_driver,
    &g_vprx_anchor_ps5_lib_sce_agc,
    &g_vprx_anchor_ps5_lib_sce_ngs2,
    &g_vprx_anchor_ps5_lib_sce_ime,
    &g_vprx_anchor_ps5_lib_sce_app_content,
    &g_vprx_anchor_ps5_lib_sce_system_service,
    &g_vprx_anchor_ps5_lib_sce_fiber,
    &g_vprx_anchor_lib_sce_save_data,
    &g_vprx_anchor_lib_sce_save_data_dialog,
    &g_vprx_anchor_libfmod,
    &g_vprx_anchor_lib_sce_gnm_driver,
    &g_vprx_anchor_lib_sce_msg_dialog,
    &g_vprx_anchor_lib_sce_pad,
    &g_vprx_anchor_lib_sce_user_service,
    &g_vprx_anchor_lib_sce_usbd,
    &g_vprx_anchor_lib_sce_audio_out,
    &g_vprx_anchor_lib_sce_audio_in,
    &g_vprx_anchor_lib_sce_np_trophy,
    &g_vprx_anchor_lib_sce_av_player,
    &g_vprx_anchor_lib_sce_videodec2,
    &g_vprx_anchor_lib_sce_system_service,
    &g_vprx_anchor_lib_sce_net_ctl};

void Init() {
  // Touch the anchors so the references aren't optimized away.
  int sum = 0;
  for (auto* a : kVprxAnchors)
    sum += *a;
  (void)sum;
  runtime::InitFunction::Init();
}

void Register(const ExportTable* info) {
  g_vprx_table.push_back(info);
}
void RegisterPs5(const ExportTable* info) {
  g_vprx_table_ps5.push_back(info);
}

// Per-module HLE policy. We prefer running the real sprx (LLE) for modules
// whose syscall/device backing we emulate, falling back to the HLE shim only
// when the real path isn't ready or is forced off.
//   - libSceGnmDriver: LLE by default (PM4 via ioctl(/dev/gc) -> GcDevice ->
//   the
//     GPU command processor). Force the HLE submit shim with DELTA_GNM_HLE.
//   - libSceVideoOut: LLE by default; the real module drives the framebuffer
//     through ioctl(/dev/dce) + mmap (DceDevice) and flips via the videoout
//     service thread. Force the HLE shim with DELTA_VO_HLE.
// DIAGNOSTIC: force just a few specific NIDs of an otherwise-LLE module onto
// the HLE shim. Env is a comma/space list of hex hids, e.g.
//   DELTA_HLE_NIDS_VO=0x1234...,0xabcd...
// Lets us binary-search which single videoout/gnm export's real behavior
// triggers the both-LLE Isaac crash, without recompiling per test.
static bool NidForcedHle(const char* list, u64 hid) {
  if (!list)
    return false;
  for (const char* p = list; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char* end = nullptr;
    u64 v = std::strtoull(p, &end, 16);
    if (end == p)
      break;
    if (v == hid)
      return true;
    p = end;
  }
  return false;
}

// Does `lib` appear in a comma/space separated env list? "all" matches every
// library, so one variable can flip the whole default. Names match on a
// substring so "SaveData" covers libSceSaveData and libSceSaveDataDialog.
static bool LibListed(const char* list, const char* lib) {
  if (!list || !*list)
    return false;
  if (std::strcmp(list, "all") == 0 || std::strcmp(list, "1") == 0)
    return true;
  for (const char* p = list; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    const char* end = p;
    while (*end && *end != ',' && *end != ' ')
      end++;
    const size_t n = static_cast<size_t>(end - p);
    if (n) {
      // Substring match of the list entry against the library name.
      for (const char* h = lib; *h; h++) {
        if (std::strncmp(h, p, n) == 0)
          return true;
      }
    }
    p = end;
  }
  return false;
}

// Returns true when `lib`'s HLE shim should be used for this NID (skip = LLE).
//
// The HLE shims for the non-graphics service modules exist because their real
// sprx forwards over IPMI to a system service we do not host (SceShellCore /
// SceShellUI / SceSysCore), not because LLE was tried and rejected, so which of
// them could actually run LLE is an open question per module. Rather than
// answer it by recompiling, make the policy switchable:
//   DELTA_LLE=<list>  force LLE (ignore the HLE shim) for these libraries
//   DELTA_HLE=<list>  force HLE, and it wins over DELTA_LLE
// Both take a comma/space list of substrings, or "all". A bisect looks like
//   DELTA_LLE=all DELTA_HLE=libSceSaveDataDialog
// A library with no HLE table registered is LLE regardless; forcing LLE on one
// that the guest then calls into an unhosted IPMI service will hang or fault,
// which is exactly the information the switch is there to obtain.
//
// MEASURED, so nobody repeats the mistake: "boots and does not crash under
// DELTA_LLE=all" is NOT the same as "these modules work". Two known traps:
//  - libSceAudioOut LLE was silent: the real module needs no /dev node, it
//    hands blocks to the system audio daemon over POSIX shm and waits on a
//    named event flag, so with no daemon it wrote into nothing and no crash/fps
//    check could see it. That daemon is now hosted (kern/ps4/audio_daemon.cc);
//    the lesson stands for every module whose LLE partner is a system service.
//  - The common dialogs (libSceSaveDataDialog, libSceMsgDialog) LLE forward to
//    a ShellUI daemon that kern/ipmi does not stand in for yet, so their status
//    never leaves RUNNING for a title that actually opens one.
//
// What IS verified: the switch itself is airtight; under DELTA_LLE=all the HLE
// trace records zero thunk calls, so every registered shim really is bypassed.
static bool UseHleShim(const char* lib, u64 hid) {
  if (LibListed(kHleLibs, lib))
    return true;
  if (LibListed(kLleLibs, lib))
    return false;
  if (std::strcmp(lib, "libSceGnmDriver") == 0)
    // sceGnmDingDong is the exception to keeping this module LLE. It is the
    // doorbell: the real driver stores the ring's write pointer into its own
    // /dev/gc mapping, which announces nothing to us, so the async-compute work
    // a title queues there is either never run (the flip-time drain budget
    // defaults off) or run at the next flip, by which time the guest has
    // recycled the buffers those packets point at. Taking the call gives us the
    // one moment the ring is known good.
    return kGnmHle || hid == 0x6D7E486D1BC40979ull ||
           NidForcedHle(kHleNidsGnm, hid);
  if (std::strcmp(lib, "libSceVideoOut") == 0)
    return kVoHle || NidForcedHle(kHleNidsVo, hid);
  return true;  // every other HLE module stays HLE
}

uintptr_t LookupForced(const char* lib, u64 hid) {
  // Keep native decoder execution available for GPU accuracy investigations.
  static const bool kNativeVideo = [] {
    const char* value = std::getenv("DELTA_PS5_NATIVE_VIDEO");
    return value && std::strcmp(value, "1") == 0;
  }();
  if (kNativeVideo && std::strcmp(lib, "libSceVideodec2") == 0)
    return 0;
  // PS5-only: resolve exclusively from the PS5 registry (runtime/vprx/ps5/*).
  // PS5 must NOT borrow the PS4 HLE modules, since each forced-HLE library has
  // its own full PS5 copy so behaviour can diverge safely. A miss here falls
  // through to the real .sprx (LLE) in the caller, never to a PS4 stub.
  for (const auto& t : g_vprx_table_ps5) {
    if (std::strcmp(lib, t->library) != 0)
      continue;
    for (int i = 0; i < t->count; i++)
      if (t->entries[i].nid == hid)
        return reinterpret_cast<uintptr_t>(t->entries[i].address);
  }
  return 0;
}

uintptr_t Lookup(const char* lib, u64 hid) {
  // The Neo SPRX is a filename variant of the libSceGnmDriver ABI. Keep its
  // imports on the same HLE/LLE policy and HLE export table as the Base module.
  if (std::strcmp(lib, "libSceGnmDriver") == 0 ||
      std::strcmp(lib, "libSceGnmDriverForNeoMode") == 0)
    lib = "libSceGnmDriver";

  // Same idea for NpToolkit2's private entry points into libSceNetCtl: they
  // live in the .sprx we already shim, so leaving them LLE means they read an
  // "initialized" flag our HLE never sets.
  if (std::strcmp(lib, "libSceNetCtlForNpToolkit") == 0)
    lib = "libSceNetCtl";
  if (std::strcmp(lib, "libSceUserServiceForNpToolkit") == 0)
    lib = "libSceUserService";

  if (!UseHleShim(lib, hid))
    return 0;

  const ExportTable* table = nullptr;

  // find the right table
  for (const auto& t : g_vprx_table) {
    if (std::strcmp(lib, t->library) == 0) {
      table = t;
      break;
    }
  }

  if (table) {
    // search the table
    for (int i = 0; i < table->count; i++) {
      auto* f = &table->entries[i];
      if (f->nid == hid) {
        return reinterpret_cast<uintptr_t>(f->address);
      }
    }
  }

  // DELTA_NID_TRACE: report imports with no HLE override (resolved to the LLE
  // module). Set it to a library-name substring to focus the dump, or "1" for
  // all. Fires once per import at load time, so it stays bounded.
  if (const char* t = kNidTrace) {
    if (t[0] == '1' || std::strstr(lib, t))
      BASE_LOGI("nid", "{} hid={:#018x} -> LLE (no HLE)", lib,
                (unsigned long long)hid);
  }
  return 0;
}

}  // namespace runtime::vprx
