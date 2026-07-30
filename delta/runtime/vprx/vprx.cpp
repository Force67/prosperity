
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "vprx.h"
#include <crypto/sha1.h>
#include <base/containers/vector.h>
#include <cstdlib>
#include <cstring>

#include "kern/proc.h"

namespace runtime {
static base::Vector<const modInfo *> vprxTable;
// PS5-only NID alias tables (runtime/vprx/ps5/*). Kept separate from vprxTable so
// PS4 resolution is byte-for-byte unchanged; only vprx_get_forced (PS5) reads it.
static base::Vector<const modInfo *> vprxTablePs5;

// HLE-module anchors. Each vprx HLE module's _api.cpp defines one of these; we
// reference them here so the linker keeps those archive members (otherwise the
// MODULE_INIT static initializers never run and the HLE tables stay empty).
extern "C" int vprx_anchor_libSceVideoOut;
// PS5 module copies (runtime/vprx/ps5/*). Separate registry (vprxTablePs5).
extern "C" int vprx_anchor_ps5_libSceVideoOut;
extern "C" int vprx_anchor_ps5_libSceUserService;
// A few libkernel exports newer SDK libc.prx builds import that firmware 01.14.00
// doesn't export at all; the rest of libkernel stays LLE.
extern "C" int vprx_anchor_ps5_libkernel;
// Same story for the AGC/Ngs2 exports newer-SDK titles import.
extern "C" int vprx_anchor_ps5_libSceAgcDriver;
extern "C" int vprx_anchor_ps5_libSceAgc;
extern "C" int vprx_anchor_ps5_libSceNgs2;
extern "C" int vprx_anchor_libSceGnmDriver;
extern "C" int vprx_anchor_libSceMsgDialog;
// Pad + userService HLE: a connected controller + one logged-in user lets the
// title advance into actual gameplay (the userService init override avoids the
// IPMI sign-in spin). Mbus still busy-polls /dev/usbctl on a worker but that no
// longer blocks boot or rendering.
extern "C" int vprx_anchor_libScePad;
extern "C" int vprx_anchor_libSceUserService;
extern "C" int vprx_anchor_libSceUsbd;
extern "C" int vprx_anchor_libSceAudioOut;
extern "C" int vprx_anchor_libSceAudioIn;
extern "C" int vprx_anchor_libSceNpTrophy;
// HLE libSceAvPlayer: stub the movie player so intro/cutscene playback is skipped
// instead of crashing the un-emulated H.264/Atrac9 decode threads.
extern "C" int vprx_anchor_libSceAvPlayer;
// Partial HLE override: only sceSystemServiceReportAbnormalTermination (the rest
// of libSceSystemService stays LLE). Stops the title's fatal-error reporter from
// tripping the real .sprx's NULL-arg assert.
extern "C" int vprx_anchor_libSceSystemService;
// HLE libfmod: the game's bundled FMOD .prx. Its real init needs the un-emulated
// AJM ATRAC9 decoder; stub the API to "succeed" with null audio so Doom64 boots.
extern "C" int vprx_anchor_libfmod;
// HLE libSceNetCtl: report a connected wired network (state IPOBTAINED). The
// LLE .sprx polls a non-existent system net daemon, so titles that gate boot on
// connectivity (PT) would stall 10s and then continue down a broken init path.
extern "C" int vprx_anchor_libSceNetCtl;
// HLE libSceSaveData: PS4 saves are client/server (the LLE .sprx forwards over
// IPMI to the SceSaveData system-service process we don't host, so it blocks
// forever). Replace the library and back saves with a writable host directory.
extern "C" int vprx_anchor_libSceSaveData;
// HLE libSceSaveDataDialog: the LLE .sprx forwards the dialog to the SceShellUI
// service (over IPMI) we don't host, so its status never reaches FINISHED and a
// title that waits for the save dialog to close (PT's world-load save flow)
// hangs. Complete the dialog immediately with a default OK.
extern "C" int vprx_anchor_libSceSaveDataDialog;
static volatile int *const vprx_anchors[] = {&vprx_anchor_libSceVideoOut,
                                             &vprx_anchor_ps5_libSceVideoOut,
                                             &vprx_anchor_ps5_libSceUserService,
                                             &vprx_anchor_ps5_libkernel,
                                             &vprx_anchor_ps5_libSceAgcDriver,
                                             &vprx_anchor_ps5_libSceAgc,
                                             &vprx_anchor_ps5_libSceNgs2,
                                             &vprx_anchor_libSceSaveData,
                                             &vprx_anchor_libSceSaveDataDialog,
                                             &vprx_anchor_libfmod,
                                             &vprx_anchor_libSceGnmDriver,
                                             &vprx_anchor_libSceMsgDialog,
                                             &vprx_anchor_libScePad,
                                             &vprx_anchor_libSceUserService,
                                             &vprx_anchor_libSceUsbd,
                                             &vprx_anchor_libSceAudioOut,
                                             &vprx_anchor_libSceAudioIn,
                                             &vprx_anchor_libSceNpTrophy,
                                             &vprx_anchor_libSceAvPlayer,
                                             &vprx_anchor_libSceSystemService,
                                             &vprx_anchor_libSceNetCtl};

void vprx_init() {
  // Touch the anchors so the references aren't optimized away.
  int sum = 0;
  for (auto *a : vprx_anchors)
    sum += *a;
  (void)sum;
  utl::init_function::init();
}

void vprx_reg(const modInfo *info) { vprxTable.push_back(info); }
void vprx_reg_ps5(const modInfo *info) { vprxTablePs5.push_back(info); }

// Per-module HLE policy. We prefer running the real sprx (LLE) for modules whose
// syscall/device backing we emulate, falling back to the HLE shim only when the
// real path isn't ready or is forced off.
//   - libSceGnmDriver: LLE by default (PM4 via ioctl(/dev/gc) -> gcDevice -> the
//     GPU command processor). Force the HLE submit shim with DELTA_GNM_HLE.
//   - libSceVideoOut: LLE by default; the real module drives the framebuffer
//     through ioctl(/dev/dce) + mmap (dceDevice) and flips via the videoout
//     service thread. Force the HLE shim with DELTA_VO_HLE.
// DIAGNOSTIC: force just a few specific NIDs of an otherwise-LLE module onto the
// HLE shim. Env is a comma/space list of hex hids, e.g.
//   DELTA_HLE_NIDS_VO=0x1234...,0xabcd...
// Lets us binary-search which single videoout/gnm export's real behavior triggers
// the both-LLE Isaac crash, without recompiling per test.
static bool nidForcedHle(const char *envName, uint64_t hid) {
  const char *list = std::getenv(envName);
  if (!list)
    return false;
  for (const char *p = list; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *end = nullptr;
    uint64_t v = std::strtoull(p, &end, 16);
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
static bool libListed(const char *envName, const char *lib) {
  const char *list = std::getenv(envName);
  if (!list || !*list)
    return false;
  if (std::strcmp(list, "all") == 0 || std::strcmp(list, "1") == 0)
    return true;
  for (const char *p = list; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    const char *end = p;
    while (*end && *end != ',' && *end != ' ')
      end++;
    const size_t n = static_cast<size_t>(end - p);
    if (n) {
      // Substring match of the list entry against the library name.
      for (const char *h = lib; *h; h++) {
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
// SceShellUI / SceSysCore), not because LLE was tried and rejected -- so which
// of them could actually run LLE is an open question per module. Rather than
// answer it by recompiling, make the policy switchable:
//   DELTA_LLE=<list>  force LLE (ignore the HLE shim) for these libraries
//   DELTA_HLE=<list>  force HLE, and it wins over DELTA_LLE
// Both take a comma/space list of substrings, or "all". So a bisect looks like
//   DELTA_LLE=all DELTA_HLE=libSceSaveDataDialog
// A library with no HLE table registered is LLE regardless; forcing LLE on one
// that the guest then calls into an unhosted IPMI service will hang or fault,
// which is exactly the information the switch is there to obtain.
//
// MEASURED, so nobody repeats the mistake: "boots and does not crash under
// DELTA_LLE=all" is NOT the same as "these modules work". Two known traps.
//
//  - libSceAudioOut LLE is SILENT. The host sink (prosperity_audio_output ->
//    gfx_audio -> SDL) is reached only from the HLE shim, and there is no audio
//    device under kern/ps4/dev, so the real module opens its port and writes
//    into nothing. Nothing in a crash/fps check can see this. Hosting audio LLE
//    needs a device, not a shim bypass.
//  - The common dialogs (libSceSaveDataDialog, libSceMsgDialog) LLE forward to
//    a ShellUI daemon that kern/ipmi does not stand in for yet (it has PlayGo,
//    NpManager, NpWeb and UserService), so their status never leaves RUNNING for
//    a title that actually opens one.
//
// What IS verified: the switch itself is airtight -- under DELTA_LLE=all the HLE
// trace records zero thunk calls, so every registered shim really is bypassed.
static bool useHleShim(const char *lib, uint64_t hid) {
  if (libListed("DELTA_HLE", lib))
    return true;
  if (libListed("DELTA_LLE", lib))
    return false;
  if (std::strcmp(lib, "libSceGnmDriver") == 0)
    return std::getenv("DELTA_GNM_HLE") != nullptr ||
           nidForcedHle("DELTA_HLE_NIDS_GNM", hid);
  if (std::strcmp(lib, "libSceVideoOut") == 0)
    return std::getenv("DELTA_VO_HLE") != nullptr ||
           nidForcedHle("DELTA_HLE_NIDS_VO", hid);
  return true;  // every other HLE module stays HLE
}

uintptr_t vprx_get_forced(const char *lib, uint64_t hid) {
  // PS5-only: resolve exclusively from the PS5 registry (runtime/vprx/ps5/*).
  // PS5 must NOT borrow the PS4 HLE modules -- each forced-HLE library has its own
  // full PS5 copy so behaviour can diverge safely. A miss here falls through to the
  // real .sprx (LLE) in the caller, never to a PS4 stub.
  for (const auto &t : vprxTablePs5) {
    if (std::strcmp(lib, t->namePtr) != 0)
      continue;
    for (int i = 0; i < t->funcCount; i++)
      if (t->funcNodes[i].hashId == hid)
        return reinterpret_cast<uintptr_t>(t->funcNodes[i].address);
  }
  return 0;
}

uintptr_t vprx_get(const char *lib, uint64_t hid) {
  // The Neo SPRX is a filename variant of the libSceGnmDriver ABI. Keep its
  // imports on the same HLE/LLE policy and HLE export table as the Base module.
  if (std::strcmp(lib, "libSceGnmDriver") == 0 ||
      std::strcmp(lib, "libSceGnmDriverForNeoMode") == 0)
    lib = "libSceGnmDriver";

  if (!useHleShim(lib, hid))
    return 0;

  const modInfo *table = nullptr;

  // find the right table
  for (const auto &t : vprxTable) {
    if (std::strcmp(lib, t->namePtr) == 0) {
      table = t;
      break;
    }
  }

  if (table) {
    // search the table
    for (int i = 0; i < table->funcCount; i++) {
      auto *f = &table->funcNodes[i];
      if (f->hashId == hid) {
        return reinterpret_cast<uintptr_t>(f->address);
      }
    }
  }

  // DELTA_NID_TRACE: report imports with no HLE override (resolved to the LLE
  // module). Set it to a library-name substring to focus the dump, or "1" for
  // all. Fires once per import at load time, so it stays bounded.
  if (const char *t = std::getenv("DELTA_NID_TRACE")) {
    if (t[0] == '1' || std::strstr(lib, t))
      std::fprintf(stderr, "[nid] %s hid=%#018llx -> LLE (no HLE)\n", lib,
                   (unsigned long long)hid);
  }
  return 0;
}

const char base64Lookup[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

// base64 fast lookup
bool decode_nid(const char *subset, size_t len, uint64_t &out) {
  for (size_t i = 0; i < len; i++) {
    auto pos = std::strchr(base64Lookup, subset[i]);

    // invalid NID?
    if (!pos) {
      return false;
    }

    auto offset = static_cast<uint32_t>(pos - base64Lookup);

    // max NID is 11
    if (i < 10) {
      out <<= 6;
      out |= offset;
    } else {
      out <<= 4;
      out |= (offset >> 2);
    }
  }

  return true;
}

static void obfuscate_sym(uint64_t in, uint8_t *out, size_t xlen) {
  out[xlen--] = 0;
  out[xlen--] = base64Lookup[(in & 0xF) * 4];
  uint64_t exp = in >> 4;
  while (exp != 0) {
    out[xlen--] = base64Lookup[exp & 0x3F];
    exp = exp >> 6;
  }
}

void encode_nid(const char *name, uint8_t *x) {
  static const char suffix[] =
      "\x51\x8D\x64\xA6\x35\xDE\xD8\xC1\xE6\xB0\x39\xB1\xC3\xE5\x52\x30";

  uint8_t sha[20]{};
  sha1_context ctx;

  sha1_starts(&ctx);
  sha1_update(&ctx, reinterpret_cast<const uint8_t *>(name), std::strlen(name));
  sha1_update(&ctx, reinterpret_cast<const uint8_t *>(suffix),
              std::strlen(suffix));
  sha1_finish(&ctx, sha);

  /*the rest is ignored*/
  uint64_t target = *(uint64_t *)(&sha);

  // uint8_t out[11]{};
  obfuscate_sym(target, x, 11);
}
} // namespace runtime
