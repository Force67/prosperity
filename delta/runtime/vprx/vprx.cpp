
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

// HLE-module anchors. Each vprx HLE module's _api.cpp defines one of these; we
// reference them here so the linker keeps those archive members (otherwise the
// MODULE_INIT static initializers never run and the HLE tables stay empty).
extern "C" int vprx_anchor_libSceVideoOut;
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

// Returns true when `lib`'s HLE shim should be used for this NID (skip = LLE).
static bool useHleShim(const char *lib, uint64_t hid) {
  if (std::strcmp(lib, "libSceGnmDriver") == 0)
    return std::getenv("DELTA_GNM_HLE") != nullptr ||
           nidForcedHle("DELTA_HLE_NIDS_GNM", hid);
  if (std::strcmp(lib, "libSceVideoOut") == 0)
    return std::getenv("DELTA_VO_HLE") != nullptr ||
           nidForcedHle("DELTA_HLE_NIDS_VO", hid);
  return true;  // every other HLE module stays HLE
}

uintptr_t vprx_get_forced(const char *lib, uint64_t hid) {
  for (const auto &t : vprxTable) {
    if (std::strcmp(lib, t->namePtr) != 0)
      continue;
    for (int i = 0; i < t->funcCount; i++)
      if (t->funcNodes[i].hashId == hid)
        return reinterpret_cast<uintptr_t>(t->funcNodes[i].address);
  }
  return 0;
}

uintptr_t vprx_get(const char *lib, uint64_t hid) {
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
