// libScePlayGo queries, answered as the whole-pkg mount they describe: every
// chunk is local and nothing is left to install. Open/Close and friends stay
// LLE on the emulated daemon (kern/ipmi/svc_playgo.cc); the getters went
// through IPMI operations that change between firmware versions, and
// Uncharted 2's scePlayGoGetLocus came back empty, so its level loader waited
// for data that was already there.
#include <cstring>

#include "base/arch.h"
#include "guest_abi.h"
#include "kern/ipmi/services.h"
#include "runtime/vprx/vprx.h"

namespace {
constexpr int kInvalidArgument = static_cast<int>(0x80b20004);
constexpr int kBadChunkId = static_cast<int>(0x80b2000c);
constexpr u8 kLocusLocalFast = 3;
constexpr u32 kInstallSpeedFull = 2;
u64 g_language_mask = ~0ull;

// Ids past the title's chunk count are rejected: Uncharted 2 finds its chunk
// count by probing downward from 99 for the first id that succeeds.
int PS4ABI scePlayGoGetLocus(u32, const u16* ids, u32 count, u8* loci) {
  if (!ids || !loci)
    return kInvalidArgument;
  const u32 chunks = kern::ipmi::PlayGoChunkCount();
  for (u32 i = 0; i < count; i++) {
    if (ids[i] >= chunks)
      return kBadChunkId;
    loci[i] = kLocusLocalFast;
  }
  return 0;
}

int PS4ABI scePlayGoGetChunkId(u32, u16* ids, u32 max, u32* count) {
  if (!count)
    return kInvalidArgument;
  const u32 n = kern::ipmi::PlayGoChunkCount();
  *count = ids ? (n < max ? n : max) : n;
  for (u32 i = 0; ids && i < *count; i++)
    ids[i] = static_cast<u16>(i);
  return 0;
}

int PS4ABI scePlayGoGetProgress(u32, const u16*, u32, u64* progress) {
  if (!progress)
    return kInvalidArgument;
  progress[0] = progress[1] = 1;  // {progressSize, totalSize}: complete
  return 0;
}

int PS4ABI scePlayGoGetToDoList(u32, void*, u32, u32* count) {
  if (!count)
    return kInvalidArgument;
  *count = 0;
  return 0;
}

int PS4ABI scePlayGoGetEta(u32, const u16*, u32, i64* eta) {
  if (!eta)
    return kInvalidArgument;
  *eta = 0;
  return 0;
}

int PS4ABI scePlayGoGetInstallSpeed(u32, u32* speed) {
  if (!speed)
    return kInvalidArgument;
  *speed = kInstallSpeedFull;
  return 0;
}

int PS4ABI scePlayGoGetLanguageMask(u32, u64* mask) {
  if (!mask)
    return kInvalidArgument;
  *mask = g_language_mask;
  return 0;
}

int PS4ABI scePlayGoSetLanguageMask(u32, u64 mask) {
  g_language_mask = mask;
  return 0;
}

int PS4ABI scePlayGoAccepted() {
  return 0;
}

const runtime::vprx::ExportEntry kExports[] = {
    {0xb962182c5924c2a9, (void*)&scePlayGoGetLocus},
    {0xef77c5d4c154f210, (void*)&scePlayGoGetChunkId},
    {0xfd125634c2b77c2f, (void*)&scePlayGoGetProgress},
    {0x367ef32b09c0e6ad, (void*)&scePlayGoGetToDoList},
    {0xbfa119fd859174cb, (void*)&scePlayGoGetEta},
    {0xaef0527d38a67a31, (void*)&scePlayGoGetInstallSpeed},
    {0xdce31b61905a6b9d, (void*)&scePlayGoGetLanguageMask},
    {0x2e8b0b9473a936a4, (void*)&scePlayGoSetLanguageMask},
    {0x8143c688e435b664, (void*)&scePlayGoAccepted},  // SetToDoList
    {0xe0001c4d4f51dd73, (void*)&scePlayGoAccepted},  // SetInstallSpeed
    {0xfd0d7fbb56bba748, (void*)&scePlayGoAccepted},  // Prefetch
};
MODULE_INIT(libScePlayGo);
}  // namespace

extern "C" int g_vprx_anchor_lib_sce_play_go = 1;
