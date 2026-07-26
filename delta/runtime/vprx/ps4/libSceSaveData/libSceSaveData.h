#pragma once

#include "../../vprx.h"

// libSceSaveData HLE. PS4 savedata is client/server: the LLE libSceSaveData.sprx
// only forwards each call over IPMI to the SceSaveData system-service process,
// which we do not host, so the sprx blocks forever waiting for a reply. We
// cannot LLE the daemon (no decrypted binary / sealed-image crypto), so -- like
// shadPS4 -- we replace the library functions and back them with a plain host
// directory (a writable VFS mount per mounted save). Mount returns a /savedataN
// mount point the game then does normal file I/O under.
//
// Saves live under $DELTA_SAVEDATA_DIR (default ~/.prosperity/savedata) keyed by
// <TITLE_ID>/<dirName>, so different games can't collide on a shared dirName.
// The title id comes from the pkg's param.sfo (plumbed by dcore). Saves created
// before per-title roots existed (a bare <dirName> under the root) are still
// found: mount/param lookups fall back to that legacy path when no per-title
// directory exists yet.
//
// Layouts (Orbis, byte offsets):
//   Mount2 { s32 userId@0; DirName*@8; u64 blocks@16; u32 mountMode@24; }
//   Mount  { s32 userId@0; TitleId*@8; DirName*@16; Fingerprint*@24;
//            u64 blocks@32; u32 mountMode@40; }
//   Mount3 { s32 userId@0; DirName*@8; u64 blocks@16; u64 systemBlocks@24;
//            u32 mountMode@32; s32 resource@40; }
//   MountResult { char mount_point[16]@0; u64 required_blocks@16; u32 unused@24;
//                 u32 mount_status@28; u8 rsv[28]; }
//   DirName { char data[32] }.  MountPoint { char data[16] }.
//   MountInfo { u64 blocks@0; u64 freeBlocks@8; u8 rsv[32] }.
//   Delete / CheckBackupData / RestoreBackupData: DirName*@16.
//   DirNameSearchCond { s32 userId@0; TitleId*@8; DirName*@16; u32 key@24;
//                       u32 order@28; }.
//   DirNameSearchResult { u32 hitNum@0; DirName* dirNames@8; u32 dirNamesNum@16;
//                         u32 setNum@20; Param* params@24; SearchInfo* infos@32; }
//   Param { char title[128]@0; char subTitle[128]@128; char detail[1024]@256;
//           u32 userParam@1280; s64 mtime@1288; u8 rsv[32]@1296; } (1328 bytes)
//   Icon { void* buf@0; u64 bufSize@8; u64 dataSize@16; }.
//   MemoryData { void* buf@0; u64 bufSize@8; s64 offset@16; }.
//   MemoryGet2 { s32 userId@0; MemoryData* data@8; ...; u32 slotId@32; }
//   MemorySet2 { s32 userId@0; MemoryData* data@8; ...; u32 slotId@36; }
//   MemorySetup2 { u32 option@0; s32 userId@4; u64 memorySize@8;
//                  u64 iconMemorySize@16; ...; u32 slotId@40; }
//   MemorySetupResult { u64 existedMemorySize@0; }.

int PS4ABI sceSaveDataInitialize(void *param);
int PS4ABI sceSaveDataInitialize2(void *param);
int PS4ABI sceSaveDataInitialize3(void *param);
int PS4ABI sceSaveDataTerminate();

int PS4ABI sceSaveDataMount(const void *mount, void *result);
int PS4ABI sceSaveDataMount2(const void *mount, void *result);
int PS4ABI sceSaveDataMount3(const void *mount, void *result);
int PS4ABI sceSaveDataMount5(const void *mount, void *result);
int PS4ABI sceSaveDataUmount(const void *mountPoint);
int PS4ABI sceSaveDataUmountWithBackup(const void *mountPoint);
int PS4ABI sceSaveDataUmount2(uint32_t mode, const void *mountPoint);
int PS4ABI sceSaveDataGetMountInfo(const void *mountPoint, void *info);

int PS4ABI sceSaveDataCreateTransactionResource(uint32_t size);
int PS4ABI sceSaveDataDeleteTransactionResource(int32_t resource);
int PS4ABI sceSaveDataPrepare(const void *mountPoint, const void *param);
int PS4ABI sceSaveDataCommit(const void *param);

int PS4ABI sceSaveDataDirNameSearch(const void *cond, void *result);
int PS4ABI sceSaveDataDelete(const void *del);
int PS4ABI sceSaveDataCheckBackupData(const void *check);
int PS4ABI sceSaveDataRestoreBackupData(const void *restore);

int PS4ABI sceSaveDataGetParam(const void *mountPoint, uint32_t paramType,
                               void *buf, uint64_t size, uint64_t *result);
int PS4ABI sceSaveDataSetParam(const void *mountPoint, uint32_t paramType,
                               const void *buf, uint64_t size);
int PS4ABI sceSaveDataSaveIcon(const void *mountPoint, const void *icon);
int PS4ABI sceSaveDataLoadIcon(const void *mountPoint, void *icon);

// SaveDataMemory: a small fixed-size memory blob per (title, slot), used for
// auto-save style data. Backed by a host file under sce_sdmemory/.
int PS4ABI sceSaveDataSetupSaveDataMemory(uint32_t userId, uint64_t memorySize,
                                          void *param);
int PS4ABI sceSaveDataSetupSaveDataMemory2(const void *setupParam, void *result);
int PS4ABI sceSaveDataGetSaveDataMemory(uint32_t userId, void *buf,
                                        uint64_t bufSize, int64_t offset);
int PS4ABI sceSaveDataGetSaveDataMemory2(void *getParam);
int PS4ABI sceSaveDataSetSaveDataMemory(uint32_t userId, const void *buf,
                                        uint64_t bufSize, int64_t offset);
int PS4ABI sceSaveDataSetSaveDataMemory2(const void *setParam);
int PS4ABI sceSaveDataSyncSaveDataMemory(void *syncParam);
int PS4ABI sceSaveDataRestoreLoadSaveDataMemory(const void *param);
