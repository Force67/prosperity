#pragma once

#include "../vprx.h"

// libSceSaveData HLE. PS4 savedata is client/server: the LLE libSceSaveData.sprx
// only forwards each call over IPMI to the SceSaveData system-service process,
// which we do not host, so the sprx blocks forever waiting for a reply. We
// cannot LLE the daemon (no decrypted binary / sealed-image crypto), so -- like
// shadPS4 -- we replace the library functions and back them with a plain host
// directory (a writable VFS mount per mounted save). Mount returns a
// /savedataN mount point the game then does normal file I/O under.
//
// Layouts (Orbis, byte offsets): Mount2 { s32 userId@0; dirName*@8; u64
// blocks@16; u32 mountMode@24; u8 rsv[32] }. Mount { s32 userId@0; titleId*@8;
// dirName*@16; fingerprint*@24; u64 blocks@32; u32 mountMode@40; u8 rsv[32] }.
// MountResult { char mount_point[16]@0; u64 required_blocks@16; u32 unused@24;
// u32 mount_status@28; u8 rsv[28] }. DirName { char data[32] }. MountPoint {
// char data[16] }. MountInfo { u64 blocks@0; u64 freeBlocks@8; u8 rsv[32] }.

int PS4ABI sceSaveDataInitialize(void *param);
int PS4ABI sceSaveDataInitialize2(void *param);
int PS4ABI sceSaveDataInitialize3(void *param);
int PS4ABI sceSaveDataTerminate();
int PS4ABI sceSaveDataMount(const void *mount, void *result);
int PS4ABI sceSaveDataMount2(const void *mount, void *result);
int PS4ABI sceSaveDataMount5(const void *mount, void *result);
int PS4ABI sceSaveDataUmount(const void *mountPoint);
int PS4ABI sceSaveDataUmountWithBackup(const void *mountPoint);
int PS4ABI sceSaveDataGetMountInfo(const void *mountPoint, void *info);
int PS4ABI sceSaveDataDirNameSearch(const void *cond, void *result);
int PS4ABI sceSaveDataGetParam(const void *mountPoint, int paramType, void *buf,
                               uint64_t size, uint64_t *result);
int PS4ABI sceSaveDataSetParam(const void *mountPoint, int paramType,
                               const void *buf, uint64_t size);
int PS4ABI sceSaveDataSaveIcon(const void *mountPoint, const void *icon);
