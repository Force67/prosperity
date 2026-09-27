#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <guest_abi.h>
#include "base/arch.h"

namespace krnl {
// Extra VFS-adjacent syscall handlers (access/stat-family, fcntl, dup, the
// scatter/gather read/write ops, poll/select stubs and the soft directory
// mutation stubs). Kept separate from sys_vfs.cc to avoid touching that file.

int PS4ABI sys_access(const char *path, int mode);
int PS4ABI sys_faccessat(int fd, const char *path, int mode, int flag);

int PS4ABI sys_readlink(const char *path, char *buf, size_t bufsize);
int PS4ABI sys_readlinkat(int fd, const char *path, char *buf, size_t bufsize);

// lstat == stat (no symlinks). 40/190/493 all route here (fstatat ignores
// dirfd and treats path as absolute, so it shares the same body).
int PS4ABI sys_lstat(const char *path, void *stat);
int PS4ABI sys_fstatat(int fd, const char *path, void *stat, int flag);

int PS4ABI sys_fcntl(u32 fd, int cmd, i64 arg);

int PS4ABI sys_dup(u32 fd);
int PS4ABI sys_dup2(u32 oldfd, u32 newfd);

int PS4ABI sys_fsync(u32 fd);
int PS4ABI sys_fdatasync(u32 fd);

int PS4ABI sys_getcwd(char *buf, size_t size);

i64 PS4ABI sys_pread(u32 fd, void *buf, size_t nbytes, i64 offset);
i64 PS4ABI sys_preadv(u32 fd, const void *iov, int iovcnt, i64 offset);
i64 PS4ABI sys_pwritev(u32 fd, const void *iov, int iovcnt, i64 offset);
i64 PS4ABI sys_pwrite(u32 fd, const void *buf, size_t nbytes,
                          i64 offset);

i64 PS4ABI sys_writev(u32 fd, const void *iov, int iovcnt);
i64 PS4ABI sys_readv(u32 fd, const void *iov, int iovcnt);

int PS4ABI sys_poll(void *fds, u32 nfds, int timeout);
int PS4ABI sys_select(int nfds, void *readfds, void *writefds, void *exceptfds,
                      void *timeout);

int PS4ABI sys_openat(int fd, const char *path, u32 flags, u32 mode);

int PS4ABI sys_chdir(const char *path);
int PS4ABI sys_fchdir(u32 fd);

int PS4ABI sys_unlink(const char *path);
int PS4ABI sys_unlinkat(int fd, const char *path, int flag);
int PS4ABI sys_rmdir(const char *path);
int PS4ABI sys_mkdir(const char *path, u32 mode);
int PS4ABI sys_mkdirat(int fd, const char *path, u32 mode);
int PS4ABI sys_rename(const char *from, const char *to);
int PS4ABI sys_renameat(int fd_old, const char *old, int fd_new,
                        const char *new_);

i64 PS4ABI sys_getdirentries(u32 fd, void *buf, size_t nbytes,
                                 i64 *basep);

int PS4ABI sys_closefrom(u32 lowfd);

// DELTA_QARBUF diagnostic: flag fds opened on a *.qar archive so sys_pread can
// report where the streamed texture data lands (a GPU-mapped 0x81xx region vs a
// low staging buffer). Set from sys_vfs.cc at open time.
void markQarFd(u32 fd, bool v);
} // namespace krnl
