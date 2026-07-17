#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

namespace krnl {
// Extra VFS-adjacent syscall handlers (access/stat-family, fcntl, dup, the
// scatter/gather read/write ops, poll/select stubs and the soft directory
// mutation stubs). Kept separate from sys_vfs.cpp to avoid touching that file.

int PS4ABI sys_access(const char *path, int mode);
int PS4ABI sys_faccessat(int fd, const char *path, int mode, int flag);

int PS4ABI sys_readlink(const char *path, char *buf, size_t bufsize);
int PS4ABI sys_readlinkat(int fd, const char *path, char *buf, size_t bufsize);

// lstat == stat (no symlinks). 40/190/493 all route here (fstatat ignores
// dirfd and treats path as absolute, so it shares the same body).
int PS4ABI sys_lstat(const char *path, void *stat);
int PS4ABI sys_fstatat(int fd, const char *path, void *stat, int flag);

int PS4ABI sys_fcntl(uint32_t fd, int cmd, int64_t arg);

int PS4ABI sys_dup(uint32_t fd);
int PS4ABI sys_dup2(uint32_t oldfd, uint32_t newfd);

int PS4ABI sys_fsync(uint32_t fd);
int PS4ABI sys_fdatasync(uint32_t fd);

int PS4ABI sys_getcwd(char *buf, size_t size);

int64_t PS4ABI sys_pread(uint32_t fd, void *buf, size_t nbytes, int64_t offset);
int64_t PS4ABI sys_pwrite(uint32_t fd, const void *buf, size_t nbytes,
                          int64_t offset);

int64_t PS4ABI sys_writev(uint32_t fd, const void *iov, int iovcnt);
int64_t PS4ABI sys_readv(uint32_t fd, const void *iov, int iovcnt);

int PS4ABI sys_poll(void *fds, uint32_t nfds, int timeout);
int PS4ABI sys_select(int nfds, void *readfds, void *writefds, void *exceptfds,
                      void *timeout);

int PS4ABI sys_openat(int fd, const char *path, uint32_t flags, uint32_t mode);

int PS4ABI sys_chdir(const char *path);
int PS4ABI sys_fchdir(uint32_t fd);

int PS4ABI sys_unlink(const char *path);
int PS4ABI sys_rmdir(const char *path);
int PS4ABI sys_mkdir(const char *path, uint32_t mode);
int PS4ABI sys_rename(const char *from, const char *to);

int64_t PS4ABI sys_getdirentries(uint32_t fd, void *buf, size_t nbytes,
                                 int64_t *basep);

int PS4ABI sys_closefrom(uint32_t lowfd);

// DELTA_QARBUF diagnostic: flag fds opened on a *.qar archive so sys_pread can
// report where the streamed texture data lands (a GPU-mapped 0x81xx region vs a
// low staging buffer). Set from sys_vfs.cpp at open time.
void markQarFd(uint32_t fd, bool v);
} // namespace krnl
