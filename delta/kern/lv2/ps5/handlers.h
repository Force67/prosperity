/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero syscall handlers whose PS5 kernel behaviour differs from the Orbis
 * handler of the same number, or that only exist on a PS5. Each was checked
 * against the 13.42 kernel's sysent; lv2/ps5/table.cc routes to them.
 */

#pragma once

#include <cstddef>
#include "base/arch.h"
#include "guest_abi.h"
#include "kern/lv2/dispatch.h"

namespace kern {

// Answers shared by many rows.
int PS4ABI Ps5Ok();
int PS4ABI Ps5Nosys();
int PS4ABI Ps5Eperm();
int PS4ABI Ps5Eopnotsupp();

// time
int PS4ABI Ps5ClockGettime(int clock_id, void* tp);
int PS4ABI Ps5ClockGetres(int clock_id, void* res);
int PS4ABI Ps5ClockGetcpuclockid2(i64 id, int which, int* clock_id);
int PS4ABI Ps5UtcToLocaltime(i64 time, i64* local, void* tz, int* dst);
int PS4ABI Ps5LocaltimeToUtc(i64 time, int type, i64* utc, void* tz, int* dst);

// memory
int PS4ABI Ps5Mtypeprotect(u8* addr, size_t len, int mtype, int prot);
i64 PS4ABI
Ps5Mmap2(void* addr, size_t len, u32 prot, u32 flags, u64 kind, u32 fd);
int PS4ABI Ps5VirtualQuery(const void* addr,
                           int flags,
                           void* info,
                           size_t info_size);
int PS4ABI Ps5Mname(u8* addr, size_t len, const char* name, void* unused);
int PS4ABI Ps5GetResidentFmemCount2(int pid, int which, u64* count, u64* peak);

// files and descriptors
int PS4ABI Ps5Chflagsat(int fd, const char* path, u64 flags, int atflag);
int PS4ABI Ps5Fchmodat(int fd, const char* path, u32 mode, int flag);
int PS4ABI Ps5Fchownat(int fd, const char* path, int uid, int gid, int flag);
int PS4ABI Ps5Futimesat(int fd, const char* path, const void* times);
int PS4ABI Ps5Futimens(int fd, const void* times);
int PS4ABI Ps5Utimensat(int fd, const char* path, const void* times, int flag);
int PS4ABI Ps5Pipe2(int* fds, int flags);
int PS4ABI Ps5Ppoll(void* fds, u32 nfds, const void* ts, const void* sigmask);
int PS4ABI Ps5Openintr(const char* path, u32 flags, u32 mode);

// sockets
int PS4ABI Ps5Accept(int fd, void* addr, u32* addrlen);
int PS4ABI Ps5Accept4(int fd, void* addr, u32* addrlen, int flags);
int PS4ABI Ps5Bindat(int dirfd, int fd, const void* addr, u32 addrlen);
int PS4ABI Ps5Connectat(int dirfd, int fd, const void* addr, u32 addrlen);
int PS4ABI Ps5Netabort(int fd, int flags);

// process and threads
int PS4ABI Ps5Wait6(int idtype,
                    i64 id,
                    int* status,
                    int options,
                    void* wrusage,
                    void* info);
int PS4ABI Ps5Procctl(int idtype, i64 id, int cmd, void* data);
int PS4ABI Ps5NumaGetaffinity(int which, i64 id, void* policy);
int PS4ABI Ps5NumaSetaffinity(int which, i64 id, const void* policy);
int PS4ABI Ps5Ioreq(int op, u32 value, u32* out);
int PS4ABI Ps5ThrSetName(i64 tid, const char* name);
int PS4ABI Ps5GetProcTypeInfo(void* info);
int PS4ABI Ps5DynlibDlsym(u32 handle, const char* name, void** sym);
int PS4ABI Ps5DynlibGetInfoEx(u32 handle, i32 flags, void* info);
int PS4ABI Ps5BudgetGetPtype(int pid);
int PS4ABI Ps5SetPhysFmemLimit(int pid, i64 limit);
int PS4ABI Ps5Workaround8849(u32 key);
int PS4ABI Ps5AmprCtrlDebug(int op, void* arg, size_t size);
int PS4ABI Ps5Streamwrite(int op, int fd, size_t size, const void* arg);

// APR. Resolve takes eight arguments; Lv2ContextHandler serves 701 with
// Ps5AprResolveContext, which reads the last two off the guest stack.
int PS4ABI Ps5AprResolve(const char* prefix,
                         const char* const* paths,
                         u64 num,
                         u32* ids,
                         u64* sizes,
                         u32* error_index,
                         int* results,
                         u32* count);
void PS4ABI Ps5AprResolveContext(u32 sid, GuestFrame* frame, u8* xsave);
int PS4ABI
Ps5AprSubmit(const u32* cmd, u64 ndw, u32 prio, u32 flag, u32* result, u32* id);
int PS4ABI Ps5AprStat(const u32* ids, u64 num, void* stat);
int PS4ABI Ps5AprWait(const u32* ids, u64 num);

}  // namespace kern
