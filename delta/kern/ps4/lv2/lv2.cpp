
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_map>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

#include "error_table.h"
#include "sys_debug.h"
#include "sys_dynlib.h"
#include "sys_event.h"
#include "sys_event_flag.h"
#include "sys_generic.h"
#include "sys_info.h"
#include "sys_mem.h"
#include "sys_semaphore.h"
#include "sys_thread.h"
#include "sys_time.h"
#include "sys_vfs.h"
#include "sys_net.h"
#include "sys_procid.h"
#include "sys_mem_ext.h"
#include "sys_thread_ext.h"
#include "sys_time_ext.h"
#include "sys_ksem.h"
#include "sys_vfs_ext.h"
#include "sys_sce_misc.h"

#include "kern/module.h"
#include "kern/proc.h"

namespace krnl {
const char *syscall_getname(uint32_t idx);

int sys_write(uint32_t fd, const void *buf, size_t nbytes);
int sys_sigprocmask(int, const int *, int *);
int sys_sigaction(int sig, const void *act, void *oact);
int sys_regmgr_call(uint32_t op, uint32_t id, void *result, void *value,
                    uint64_t type);

int PS4ABI sys_ipmimgr_call(uint32_t op, uint32_t kid, void *out, void *in,
                            uint64_t insize, uint64_t arg6);

int sys_randomized_path(const char *set_path, char *out, size_t *out_len);
int sys_workaround8849();
int sys_blockpool_open();
int sys_dynlib_do_copy_relocations();

int sys_namedobj_create(const char *name, void *arg2, uint32_t arg3);
int sys_namedobj_delete();

int sys_budget_get_ptype();

int sys_getpid();
int sys_exit();
int sys_rfork();
int sys_execve();

int PS4ABI sys_sysarch(int num, void *args);

moduleInfo *called_in(void *addr) {
  uintptr_t addrsafe = (uintptr_t)addr;

  for (auto &mod : proc::getActive()->getModuleList()) {
    auto &info = mod->getInfo();

    if (addrsafe <= (uintptr_t)(info.base + info.codeSize) &&
        (addrsafe >= (uintptr_t)info.base)) {
      std::printf("%p called in %s\n", addr, info.name.c_str());
      return &info;
    }
  }
  return nullptr;
}

void dumpSyscallHist();
extern const bool g_scHist;

static int PS4ABI null_handler() {
#ifdef _MSC_VER
  void *ret = _ReturnAddress();
#else
  void *ret = __builtin_return_address(0);
#endif
  called_in(ret);

  static std::atomic<uint64_t> nulls{0};
  const uint64_t n = ++nulls;
  if (n == 1 || (g_scHist && n % 400 == 0)) {
    std::printf(">>>>>>>>>>>>> NULL HANDLER called %llu times\n",
                (unsigned long long)n);
    if (g_scHist)
      dumpSyscallHist();
  }
  return 0;
}

static int PS4ABI null_handler_notable() {
#ifdef _MSC_VER
  void *ret = _ReturnAddress();
#else
  void *ret = __builtin_return_address(0);
#endif
  called_in(ret);

  std::printf(">>>>>>>>>>>>> NULL HANDLER NULLTABLE CALLED BY %p\n", ret);
  return 0;
}

struct syscall_Reg {
  uint32_t id;
  const void *ptr;
};

static const syscall_Reg syscall_dpt[] = {
    {0, (void *)&null_handler}, // sys_nosys
    {1, (void *)&sys_exit},
    {2, (void *)&null_handler}, // sys_fork
    {3, (void *)&sys_read},
    {4, (void *)&sys_write},
    {5, (void *)&sys_open},
    {6, (void *)&sys_close},
    {7, (void *)&sys_wait4},  // sys_wait4
    {8, (void *)&null_handler},  // sys_creat
    {9, (void *)&null_handler},  // sys_link
    {10, (void *)&sys_unlink}, // sys_unlink
    {11, (void *)&null_handler}, // sys_execv
    {12, (void *)&sys_chdir}, // sys_chdir
    {13, (void *)&sys_fchdir}, // sys_fchdir
    {14, (void *)&null_handler}, // sys_mkd
    {15, (void *)&null_handler}, // sys_chmod
    {16, (void *)&null_handler}, // sys_chown
    {17, (void *)&sys_obreak}, // sys_obreak
    {18, (void *)&null_handler}, // sys_getfsstat
    {19, (void *)&sys_lseek},
    {20, (void *)&sys_getpid},
    {21, (void *)&null_handler}, // sys_mount
    {22, (void *)&null_handler}, // sys_unmount
    {23, (void *)&sys_setuid}, // sys_setuid
    {24, (void *)&sys_getuid}, // sys_getuid
    {25, (void *)&sys_geteuid}, // sys_geteuid
    {26, (void *)&null_handler}, // sys_ptrace
    {27, (void *)&sys_recvmsg},
    {28, (void *)&null_handler}, // sys_sendmsg
    {29, (void *)&sys_recvfrom},
    {30, (void *)&null_handler}, // sys_accept
    {31, (void *)&null_handler}, // sys_getpeername
    {32, (void *)&sys_getsockname},
    {33, (void *)&sys_access}, // sys_access
    {34, (void *)&null_handler}, // sys_chflags
    {35, (void *)&null_handler}, // sys_fchflags
    {36, (void *)&sys_sync}, // sys_sync
    {37, (void *)&sys_kill}, // sys_kill
    {38, (void *)&sys_stat},
    {39, (void *)&sys_getppid}, // sys_getppid
    {40, (void *)&sys_lstat}, // sys_lstat
    {41, (void *)&sys_dup}, // sys_dup
    {42, (void *)&null_handler}, // sys_pipe
    {43, (void *)&sys_getegid}, // sys_getegid
    {44, (void *)&null_handler}, // sys_profil
    {45, (void *)&null_handler}, // sys_ktrace
    {46, (void *)&sys_sigaction},
    {47, (void *)&sys_getgid}, // sys_getgid
    {48, (void *)&sys_sigprocmask},
    {49, (void *)&sys_getlogin}, // sys_getlogin
    {50, (void *)&sys_setlogin}, // sys_setlogin
    {51, (void *)&null_handler}, // sys_acct
    {52, (void *)&sys_sigpending}, // sys_sigpending
    {53, (void *)&sys_sigaltstack}, // sys_sigaltstack
    {54, (void *)&sys_ioctl},
    {55, (void *)&null_handler}, // sys_reboot
    {56, (void *)&null_handler}, // sys_revoke
    {57, (void *)&null_handler}, // sys_symlink
    {58, (void *)&sys_readlink}, // sys_readlink
    {59, (void *)&sys_execve},
    {60, (void *)&sys_umask}, // sys_umask
    {61, (void *)&null_handler}, // sys_chroot
    {62, (void *)&sys_fstat},
    {63, (void *)&null_handler}, // sys_getkerninfo
    {64, (void *)&sys_getpagesize}, // sys_getpagesize
    {65, (void *)&sys_msync}, // sys_msync
    {66, (void *)&null_handler}, // sys_vfork
    {67, (void *)&null_handler}, // sys_vread
    {68, (void *)&null_handler}, // sys_vwrite
    {69, (void *)&sys_sbrk}, // sys_sbrk
    {70, (void *)&null_handler}, // sys_sstk
    {71, (void *)&sys_mmap},
    {72, (void *)&null_handler},  // sys_ovadvise
    {73, (void *)&sys_munmap},  // sys_munmap
    {74, (void *)&sys_mprotect},
    {75, (void *)&sys_madvise},  // sys_madvise
    {76, (void *)&null_handler},  // sys_vhangup
    {77, (void *)&null_handler},  // sys_vlimit
    {78, (void *)&sys_mincore},  // sys_mincore
    {79, (void *)&sys_getgroups},  // sys_getgroups
    {80, (void *)&sys_setgroups},  // sys_setgroups
    {81, (void *)&sys_getpgrp},  // sys_getpgrp
    {82, (void *)&sys_setpgid},  // sys_setpgid
    {83, (void *)&sys_setitimer},  // sys_setitimer
    {84, (void *)&null_handler},  // sys_wait
    {85, (void *)&null_handler},  // sys_swapon
    {86, (void *)&sys_getitimer},  // sys_getitimer
    {87, (void *)&sys_gethostname},  // sys_gethostname
    {88, (void *)&sys_sethostname},  // sys_sethostname
    {89, (void *)&sys_getdtablesize},  // sys_getdtablesize
    {90, (void *)&sys_dup2},  // sys_dup2
    {91, (void *)&null_handler},  // sys_getdopt
    {92, (void *)&sys_fcntl},  // sys_fcntl
    {93, (void *)&sys_select},  // sys_select
    {94, (void *)&null_handler},  // sys_setdopt
    {95, (void *)&sys_fsync},  // sys_fsync
    {96, (void *)&sys_setpriority},  // sys_setpriority
    {97, (void *)&sys_socket},
    {98, (void *)&sys_connect},
    {99, (void *)&sys_netcontrol},
    {100, (void *)&sys_getpriority}, // sys_getpriority
    {101, (void *)&null_handler}, // sys_netabort
    {102, (void *)&null_handler}, // sys_netgetsockinfo
    {103, (void *)&null_handler}, // sys_sigreturn
    {104, (void *)&sys_bind},
    {105, (void *)&sys_setsockopt}, // sys_setsockopt
    {106, (void *)&null_handler}, // sys_listen
    {107, (void *)&null_handler}, // sys_vtimes
    {108, (void *)&null_handler}, // sys_sigvec
    {109, (void *)&null_handler}, // sys_sigblock
    {110, (void *)&null_handler}, // sys_sigsetmask
    {111, (void *)&sys_sigsuspend}, // sys_sigsuspend
    {112, (void *)&null_handler}, // sys_sigstack
    {113, (void *)&sys_socketex},
    {114, (void *)&sys_socketclose},
    {115, (void *)&null_handler}, // sys_vtrace
    {116, (void *)&sys_gettimeofday}, // sys_gettimeofday
    {117, (void *)&sys_getrusage}, // sys_getrusage
    {118, (void *)&sys_getsockopt}, // sys_getsockopt
    {119, (void *)&null_handler}, // sys_resuba
    {120, (void *)&sys_readv}, // sys_readv
    {121, (void *)&sys_writev}, // sys_writev
    {122, (void *)&sys_settimeofday}, // sys_settimeofday
    {123, (void *)&null_handler}, // sys_fchown
    {124, (void *)&null_handler}, // sys_fchmod
    {125, (void *)&null_handler}, // sys_netgetiflist
    {126, (void *)&null_handler}, // sys_setreuid
    {127, (void *)&null_handler}, // sys_setregid
    {128, (void *)&sys_rename}, // sys_rename
    {129, (void *)&null_handler}, // sys_truncate
    {130, (void *)&null_handler}, // sys_ftruncate
    {131, (void *)&sys_flock}, // sys_flock
    {132, (void *)&null_handler}, // sys_mkfifo
    {133, (void *)&sys_sendto},
    {134, (void *)&null_handler}, // sys_shutdown
    {135, (void *)&null_handler}, // sys_socketpair
    {136, (void *)&sys_mkdir}, // sys_mkdir
    {137, (void *)&sys_rmdir}, // sys_rmdir
    {138, (void *)&sys_utimes}, // sys_utimes
    {139, (void *)&null_handler}, // sys_sigreturn
    {140, (void *)&null_handler}, // sys_adjtime
    {141, (void *)&sys_kqueueex},
    {142, (void *)&null_handler}, // sys_gethostid
    {143, (void *)&null_handler}, // sys_sethostid
    {144, (void *)&sys_getrlimit}, // sys_getrlimit
    {145, (void *)&sys_setrlimit}, // sys_setrlimit
    {146, (void *)&null_handler}, // sys_killpg
    {147, (void *)&sys_setsid}, // sys_setsid
    {148, (void *)&null_handler}, // sys_quotactl
    {149, (void *)&null_handler}, // sys_quota
    {150, (void *)&sys_getsockname},
    {151, (void *)&null_handler}, // sys_sem_lock
    {152, (void *)&null_handler}, // sys_sem_wakeup
    {153, (void *)&null_handler}, // sys_asyncdaemon
    {154, (void *)&null_handler}, // sys_nlm_syscall
    {155, (void *)&null_handler}, // sys_nfssvc
    {156, (void *)&sys_getdirentries}, // sys_getdirentries
    {157, (void *)&null_handler}, // sys_statfs
    {158, (void *)&null_handler}, // sys_fstatfs
    {160, (void *)&null_handler}, // sys_lgetfh
    {161, (void *)&null_handler}, // sys_getfh
    {162, (void *)&null_handler}, // sys_getdomainname
    {163, (void *)&null_handler}, // sys_setdomainname
    {164, (void *)&sys_uname}, // sys_uname
    {165, (void *)&sys_sysarch},
    {166, (void *)&sys_rtprio}, // sys_rtprio
    {169, (void *)&null_handler}, // sys_semsys
    {170, (void *)&null_handler}, // sys_msgsys
    {171, (void *)&null_handler}, // sys_shmsys
    {173, (void *)&sys_pread}, // sys_pread
    {174, (void *)&sys_pwrite}, // sys_pwrite
    {175, (void *)&null_handler}, // sys_setfib
    {176, (void *)&null_handler}, // sys_ntp_adjtime
    {177, (void *)&null_handler}, // sys_sfork
    {178, (void *)&null_handler}, // sys_getdescriptor
    {179, (void *)&null_handler}, // sys_setdescriptor
    {181, (void *)&sys_setgid}, // sys_setgid
    {182, (void *)&sys_setegid}, // sys_setegid
    {183, (void *)&sys_seteuid}, // sys_seteuid
    {184, (void *)&null_handler}, // sys_lfs_bmapv
    {185, (void *)&null_handler}, // sys_lfs_markv
    {186, (void *)&null_handler}, // sys_lfs_segclean
    {187, (void *)&null_handler}, // sys_lfs_segwait
    {188, (void *)&sys_stat},
    {189, (void *)&sys_fstat},
    {190, (void *)&sys_lstat}, // sys_lstat
    {191, (void *)&sys_pathconf}, // sys_pathconf
    {192, (void *)&sys_fpathconf}, // sys_fpathconf
    {194, (void *)&sys_getrlimit}, // sys_getrlimit
    {195, (void *)&sys_setrlimit}, // sys_setrlimit
    {196, (void *)&sys_getdirentries}, // sys_getdirentries
    {197, (void *)&sys_mmap},
    {198, (void *)&null_handler}, // sys_nosys
    {199, (void *)&sys_lseek},
    {200, (void *)&null_handler}, // sys_truncate
    {201, (void *)&null_handler}, // sys_ftruncate
    {202, (void *)&sys_sysctl},
    {203, (void *)&sys_mlock}, // sys_mlock
    {204, (void *)&sys_munlock}, // sys_munlock
    {205, (void *)&null_handler}, // sys_undelete
    {206, (void *)&sys_futimes}, // sys_futimes
    {207, (void *)&sys_getpgid}, // sys_getpgid
    {208, (void *)&null_handler}, // sys_newreboot
    {209, (void *)&sys_poll}, // sys_poll
    {220, (void *)&null_handler}, // sys_semctl
    {221, (void *)&null_handler}, // sys_semget
    {222, (void *)&null_handler}, // sys_semop
    {223, (void *)&null_handler}, // sys_semconfig
    {224, (void *)&null_handler}, // sys_msgctl
    {225, (void *)&null_handler}, // sys_msgget
    {226, (void *)&null_handler}, // sys_msgsnd
    {227, (void *)&null_handler}, // sys_msgrcv
    {228, (void *)&null_handler}, // sys_shmat
    {229, (void *)&null_handler}, // sys_shmctl
    {230, (void *)&null_handler}, // sys_shmdt
    {231, (void *)&null_handler}, // sys_shmget
    {232, (void *)&sys_clock_gettime},
    {233, (void *)&sys_clock_settime}, // sys_clock_settime
    {234, (void *)&sys_clock_getres}, // sys_clock_getres
    {235, (void *)&sys_ktimer_create}, // sys_ktimer_create
    {236, (void *)&sys_ktimer_delete}, // sys_ktimer_delete
    {237, (void *)&sys_ktimer_settime}, // sys_ktimer_settime
    {238, (void *)&sys_ktimer_gettime}, // sys_ktimer_gettime
    {239, (void *)&sys_ktimer_getoverrun}, // sys_ktimer_getoverrun
    {240, (void *)&sys_nanosleep},
    {241, (void *)&sys_ffclock_getcounter}, // sys_ffclock_getcounter
    {242, (void *)&sys_ffclock_setestimate}, // sys_ffclock_setestimate
    {243, (void *)&sys_ffclock_getestimate}, // sys_ffclock_getestimate
    {247, (void *)&sys_clock_getcpuclockid2}, // sys_clock_getcpuclockid2
    {248, (void *)&sys_ntp_gettime}, // sys_ntp_gettime
    {250, (void *)&sys_minherit}, // sys_minherit
    {251, (void *)&sys_rfork},
    {252, (void *)&null_handler}, // sys_openbsd_poll
    {253, (void *)&sys_issetugid}, // sys_issetugid
    {254, (void *)&null_handler}, // sys_lchown
    {255, (void *)&sys_aio_unsupported}, // sys_aio_read
    {256, (void *)&sys_aio_unsupported}, // sys_aio_write
    {257, (void *)&sys_aio_unsupported}, // sys_lio_listio
    {272, (void *)&sys_getdents},
    {274, (void *)&null_handler}, // sys_lchmod
    {275, (void *)&null_handler}, // sys_lchown
    {276, (void *)&null_handler}, // sys_lutimes
    {277, (void *)&sys_msync}, // sys_msync
    {278, (void *)&null_handler}, // sys_nstat
    {279, (void *)&null_handler}, // sys_nfstat
    {280, (void *)&null_handler}, // sys_nlstat
    {289, (void *)&sys_preadv}, // sys_preadv
    {290, (void *)&sys_pwritev}, // sys_pwritev
    {297, (void *)&null_handler}, // sys_fhstatfs
    {298, (void *)&null_handler}, // sys_fhopen
    {299, (void *)&null_handler}, // sys_fhstat
    {300, (void *)&null_handler}, // sys_modnext
    {301, (void *)&null_handler}, // sys_modstat
    {302, (void *)&null_handler}, // sys_modfnext
    {303, (void *)&null_handler}, // sys_modfind
    {304, (void *)&null_handler}, // sys_kldload
    {305, (void *)&null_handler}, // sys_kldunload
    {306, (void *)&null_handler}, // sys_kldfind
    {307, (void *)&null_handler}, // sys_kldnext
    {308, (void *)&null_handler}, // sys_kldstat
    {309, (void *)&null_handler}, // sys_kldfirstmod
    {310, (void *)&sys_getsid}, // sys_getsid
    {311, (void *)&sys_setresuid}, // sys_setresuid
    {312, (void *)&sys_setresgid}, // sys_setresgid
    {313, (void *)&null_handler}, // sys_signasleep
    {314, (void *)&sys_aio_unsupported}, // sys_aio_return
    {315, (void *)&sys_aio_unsupported}, // sys_aio_suspend
    {316, (void *)&sys_aio_unsupported}, // sys_aio_cancel
    {317, (void *)&sys_aio_unsupported}, // sys_aio_error
    {318, (void *)&sys_aio_unsupported}, // sys_aio_read
    {319, (void *)&sys_aio_unsupported}, // sys_aio_write
    {320, (void *)&sys_aio_unsupported}, // sys_lio_listio
    {321, (void *)&sys_yield}, // sys_yield
    {322, (void *)&sys_thr_sleep}, // sys_thr_sleep
    {323, (void *)&sys_thr_wakeup}, // sys_thr_wakeup
    {324, (void *)&sys_mlockall}, // sys_mlockall
    {325, (void *)&sys_munlockall}, // sys_munlockall
    {326, (void *)&sys_getcwd}, // sys_getcwd
    {327, (void *)&sys_sched_setparam}, // sys_sched_setparam
    {328, (void *)&sys_sched_getparam}, // sys_sched_getparam
    {329, (void *)&sys_sched_setscheduler}, // sys_sched_setscheduler
    {330, (void *)&sys_sched_getscheduler}, // sys_sched_getscheduler
    {331, (void *)&sys_sched_yield}, // sys_sched_yield
    {332, (void *)&sys_sched_get_priority_max}, // sys_sched_get_priority_max
    {333, (void *)&sys_sched_get_priority_min}, // sys_sched_get_priority_min
    {334, (void *)&sys_sched_rr_get_interval}, // sys_sched_rr_get_interval
    {335, (void *)&null_handler}, // sys_utrace
    {336, (void *)&null_handler}, // sys_sendfile
    {337, (void *)&null_handler}, // sys_kldsym
    {338, (void *)&null_handler}, // sys_jail
    {339, (void *)&null_handler}, // sys_nnpfs_syscall
    {340, (void *)&sys_sigprocmask},
    {341, (void *)&sys_sigsuspend}, // sys_sigsuspend
    {342, (void *)&sys_sigaction},
    {343, (void *)&sys_sigpending}, // sys_sigpending
    {344, (void *)&null_handler}, // sys_sigreturn
    {345, (void *)&sys_sigtimedwait}, // sys_sigtimedwait
    {346, (void *)&sys_sigwaitinfo}, // sys_sigwaitinfo
    {347, (void *)&null_handler}, // sys_acl_get_file
    {348, (void *)&null_handler}, // sys_acl_set_file
    {349, (void *)&null_handler}, // sys_acl_get_fd
    {350, (void *)&null_handler}, // sys_acl_set_fd
    {351, (void *)&null_handler}, // sys_acl_delete_file
    {352, (void *)&null_handler}, // sys_acl_delete_fd
    {353, (void *)&null_handler}, // sys_acl_aclcheck_file
    {354, (void *)&null_handler}, // sys_acl_aclcheck_fd
    {355, (void *)&null_handler}, // sys_extattrctl
    {356, (void *)&null_handler}, // sys_extattr_set_file
    {357, (void *)&null_handler}, // sys_extattr_get_file
    {358, (void *)&null_handler}, // sys_extattr_delete_file
    {359, (void *)&sys_aio_unsupported}, // sys_aio_waitcomplete
    {360, (void *)&sys_getresuid}, // sys_getresuid
    {361, (void *)&sys_getresgid}, // sys_getresgid
    {362, (void *)&sys_kqueue},
    {363, (void *)&sys_kevent},
    {364, (void *)&null_handler}, // sys_cap_get_proc
    {365, (void *)&null_handler}, // sys_cap_set_proc
    {366, (void *)&null_handler}, // sys_cap_get_fd
    {367, (void *)&null_handler}, // sys_cap_get_file
    {368, (void *)&null_handler}, // sys_cap_set_fd
    {369, (void *)&null_handler}, // sys_cap_set_file
    {371, (void *)&null_handler}, // sys_extattr_set_fd
    {372, (void *)&null_handler}, // sys_extattr_get_fd
    {373, (void *)&null_handler}, // sys_extattr_delete_fd
    {374, (void *)&null_handler}, // sys_setugid
    {375, (void *)&null_handler}, // sys_nfsclnt
    {376, (void *)&null_handler}, // sys_eaccess
    {377, (void *)&null_handler}, // sys_afs3_syscall
    {378, (void *)&null_handler}, // sys_nmount
    {379, (void *)&null_handler}, // sys_mtypeprotect
    {380, (void *)&null_handler}, // sys_kse_wakeup
    {381, (void *)&null_handler}, // sys_kse_create
    {382, (void *)&null_handler}, // sys_kse_thr_interrupt
    {383, (void *)&null_handler}, // sys_kse_release
    {384, (void *)&null_handler}, // sys_mac_get_proc
    {385, (void *)&null_handler}, // sys_mac_set_proc
    {386, (void *)&null_handler}, // sys_mac_get_fd
    {387, (void *)&null_handler}, // sys_mac_get_file
    {388, (void *)&null_handler}, // sys_mac_set_fd
    {389, (void *)&null_handler}, // sys_mac_set_file
    {390, (void *)&null_handler}, // sys_kenv
    {391, (void *)&null_handler}, // sys_lchflags
    {392, (void *)&null_handler}, // sys_uuidgen
    {393, (void *)&null_handler}, // sys_sendfile
    {394, (void *)&null_handler}, // sys_mac_syscall
    {395, (void *)&null_handler}, // sys_getfsstat
    {396, (void *)&null_handler}, // sys_statfs
    {397, (void *)&null_handler}, // sys_fstatfs
    {398, (void *)&null_handler}, // sys_fhstatfs
    {400, (void *)&sys_ksem_close}, // sys_ksem_close
    {401, (void *)&sys_ksem_post}, // sys_ksem_post
    {402, (void *)&sys_ksem_wait}, // sys_ksem_wait
    {403, (void *)&sys_ksem_trywait}, // sys_ksem_trywait
    {404, (void *)&sys_ksem_init}, // sys_ksem_init
    {405, (void *)&sys_ksem_open}, // sys_ksem_open
    {406, (void *)&sys_ksem_unlink}, // sys_ksem_unlink
    {407, (void *)&sys_ksem_getvalue}, // sys_ksem_getvalue
    {408, (void *)&sys_ksem_destroy}, // sys_ksem_destroy
    {409, (void *)&null_handler}, // sys_mac_get_pid
    {410, (void *)&null_handler}, // sys_mac_get_link
    {411, (void *)&null_handler}, // sys_mac_set_link
    {412, (void *)&null_handler}, // sys_extattr_set_link
    {413, (void *)&null_handler}, // sys_extattr_get_link
    {414, (void *)&null_handler}, // sys_extattr_delete_link
    {415, (void *)&null_handler}, // sys_mac_execve
    {416, (void *)&sys_sigaction},
    {417, (void *)&null_handler}, // sys_sigreturn
    {418, (void *)&null_handler}, // sys_xstat
    {419, (void *)&null_handler}, // sys_xfstat
    {420, (void *)&null_handler}, // sys_xlstat
    {421, (void *)&null_handler}, // sys_getcontext
    {422, (void *)&null_handler}, // sys_setcontext
    {423, (void *)&null_handler}, // sys_swapcontext
    {424, (void *)&null_handler}, // sys_swapoff
    {425, (void *)&null_handler}, // sys_acl_get_link
    {426, (void *)&null_handler}, // sys_acl_set_link
    {427, (void *)&null_handler}, // sys_acl_delete_link
    {428, (void *)&null_handler}, // sys_acl_aclcheck_link
    {429, (void *)&sys_sigwait}, // sys_sigwait
    {430, (void *)&null_handler}, // sys_thr_create
    {431, (void *)&sys_thr_exit}, // sys_thr_exit
    {432, (void *)&sys_thr_self},
    {433, (void *)&sys_thr_kill}, // sys_thr_kill
    {436, (void *)&null_handler}, // sys_jail_attach
    {437, (void *)&null_handler}, // sys_extattr_list_fd
    {438, (void *)&null_handler}, // sys_extattr_list_file
    {439, (void *)&null_handler}, // sys_extattr_list_link
    {440, (void *)&null_handler}, // sys_kse_switchin
    {441, (void *)&sys_ksem_timedwait}, // sys_ksem_timedwait
    {442, (void *)&sys_thr_suspend}, // sys_thr_suspend
    {443, (void *)&sys_thr_wake}, // sys_thr_wake
    {444, (void *)&null_handler}, // sys_kldunloadf
    {445, (void *)&null_handler}, // sys_audit
    {446, (void *)&null_handler}, // sys_auditon
    {447, (void *)&null_handler}, // sys_getauid
    {448, (void *)&null_handler}, // sys_setauid
    {449, (void *)&null_handler}, // sys_getaudit
    {450, (void *)&null_handler}, // sys_setaudit
    {451, (void *)&null_handler}, // sys_getaudit_addr
    {452, (void *)&null_handler}, // sys_setaudit_addr
    {453, (void *)&null_handler}, // sys_auditctl
    {454, (void *)&sys_umtx_op},
    {455, (void *)&sys_thr_new},
    {456, (void *)&sys_sigqueue}, // sys_sigqueue
    {457, (void *)&null_handler}, // sys_kmq_open
    {458, (void *)&null_handler}, // sys_kmq_setattr
    {459, (void *)&null_handler}, // sys_kmq_timedreceive
    {460, (void *)&null_handler}, // sys_kmq_timedsend
    {461, (void *)&null_handler}, // sys_kmq_tify
    {462, (void *)&null_handler}, // sys_kmq_unlink
    {463, (void *)&sys_abort2}, // sys_abort2
    {464, (void *)&sys_thr_set_name}, // sys_thr_set_name
    {465, (void *)&sys_aio_unsupported}, // sys_aio_fsync
    {466, (void *)&sys_rtprio_thread},
    {469, (void *)&null_handler}, // sys_getpath_fromfd
    {470, (void *)&null_handler}, // sys_getpath_fromaddr
    {471, (void *)&null_handler}, // sys_sctp_peeloff
    {472, (void *)&null_handler}, // sys_sctp_generic_sendmsg
    {473, (void *)&null_handler}, // sys_sctp_generic_sendmsg_iov
    {474, (void *)&null_handler}, // sys_sctp_generic_recvmsg
    {475, (void *)&sys_pread}, // sys_pread
    {476, (void *)&sys_pwrite}, // sys_pwrite
    {477, (void *)&sys_mmap},
    {478, (void *)&sys_lseek}, // modern lseek (the game uses 478, not the old 19)
    {479, (void *)&null_handler}, // sys_truncate
    {480, (void *)&sys_ftruncate},
    {481, (void *)&sys_thr_kill2}, // sys_thr_kill2
    {482, (void *)&sys_shm_open},
    {483, (void *)&sys_shm_unlink},
    {484, (void *)&sys_cpuset}, // sys_cpuset
    {485, (void *)&sys_cpuset_setid}, // sys_cpuset_setid
    {486, (void *)&sys_cpuset_getid}, // sys_cpuset_getid
    {487, (void *)&sys_cpuset_getaffinity},
    {488, (void *)&sys_cpuset_setaffinity}, // sys_cpuset_setaffinity
    {489, (void *)&sys_faccessat}, // sys_faccessat
    {490, (void *)&null_handler}, // sys_fchmodat
    {491, (void *)&null_handler}, // sys_fchownat
    {492, (void *)&null_handler}, // sys_fexecve
    {493, (void *)&sys_fstatat}, // sys_fstatat
    {494, (void *)&null_handler}, // sys_futimesat
    {495, (void *)&null_handler}, // sys_linkat
    {496, (void *)&null_handler}, // sys_mkdirat
    {497, (void *)&null_handler}, // sys_mkfifoat
    {498, (void *)&null_handler}, // sys_mkdat
    {499, (void *)&sys_openat}, // sys_openat
    {500, (void *)&sys_readlinkat}, // sys_readlinkat
    {501, (void *)&null_handler}, // sys_renameat
    {502, (void *)&null_handler}, // sys_symlinkat
    {503, (void *)&null_handler}, // sys_unlinkat
    {504, (void *)&null_handler}, // sys_posix_openpt
    {505, (void *)&null_handler}, // sys_gssd_syscall
    {506, (void *)&null_handler}, // sys_jail_get
    {507, (void *)&null_handler}, // sys_jail_set
    {508, (void *)&null_handler}, // sys_jail_remove
    {509, (void *)&sys_closefrom}, // sys_closefrom
    {510, (void *)&null_handler}, // sys_semctl
    {511, (void *)&null_handler}, // sys_msgctl
    {512, (void *)&null_handler}, // sys_shmctl
    {513, (void *)&sys_lpathconf}, // sys_lpathconf
    {514, (void *)&null_handler}, // sys_cap_new
    {515, (void *)&null_handler}, // sys_cap_rights_get
    {516, (void *)&null_handler}, // sys_cap_enter
    {517, (void *)&null_handler}, // sys_cap_getmode
    {518, (void *)&null_handler}, // sys_pdfork
    {519, (void *)&null_handler}, // sys_pdkill
    {520, (void *)&null_handler}, // sys_pdgetpid
    {521, (void *)&null_handler}, // sys_pdwait4
    {522, (void *)&null_handler}, // sys_pselect
    {523, (void *)&null_handler}, // sys_getloginclass
    {524, (void *)&null_handler}, // sys_setloginclass
    {525, (void *)&null_handler}, // sys_rctl_get_racct
    {526, (void *)&null_handler}, // sys_rctl_get_rules
    {527, (void *)&null_handler}, // sys_rctl_get_limits
    {528, (void *)&null_handler}, // sys_rctl_add_rule
    {529, (void *)&null_handler}, // sys_rctl_remove_rule
    {530, (void *)&sys_posix_fallocate}, // sys_posix_fallocate
    {531, (void *)&sys_posix_fadvise}, // sys_posix_fadvise
    {532, (void *)&sys_regmgr_call},
    {533, (void *)&sys_jitshm_create}, // sys_jitshm_create
    {534, (void *)&sys_jitshm_alias}, // sys_jitshm_alias
    {535, (void *)&sys_dl_get_list}, // sys_dl_get_list
    {536, (void *)&sys_dl_get_info}, // sys_dl_get_info
    {537, (void *)&sys_dl_notify_event}, // sys_dl_notify_event
    {538, (void *)&sys_evf_create},
    {539, (void *)&sys_evf_delete},
    {540, (void *)&sys_evf_open},
    {541, (void *)&sys_evf_close},
    {542, (void *)&sys_evf_wait},
    {543, (void *)&sys_evf_trywait},
    {544, (void *)&sys_evf_set},
    {545, (void *)&sys_evf_clear},
    {546, (void *)&sys_evf_cancel},
    {547, (void *)&sys_query_memory_protection}, // sys_query_memory_protection
    {548, (void *)&sys_batch_map}, // sys_batch_map
    {549, (void *)&sys_osem_create},
    {550, (void *)&sys_osem_delete},
    {551, (void *)&sys_osem_open},
    {552, (void *)&sys_osem_close},
    {553, (void *)&sys_osem_wait},
    {554, (void *)&sys_osem_trywait},
    {555, (void *)&sys_osem_post},
    {556, (void *)&sys_osem_cancel},
    {557, (void *)&sys_namedobj_create},
    {558, (void *)&sys_namedobj_delete},
    {559, (void *)&sys_set_vm_container}, // sys_set_vm_container
    {560, (void *)&sys_debug_init}, // sys_debug_init
    {561, (void *)&sys_suspend_process}, // sys_suspend_process
    {562, (void *)&sys_resume_process}, // sys_resume_process
    {563, (void *)&sys_opmc_enable}, // sys_opmc_enable
    {564, (void *)&sys_opmc_disable}, // sys_opmc_disable
    {565, (void *)&sys_opmc_set_ctl}, // sys_opmc_set_ctl
    {566, (void *)&sys_opmc_set_ctr}, // sys_opmc_set_ctr
    {567, (void *)&sys_opmc_get_ctr}, // sys_opmc_get_ctr
    {568, (void *)&sys_budget_create}, // sys_budget_create
    {569, (void *)&sys_budget_delete}, // sys_budget_delete
    {570, (void *)&sys_budget_get}, // sys_budget_get
    {571, (void *)&sys_budget_set}, // sys_budget_set
    {572, (void *)&sys_virtual_query}, // sys_virtual_query
    {573, (void *)&sys_mdbg_call}, // sys_mdbg_call
    {574, (void *)&sys_sblock_create}, // sys_sblock_create
    {575, (void *)&sys_sblock_delete}, // sys_sblock_delete
    {576, (void *)&sys_sblock_enter}, // sys_sblock_enter
    {577, (void *)&sys_sblock_exit}, // sys_sblock_exit
    {578, (void *)&sys_sblock_xenter}, // sys_sblock_xenter
    {579, (void *)&sys_sblock_xexit}, // sys_sblock_xexit
    {580, (void *)&sys_eport_create}, // sys_eport_create
    {581, (void *)&sys_eport_delete}, // sys_eport_delete
    {582, (void *)&sys_eport_trigger}, // sys_eport_trigger
    {583, (void *)&sys_eport_open}, // sys_eport_open
    {584, (void *)&sys_eport_close}, // sys_eport_close
    {585, (void *)&sys_is_in_sandbox},
    {586, (void *)&sys_dmem_container},
    {587, (void *)&sys_get_authinfo},
    {588, (void *)&sys_mname},
    {589, (void *)&sys_dynlib_dlopen},
    {590, (void *)&sys_dynlib_dlclose}, // sys_dynlib_dlclose
    {591, (void *)&sys_dynlib_dlsym},
    {592, (void *)&sys_dynlib_get_list},
    {593, (void *)&sys_dynlib_get_info},
    {594, (void *)&sys_dynlib_load_prx},
    {595, (void *)&sys_dynlib_unload_prx},
    {596, (void *)&sys_dynlib_do_copy_relocations},
    {597, (void *)&sys_dynlib_prepare_dlclose}, // sys_dynlib_prepare_dlclose
    {598, (void *)&sys_dynlib_get_proc_param},
    {599, (void *)&sys_dynlib_process_needed_and_relocate},
    {600, (void *)&sys_sandbox_path}, // sys_sandbox_path
    {601, (void *)&sys_mdbg_service},
    {602, (void *)&sys_randomized_path},
    {603, (void *)&sys_rdup}, // sys_rdup
    {604, (void *)&sys_dl_get_metadata}, // sys_dl_get_metadata
    {605, (void *)&sys_workaround8849},
    {606, (void *)&sys_is_development_mode}, // sys_is_development_mode
    {607, (void *)&sys_get_self_auth_info}, // sys_get_self_auth_info
    {608, (void *)&sys_dynlib_get_info_ex},
    {609, (void *)&sys_budget_getid}, // sys_budget_getid
    {610, (void *)&sys_budget_get_ptype},
    {611, (void *)&sys_get_paging_stats_of_all_threads}, // sys_get_paging_stats_of_all_threads
    {612, (void *)&sys_get_proc_type_info},
    {613, (void *)&sys_get_resident_count}, // sys_get_resident_count
    {614, (void *)&sys_prepare_to_suspend_process}, // sys_prepare_to_suspend_process
    {615, (void *)&sys_get_resident_fmem_count}, // sys_get_resident_fmem_count
    {616, (void *)&sys_thr_get_name}, // sys_thr_get_name
    {617, (void *)&sys_set_gpo}, // sys_set_gpo
    {618, (void *)&sys_get_paging_stats_of_all_objects}, // sys_get_paging_stats_of_all_objects
    {619, (void *)&sys_test_debug_rwmem}, // sys_test_debug_rwmem
    {620, (void *)&sys_free_stack}, // sys_free_stack
    {621, (void *)&sys_suspend_system}, // sys_suspend_system
    {622, (void *)&sys_ipmimgr_call},
    {623, (void *)&sys_get_gpo}, // sys_get_gpo
    {624, (void *)&sys_get_vm_map_timestamp}, // sys_get_vm_map_timestamp
    {625, (void *)&sys_opmc_set_hw}, // sys_opmc_set_hw
    {626, (void *)&sys_opmc_get_hw}, // sys_opmc_get_hw
    {627, (void *)&sys_get_cpu_usage_all}, // sys_get_cpu_usage_all
    {628, (void *)&sys_mmap_dmem}, // sys_mmap_dmem
    {629, (void *)&sys_physhm_open}, // sys_physhm_open
    {630, (void *)&sys_physhm_unlink}, // sys_physhm_unlink
    {631, (void *)&sys_resume_internal_hdd}, // sys_resume_internal_hdd
    {632, (void *)&sys_thr_suspend_ucontext}, // sys_thr_suspend_ucontext
    {633, (void *)&sys_thr_resume_ucontext}, // sys_thr_resume_ucontext
    {634, (void *)&sys_thr_get_ucontext}, // sys_thr_get_ucontext
    {635, (void *)&sys_thr_set_ucontext}, // sys_thr_set_ucontext
    {636, (void *)&sys_set_timezone_info}, // sys_set_timezone_info
    {637, (void *)&sys_set_phys_fmem_limit}, // sys_set_phys_fmem_limit
    {638, (void *)&sys_utc_to_localtime}, // sys_utc_to_localtime
    {639, (void *)&sys_localtime_to_utc}, // sys_localtime_to_utc
    {640, (void *)&sys_set_uevt}, // sys_set_uevt
    {641, (void *)&sys_get_cpu_usage_proc}, // sys_get_cpu_usage_proc
    {642, (void *)&sys_get_map_statistics}, // sys_get_map_statistics
    {643, (void *)&sys_set_chicken_switches}, // sys_set_chicken_switches
    {644, (void *)&sys_extend_page_table_pool}, // sys_extend_page_table_pool
    {645, (void *)&sys_unk645}, // sys_#645
    {646, (void *)&sys_get_kernel_mem_statistics}, // sys_get_kernel_mem_statistics
    {647, (void *)&sys_get_sdk_compiled_version}, // sys_get_sdk_compiled_version
    {648, (void *)&sys_app_state_change}, // sys_app_state_change
    {649, (void *)&sys_dynlib_get_obj_member},
    {650, (void *)&sys_budget_get_ptype_of_budget}, // sys_budget_get_ptype_of_budget
    {651, (void *)&sys_prepare_to_resume_process}, // sys_prepare_to_resume_process
    {652, (void *)&sys_process_terminate}, // sys_process_terminate
    {653, (void *)&sys_blockpool_open},
    {654, (void *)&sys_blockpool_map}, // sys_blockpool_map
    {655, (void *)&sys_blockpool_unmap}, // sys_blockpool_unmap
    {656, (void *)&sys_dynlib_get_info_for_libdbg}, // sys_dynlib_get_info_for_libdbg
    {657, (void *)&sys_blockpool_batch}, // sys_blockpool_batch
    {658, (void *)&sys_fdatasync}, // sys_fdatasync
    {659, (void *)&sys_dynlib_get_list2}, // sys_dynlib_get_list2
    {660, (void *)&sys_dynlib_get_info2}, // sys_dynlib_get_info2
    {661, (void *)&sys_aio_unsupported}, // sys_aio_submit
    {662, (void *)&sys_aio_unsupported}, // sys_aio_multi_delete
    {663, (void *)&sys_aio_unsupported}, // sys_aio_multi_wait
    {664, (void *)&sys_aio_unsupported}, // sys_aio_multi_poll
    {665, (void *)&sys_aio_unsupported}, // sys_aio_get_data
    {666, (void *)&sys_aio_unsupported}, // sys_aio_multi_cancel
    {667, (void *)&sys_get_bio_usage_all}, // sys_get_bio_usage_all
    {668, (void *)&sys_aio_unsupported}, // sys_aio_create
    {669, (void *)&sys_aio_unsupported}, // sys_aio_submit_cmd
    {670, (void *)&sys_aio_init}, // sys_aio_init
    {671, (void *)&sys_get_page_table_stats}, // sys_get_page_table_stats
    {672, (void *)&sys_dynlib_get_list_for_libdbg}, // sys_dynlib_get_list_for_libdbg
};

// BSD/PS4 syscall return convention: on failure the kernel returns the positive
// errno in rax with the carry flag SET; on success it clears carry and rax holds
// the result. Our C handlers use the Linux-style convention instead (a negative
// errno, or the result, in rax). Classify a raw handler return for both the
// native trampoline and the FEX syscall bridge.
//
// An `int` handler zero-extends its 32-bit result into rax, an `int64_t` handler
// fills all 64 bits, so a negative errno arrives as either 0x00000000_FFFFFFxx or
// 0xFFFFFFFF_FFFFFFxx. Guest pointers live at >= 64 GiB and cannot match this.
extern "C" uint32_t krnl_syscall_errno(uint64_t raw) {
  int32_t lo = static_cast<int32_t>(static_cast<uint32_t>(raw));
  uint32_t hi = static_cast<uint32_t>(raw >> 32);
  if (lo < 0 && lo >= -static_cast<int32_t>(SysError::eLAST) &&
      (hi == 0 || hi == 0xFFFFFFFFu))
    return static_cast<uint32_t>(-lo);
  return 0;
}

#if defined(DELTA_BACKEND_NATIVE)
static void PS4ABI trace_syscall(const char *name, int index, void *addr) {
  std::fprintf(stderr, "[syscall] %d %s\n", index, name);
}

static void PS4ABI trace_ret(int index, int64_t ret) {
  std::fprintf(stderr, "[syscall]   -> %d returned %lld (%#llx)\n", index,
               (long long)ret, (unsigned long long)ret);
}

// DELTA_SCERR_TRACE: emitted (when set) on the trampoline's error path so we can
// see exactly which syscall returned which errno just before a guest abort. The
// errno is the BSD-style positive value the guest's libkernel turns into an SCE
// error (0x80020000 | errno), e.g. errno 1 -> 0x80020001.
static void PS4ABI trace_syscall_err(uint32_t sid, uint32_t err) {
  const char *name = syscall_getname(sid);
  std::fprintf(stderr, "[syscallerr] %u %s -> errno %u (SCE %#010x)\n", sid,
               name ? name : "?", err, 0x80020000u | err);
}

static uintptr_t emit_calltrace(const char *name, uint32_t sid,
                                const void *dest) {
  struct callTrace : Xbyak::CodeGenerator {
    callTrace(uintptr_t name, uint32_t sid, uintptr_t dest) {
      push(rdi);
      push(rsi);
      push(rdx);
      push(rcx);
      push(r8);
      push(r9);
      push(r10);
      push(r11);

      sub(rsp, 0x28);
      mov(rdi, name);
      mov(esi, sid);
      mov(rdx, rsp);
      mov(rcx, reinterpret_cast<uintptr_t>(&trace_syscall));
      call(rcx);
      add(rsp, 0x28);

      pop(r11);
      pop(r10);
      pop(r9);
      pop(r8);
      pop(rcx);
      pop(rdx);
      pop(rsi);
      pop(rdi);

      // call the real handler (not a tail jump) so we can log its return
      sub(rsp, 8);  // keep 16-byte alignment across the call
      mov(rax, dest);
      call(rax);
      // log the return value
      push(rax);
      mov(esi, eax);
      mov(edi, sid);
      mov(rax, reinterpret_cast<uintptr_t>(&trace_ret));
      call(rax);
      pop(rax);
      add(rsp, 8);
      // FreeBSD syscall ABI: carry clear = success. Our C handlers don't touch
      // CF, so clear it here or the guest's `jb cerror` reads garbage.
      clc();
      ret();
    }
  };

  callTrace *gen = new callTrace(reinterpret_cast<uintptr_t>(name), sid,
                                 reinterpret_cast<uintptr_t>(dest));

  return reinterpret_cast<uintptr_t>(gen->getCode());
}

#endif // DELTA_BACKEND_NATIVE

// Per-syscall-id call counter (DELTA_SCHIST). The only profiler available on this
// build: perf/strace/`/proc/PID/mem` are all yama-blocked, so to find what a
// wedged/slow title hammers, count every syscall in the trampoline and dump the
// histogram from the SIGUSR1 probe (crash.cpp). Racy increments are fine here.
// Deliberately outside the DELTA_BACKEND_NATIVE guard: null_handler reads
// g_scHist/dumpSyscallHist unconditionally, so the FEX build must link them too
// (only the trampoline that increments g_sysHist is native-only).
extern "C" uint64_t g_sysHist[1024] = {};
extern const bool g_scHist;
const bool g_scHist = std::getenv("DELTA_SCHIST") != nullptr;

// The histogram was only ever printed by the crash reporter, so a clean run
// discarded it. "Which syscall is this title hammering" is the question it
// exists to answer, and most runs neither crash nor exit gracefully (the
// emulator is normally SIGKILLed), so it is also dumped from the unimplemented
// -syscall stub, which is where the question usually comes up.
void dumpSyscallHist() {
  std::fprintf(stderr, "[schist] syscall call counts:\n");
  for (int i = 0; i < 1024; i++) {
    if (!g_sysHist[i])
      continue;
    const char *n = syscall_getname(i);
    std::fprintf(stderr, "[schist] %4d %-24s %llu\n", i, n ? n : "?",
                 (unsigned long long)g_sysHist[i]);
  }
}

static const bool g_scHistDump = [] {
  if (!g_scHist)
    return false;
  std::atexit(&dumpSyscallHist);
  // The emulator is normally SIGKILLed, so also sample on a timer: what a title
  // is doing in STEADY STATE is a different question from what it did at boot,
  // and only a periodic dump answers it.
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(20));
      dumpSyscallHist();
    }
  }).detach();
  return true;
}();

#if defined(DELTA_BACKEND_NATIVE)
// One-time trampoline per handler: call it with the guest's arg registers
// untouched, then set/clear the carry flag and normalise rax per the convention
// above. Replaces the bare `call handler` so the guest sees faithful errors.
static uintptr_t emit_bsd_trampoline(const void *handler, uint32_t sid,
                                     bool trace, bool count) {
  struct bsdRet : Xbyak::CodeGenerator {
    bsdRet(uintptr_t handler, uint32_t sid, bool trace, bool count) {
      if (count) {          // DELTA_SCHIST: ++g_sysHist[sid] (rax is caller-saved)
        mov(rax, reinterpret_cast<uintptr_t>(&g_sysHist[sid & 1023]));
        inc(qword[rax]);
      }
      push(rbx);            // save guest rbx; rsp now 16-aligned for the calls
      mov(rax, handler);
      call(rax);            // handler(rdi,rsi,rdx,rcx,r8,r9) -> rax (args intact)
      mov(rbx, rax);        // stash the raw return across the helper call
      mov(rdi, rax);
      mov(rax, reinterpret_cast<uintptr_t>(&krnl_syscall_errno));
      call(rax);            // eax = errno, or 0 when the return is a result
      test(eax, eax);
      Xbyak::Label ok;
      jz(ok);
      // error path: eax = positive errno, rsp 16-aligned (after the call ret).
      if (trace) {
        push(rax);          // save errno (rsp now misaligned)
        mov(esi, eax);      // arg2 = errno
        mov(edi, sid);      // arg1 = syscall id
        sub(rsp, 8);        // realign for the call
        mov(rax, reinterpret_cast<uintptr_t>(&trace_syscall_err));
        call(rax);
        add(rsp, 8);
        pop(rax);           // restore errno
      }
      pop(rbx);
      stc();                // error: rax already = positive errno
      ret();
      L(ok);
      mov(rax, rbx);        // success: restore the raw result
      pop(rbx);
      clc();
      ret();
    }
  };
  auto *gen = new bsdRet(reinterpret_cast<uintptr_t>(handler), sid, trace, count);
  return reinterpret_cast<uintptr_t>(gen->getCode());
}
#endif // DELTA_BACKEND_NATIVE

uintptr_t lv2_get(uint32_t sid) {
  const void *handler = reinterpret_cast<const void *>(&null_handler_notable);
  for (auto &it : syscall_dpt) {
    if (sid == it.id) {
      handler = it.ptr;
      break;
    }
  }

#if defined(DELTA_BACKEND_NATIVE)
  // Wrap each handler once in a trampoline that applies the BSD carry/errno
  // return convention. Cache by handler so syscall ids that share a handler
  // (and the many duplicate sites in the guest) reuse a single stub. When
  // DELTA_SCERR_TRACE is set the stub also logs its errno on the error path, so
  // key the cache by sid instead (each id reports its own name).
  static const bool traceErr = std::getenv("DELTA_SCERR_TRACE") != nullptr;
  static std::mutex trMutex;
  static std::unordered_map<uint64_t, uintptr_t> trCache;
  std::lock_guard<std::mutex> lk(trMutex);
  // Per-sid stub (not deduped by handler) when tracing errors OR counting, so
  // each id reports/counts under its own number.
  uint64_t key = (traceErr || g_scHist) ? static_cast<uint64_t>(sid)
                                        : reinterpret_cast<uint64_t>(handler);
  auto it = trCache.find(key);
  if (it != trCache.end())
    return it->second;
  uintptr_t tr = emit_bsd_trampoline(handler, sid, traceErr, g_scHist);
  trCache.emplace(key, tr);
  return tr;
#else
  return reinterpret_cast<uintptr_t>(handler);
#endif
}

// Exposed so the PS5 syscall layer (kern/ps5/lv2) can wrap its own handlers in
// the same BSD carry/errno trampoline without duplicating the codegen.
uintptr_t lv2_trampoline(const void *handler, uint32_t sid) {
#if defined(DELTA_BACKEND_NATIVE)
  static std::mutex m;
  static std::unordered_map<uint64_t, uintptr_t> cache;
  std::lock_guard<std::mutex> lk(m);
  uint64_t key = reinterpret_cast<uint64_t>(handler);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  uintptr_t tr = emit_bsd_trampoline(handler, sid, false, false);
  cache.emplace(key, tr);
  return tr;
#else
  return reinterpret_cast<uintptr_t>(handler);
#endif
}
} // namespace krnl
