
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdint>
#include <cstdio>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <xbyak.h>

#include "sys_debug.h"
#include "sys_dynlib.h"
#include "sys_generic.h"
#include "sys_info.h"
#include "sys_mem.h"
#include "sys_thread.h"
#include "sys_vfs.h"
#include "sys_net.h"

#include "kern/module.h"
#include "kern/proc.h"

namespace krnl {
const char *syscall_getname(uint32_t idx);

int sys_write(uint32_t fd, const void *buf, size_t nbytes);
int sys_sigprocmask(int, const int *, int *);
int sys_sigaction(int how, void (*cb)(void *, void *, void *));
int sys_regmgr_call(uint32_t op, uint32_t id, void *result, void *value,
                    uint64_t type);

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

static int PS4ABI null_handler() {
#ifdef _MSC_VER
  void *ret = _ReturnAddress();
#else
  void *ret = __builtin_return_address(0);
#endif
  called_in(ret);

  std::printf(">>>>>>>>>>>>> NULL HANDLER CALLED BY %p\n", ret);
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
    {3, (void *)&null_handler}, // sys_read
    {4, (void *)&sys_write},
    {5, (void *)&sys_open},
    {6, (void *)&sys_close},
    {7, (void *)&null_handler},  // sys_wait4
    {8, (void *)&null_handler},  // sys_creat
    {9, (void *)&null_handler},  // sys_link
    {10, (void *)&null_handler}, // sys_unlink
    {11, (void *)&null_handler}, // sys_execv
    {12, (void *)&null_handler}, // sys_chdir
    {13, (void *)&null_handler}, // sys_fchdir
    {14, (void *)&null_handler}, // sys_mkd
    {15, (void *)&null_handler}, // sys_chmod
    {16, (void *)&null_handler}, // sys_chown
    {17, (void *)&null_handler}, // sys_obreak
    {18, (void *)&null_handler}, // sys_getfsstat
    {19, (void *)&null_handler}, // sys_lseek
    {20, (void *)&sys_getpid},
    {21, (void *)&null_handler}, // sys_mount
    {22, (void *)&null_handler}, // sys_unmount
    {23, (void *)&null_handler}, // sys_setuid
    {24, (void *)&null_handler}, // sys_getuid
    {25, (void *)&null_handler}, // sys_geteuid
    {26, (void *)&null_handler}, // sys_ptrace
    {27, (void *)&null_handler}, // sys_recvmsg
    {28, (void *)&null_handler}, // sys_sendmsg
    {29, (void *)&null_handler}, // sys_recvfrom
    {30, (void *)&null_handler}, // sys_accept
    {31, (void *)&null_handler}, // sys_getpeername
    {32, (void *)&null_handler}, // sys_getsockname
    {33, (void *)&null_handler}, // sys_access
    {34, (void *)&null_handler}, // sys_chflags
    {35, (void *)&null_handler}, // sys_fchflags
    {36, (void *)&null_handler}, // sys_sync
    {37, (void *)&null_handler}, // sys_kill
    {38, (void *)&null_handler}, // sys_stat
    {39, (void *)&null_handler}, // sys_getppid
    {40, (void *)&null_handler}, // sys_lstat
    {41, (void *)&null_handler}, // sys_dup
    {42, (void *)&null_handler}, // sys_pipe
    {43, (void *)&null_handler}, // sys_getegid
    {44, (void *)&null_handler}, // sys_profil
    {45, (void *)&null_handler}, // sys_ktrace
    {46, (void *)&sys_sigaction},
    {47, (void *)&null_handler}, // sys_getgid
    {48, (void *)&sys_sigprocmask},
    {49, (void *)&null_handler}, // sys_getlogin
    {50, (void *)&null_handler}, // sys_setlogin
    {51, (void *)&null_handler}, // sys_acct
    {52, (void *)&null_handler}, // sys_sigpending
    {53, (void *)&null_handler}, // sys_sigaltstack
    {54, (void *)&sys_ioctl},
    {55, (void *)&null_handler}, // sys_reboot
    {56, (void *)&null_handler}, // sys_revoke
    {57, (void *)&null_handler}, // sys_symlink
    {58, (void *)&null_handler}, // sys_readlink
    {59, (void *)&sys_execve},
    {60, (void *)&null_handler}, // sys_umask
    {61, (void *)&null_handler}, // sys_chroot
    {62, (void *)&null_handler}, // sys_fstat
    {63, (void *)&null_handler}, // sys_getkerninfo
    {64, (void *)&null_handler}, // sys_getpagesize
    {65, (void *)&null_handler}, // sys_msync
    {66, (void *)&null_handler}, // sys_vfork
    {67, (void *)&null_handler}, // sys_vread
    {68, (void *)&null_handler}, // sys_vwrite
    {69, (void *)&null_handler}, // sys_sbrk
    {70, (void *)&null_handler}, // sys_sstk
    {71, (void *)&sys_mmap},
    {72, (void *)&null_handler},  // sys_ovadvise
    {73, (void *)&null_handler},  // sys_munmap
    {74, (void *)&sys_mprotect},
    {75, (void *)&null_handler},  // sys_madvise
    {76, (void *)&null_handler},  // sys_vhangup
    {77, (void *)&null_handler},  // sys_vlimit
    {78, (void *)&null_handler},  // sys_mincore
    {79, (void *)&null_handler},  // sys_getgroups
    {80, (void *)&null_handler},  // sys_setgroups
    {81, (void *)&null_handler},  // sys_getpgrp
    {82, (void *)&null_handler},  // sys_setpgid
    {83, (void *)&null_handler},  // sys_setitimer
    {84, (void *)&null_handler},  // sys_wait
    {85, (void *)&null_handler},  // sys_swapon
    {86, (void *)&null_handler},  // sys_getitimer
    {87, (void *)&null_handler},  // sys_gethostname
    {88, (void *)&null_handler},  // sys_sethostname
    {89, (void *)&null_handler},  // sys_getdtablesize
    {90, (void *)&null_handler},  // sys_dup2
    {91, (void *)&null_handler},  // sys_getdopt
    {92, (void *)&null_handler},  // sys_fcntl
    {93, (void *)&null_handler},  // sys_select
    {94, (void *)&null_handler},  // sys_setdopt
    {95, (void *)&null_handler},  // sys_fsync
    {96, (void *)&null_handler},  // sys_setpriority
    {97, (void *)&null_handler},  // sys_socket
    {98, (void *)&null_handler},  // sys_connect
    {99, (void *)&sys_netcontrol},
    {100, (void *)&null_handler}, // sys_getpriority
    {101, (void *)&null_handler}, // sys_netabort
    {102, (void *)&null_handler}, // sys_netgetsockinfo
    {103, (void *)&null_handler}, // sys_sigreturn
    {104, (void *)&null_handler}, // sys_bind
    {105, (void *)&null_handler}, // sys_setsockopt
    {106, (void *)&null_handler}, // sys_listen
    {107, (void *)&null_handler}, // sys_vtimes
    {108, (void *)&null_handler}, // sys_sigvec
    {109, (void *)&null_handler}, // sys_sigblock
    {110, (void *)&null_handler}, // sys_sigsetmask
    {111, (void *)&null_handler}, // sys_sigsuspend
    {112, (void *)&null_handler}, // sys_sigstack
    {113, (void *)&sys_socketex},
    {114, (void *)&null_handler}, // sys_socketclose
    {115, (void *)&null_handler}, // sys_vtrace
    {116, (void *)&null_handler}, // sys_gettimeofday
    {117, (void *)&null_handler}, // sys_getrusage
    {118, (void *)&null_handler}, // sys_getsockopt
    {119, (void *)&null_handler}, // sys_resuba
    {120, (void *)&null_handler}, // sys_readv
    {121, (void *)&null_handler}, // sys_writev
    {122, (void *)&null_handler}, // sys_settimeofday
    {123, (void *)&null_handler}, // sys_fchown
    {124, (void *)&null_handler}, // sys_fchmod
    {125, (void *)&null_handler}, // sys_netgetiflist
    {126, (void *)&null_handler}, // sys_setreuid
    {127, (void *)&null_handler}, // sys_setregid
    {128, (void *)&null_handler}, // sys_rename
    {129, (void *)&null_handler}, // sys_truncate
    {130, (void *)&null_handler}, // sys_ftruncate
    {131, (void *)&null_handler}, // sys_flock
    {132, (void *)&null_handler}, // sys_mkfifo
    {133, (void *)&null_handler}, // sys_sendto
    {134, (void *)&null_handler}, // sys_shutdown
    {135, (void *)&null_handler}, // sys_socketpair
    {136, (void *)&null_handler}, // sys_mkdir
    {137, (void *)&null_handler}, // sys_rmdir
    {138, (void *)&null_handler}, // sys_utimes
    {139, (void *)&null_handler}, // sys_sigreturn
    {140, (void *)&null_handler}, // sys_adjtime
    {141, (void *)&null_handler}, // sys_kqueueex
    {142, (void *)&null_handler}, // sys_gethostid
    {143, (void *)&null_handler}, // sys_sethostid
    {144, (void *)&null_handler}, // sys_getrlimit
    {145, (void *)&null_handler}, // sys_setrlimit
    {146, (void *)&null_handler}, // sys_killpg
    {147, (void *)&null_handler}, // sys_setsid
    {148, (void *)&null_handler}, // sys_quotactl
    {149, (void *)&null_handler}, // sys_quota
    {150, (void *)&null_handler}, // sys_getsockname
    {151, (void *)&null_handler}, // sys_sem_lock
    {152, (void *)&null_handler}, // sys_sem_wakeup
    {153, (void *)&null_handler}, // sys_asyncdaemon
    {154, (void *)&null_handler}, // sys_nlm_syscall
    {155, (void *)&null_handler}, // sys_nfssvc
    {156, (void *)&null_handler}, // sys_getdirentries
    {157, (void *)&null_handler}, // sys_statfs
    {158, (void *)&null_handler}, // sys_fstatfs
    {160, (void *)&null_handler}, // sys_lgetfh
    {161, (void *)&null_handler}, // sys_getfh
    {162, (void *)&null_handler}, // sys_getdomainname
    {163, (void *)&null_handler}, // sys_setdomainname
    {164, (void *)&null_handler}, // sys_uname
    {165, (void *)&sys_sysarch},
    {166, (void *)&null_handler}, // sys_rtprio
    {169, (void *)&null_handler}, // sys_semsys
    {170, (void *)&null_handler}, // sys_msgsys
    {171, (void *)&null_handler}, // sys_shmsys
    {173, (void *)&null_handler}, // sys_pread
    {174, (void *)&null_handler}, // sys_pwrite
    {175, (void *)&null_handler}, // sys_setfib
    {176, (void *)&null_handler}, // sys_ntp_adjtime
    {177, (void *)&null_handler}, // sys_sfork
    {178, (void *)&null_handler}, // sys_getdescriptor
    {179, (void *)&null_handler}, // sys_setdescriptor
    {181, (void *)&null_handler}, // sys_setgid
    {182, (void *)&null_handler}, // sys_setegid
    {183, (void *)&null_handler}, // sys_seteuid
    {184, (void *)&null_handler}, // sys_lfs_bmapv
    {185, (void *)&null_handler}, // sys_lfs_markv
    {186, (void *)&null_handler}, // sys_lfs_segclean
    {187, (void *)&null_handler}, // sys_lfs_segwait
    {188, (void *)&null_handler}, // sys_stat
    {189, (void *)&null_handler}, // sys_fstat
    {190, (void *)&null_handler}, // sys_lstat
    {191, (void *)&null_handler}, // sys_pathconf
    {192, (void *)&null_handler}, // sys_fpathconf
    {194, (void *)&null_handler}, // sys_getrlimit
    {195, (void *)&null_handler}, // sys_setrlimit
    {196, (void *)&null_handler}, // sys_getdirentries
    {197, (void *)&sys_mmap},
    {198, (void *)&null_handler}, // sys_nosys
    {199, (void *)&null_handler}, // sys_lseek
    {200, (void *)&null_handler}, // sys_truncate
    {201, (void *)&null_handler}, // sys_ftruncate
    {202, (void *)&sys_sysctl},
    {203, (void *)&null_handler}, // sys_mlock
    {204, (void *)&null_handler}, // sys_munlock
    {205, (void *)&null_handler}, // sys_undelete
    {206, (void *)&null_handler}, // sys_futimes
    {207, (void *)&null_handler}, // sys_getpgid
    {208, (void *)&null_handler}, // sys_newreboot
    {209, (void *)&null_handler}, // sys_poll
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
    {232, (void *)&null_handler}, // sys_clock_gettime
    {233, (void *)&null_handler}, // sys_clock_settime
    {234, (void *)&null_handler}, // sys_clock_getres
    {235, (void *)&null_handler}, // sys_ktimer_create
    {236, (void *)&null_handler}, // sys_ktimer_delete
    {237, (void *)&null_handler}, // sys_ktimer_settime
    {238, (void *)&null_handler}, // sys_ktimer_gettime
    {239, (void *)&null_handler}, // sys_ktimer_getoverrun
    {240, (void *)&null_handler}, // sys_nanosleep
    {241, (void *)&null_handler}, // sys_ffclock_getcounter
    {242, (void *)&null_handler}, // sys_ffclock_setestimate
    {243, (void *)&null_handler}, // sys_ffclock_getestimate
    {247, (void *)&null_handler}, // sys_clock_getcpuclockid2
    {248, (void *)&null_handler}, // sys_ntp_gettime
    {250, (void *)&null_handler}, // sys_minherit
    {251, (void *)&sys_rfork},
    {252, (void *)&null_handler}, // sys_openbsd_poll
    {253, (void *)&null_handler}, // sys_issetugid
    {254, (void *)&null_handler}, // sys_lchown
    {255, (void *)&null_handler}, // sys_aio_read
    {256, (void *)&null_handler}, // sys_aio_write
    {257, (void *)&null_handler}, // sys_lio_listio
    {272, (void *)&null_handler}, // sys_getdents
    {274, (void *)&null_handler}, // sys_lchmod
    {275, (void *)&null_handler}, // sys_lchown
    {276, (void *)&null_handler}, // sys_lutimes
    {277, (void *)&null_handler}, // sys_msync
    {278, (void *)&null_handler}, // sys_nstat
    {279, (void *)&null_handler}, // sys_nfstat
    {280, (void *)&null_handler}, // sys_nlstat
    {289, (void *)&null_handler}, // sys_preadv
    {290, (void *)&null_handler}, // sys_pwritev
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
    {310, (void *)&null_handler}, // sys_getsid
    {311, (void *)&null_handler}, // sys_setresuid
    {312, (void *)&null_handler}, // sys_setresgid
    {313, (void *)&null_handler}, // sys_signasleep
    {314, (void *)&null_handler}, // sys_aio_return
    {315, (void *)&null_handler}, // sys_aio_suspend
    {316, (void *)&null_handler}, // sys_aio_cancel
    {317, (void *)&null_handler}, // sys_aio_error
    {318, (void *)&null_handler}, // sys_aio_read
    {319, (void *)&null_handler}, // sys_aio_write
    {320, (void *)&null_handler}, // sys_lio_listio
    {321, (void *)&null_handler}, // sys_yield
    {322, (void *)&null_handler}, // sys_thr_sleep
    {323, (void *)&null_handler}, // sys_thr_wakeup
    {324, (void *)&null_handler}, // sys_mlockall
    {325, (void *)&null_handler}, // sys_munlockall
    {326, (void *)&null_handler}, // sys_getcwd
    {327, (void *)&null_handler}, // sys_sched_setparam
    {328, (void *)&null_handler}, // sys_sched_getparam
    {329, (void *)&null_handler}, // sys_sched_setscheduler
    {330, (void *)&null_handler}, // sys_sched_getscheduler
    {331, (void *)&null_handler}, // sys_sched_yield
    {332, (void *)&null_handler}, // sys_sched_get_priority_max
    {333, (void *)&null_handler}, // sys_sched_get_priority_min
    {334, (void *)&null_handler}, // sys_sched_rr_get_interval
    {335, (void *)&null_handler}, // sys_utrace
    {336, (void *)&null_handler}, // sys_sendfile
    {337, (void *)&null_handler}, // sys_kldsym
    {338, (void *)&null_handler}, // sys_jail
    {339, (void *)&null_handler}, // sys_nnpfs_syscall
    {340, (void *)&sys_sigprocmask},
    {341, (void *)&null_handler}, // sys_sigsuspend
    {342, (void *)&sys_sigaction},
    {343, (void *)&null_handler}, // sys_sigpending
    {344, (void *)&null_handler}, // sys_sigreturn
    {345, (void *)&null_handler}, // sys_sigtimedwait
    {346, (void *)&null_handler}, // sys_sigwaitinfo
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
    {359, (void *)&null_handler}, // sys_aio_waitcomplete
    {360, (void *)&null_handler}, // sys_getresuid
    {361, (void *)&null_handler}, // sys_getresgid
    {362, (void *)&null_handler}, // sys_kqueue
    {363, (void *)&null_handler}, // sys_kevent
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
    {400, (void *)&null_handler}, // sys_ksem_close
    {401, (void *)&null_handler}, // sys_ksem_post
    {402, (void *)&null_handler}, // sys_ksem_wait
    {403, (void *)&null_handler}, // sys_ksem_trywait
    {404, (void *)&null_handler}, // sys_ksem_init
    {405, (void *)&null_handler}, // sys_ksem_open
    {406, (void *)&null_handler}, // sys_ksem_unlink
    {407, (void *)&null_handler}, // sys_ksem_getvalue
    {408, (void *)&null_handler}, // sys_ksem_destroy
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
    {429, (void *)&null_handler}, // sys_sigwait
    {430, (void *)&null_handler}, // sys_thr_create
    {431, (void *)&null_handler}, // sys_thr_exit
    {432, (void *)&sys_thr_self},
    {433, (void *)&null_handler}, // sys_thr_kill
    {436, (void *)&null_handler}, // sys_jail_attach
    {437, (void *)&null_handler}, // sys_extattr_list_fd
    {438, (void *)&null_handler}, // sys_extattr_list_file
    {439, (void *)&null_handler}, // sys_extattr_list_link
    {440, (void *)&null_handler}, // sys_kse_switchin
    {441, (void *)&null_handler}, // sys_ksem_timedwait
    {442, (void *)&null_handler}, // sys_thr_suspend
    {443, (void *)&null_handler}, // sys_thr_wake
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
    {455, (void *)&null_handler}, // sys_thr_new
    {456, (void *)&null_handler}, // sys_sigqueue
    {457, (void *)&null_handler}, // sys_kmq_open
    {458, (void *)&null_handler}, // sys_kmq_setattr
    {459, (void *)&null_handler}, // sys_kmq_timedreceive
    {460, (void *)&null_handler}, // sys_kmq_timedsend
    {461, (void *)&null_handler}, // sys_kmq_tify
    {462, (void *)&null_handler}, // sys_kmq_unlink
    {463, (void *)&null_handler}, // sys_abort2
    {464, (void *)&null_handler}, // sys_thr_set_name
    {465, (void *)&null_handler}, // sys_aio_fsync
    {466, (void *)&sys_rtprio_thread},
    {469, (void *)&null_handler}, // sys_getpath_fromfd
    {470, (void *)&null_handler}, // sys_getpath_fromaddr
    {471, (void *)&null_handler}, // sys_sctp_peeloff
    {472, (void *)&null_handler}, // sys_sctp_generic_sendmsg
    {473, (void *)&null_handler}, // sys_sctp_generic_sendmsg_iov
    {474, (void *)&null_handler}, // sys_sctp_generic_recvmsg
    {475, (void *)&null_handler}, // sys_pread
    {476, (void *)&null_handler}, // sys_pwrite
    {477, (void *)&sys_mmap},
    {478, (void *)&null_handler}, // sys_lseek
    {479, (void *)&null_handler}, // sys_truncate
    {480, (void *)&null_handler}, // sys_ftruncate
    {481, (void *)&null_handler}, // sys_thr_kill2
    {482, (void *)&null_handler}, // sys_shm_open
    {483, (void *)&null_handler}, // sys_shm_unlink
    {484, (void *)&null_handler}, // sys_cpuset
    {485, (void *)&null_handler}, // sys_cpuset_setid
    {486, (void *)&null_handler}, // sys_cpuset_getid
    {487, (void *)&sys_cpuset_getaffinity},
    {488, (void *)&null_handler}, // sys_cpuset_setaffinity
    {489, (void *)&null_handler}, // sys_faccessat
    {490, (void *)&null_handler}, // sys_fchmodat
    {491, (void *)&null_handler}, // sys_fchownat
    {492, (void *)&null_handler}, // sys_fexecve
    {493, (void *)&null_handler}, // sys_fstatat
    {494, (void *)&null_handler}, // sys_futimesat
    {495, (void *)&null_handler}, // sys_linkat
    {496, (void *)&null_handler}, // sys_mkdirat
    {497, (void *)&null_handler}, // sys_mkfifoat
    {498, (void *)&null_handler}, // sys_mkdat
    {499, (void *)&null_handler}, // sys_openat
    {500, (void *)&null_handler}, // sys_readlinkat
    {501, (void *)&null_handler}, // sys_renameat
    {502, (void *)&null_handler}, // sys_symlinkat
    {503, (void *)&null_handler}, // sys_unlinkat
    {504, (void *)&null_handler}, // sys_posix_openpt
    {505, (void *)&null_handler}, // sys_gssd_syscall
    {506, (void *)&null_handler}, // sys_jail_get
    {507, (void *)&null_handler}, // sys_jail_set
    {508, (void *)&null_handler}, // sys_jail_remove
    {509, (void *)&null_handler}, // sys_closefrom
    {510, (void *)&null_handler}, // sys_semctl
    {511, (void *)&null_handler}, // sys_msgctl
    {512, (void *)&null_handler}, // sys_shmctl
    {513, (void *)&null_handler}, // sys_lpathconf
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
    {530, (void *)&null_handler}, // sys_posix_fallocate
    {531, (void *)&null_handler}, // sys_posix_fadvise
    {532, (void *)&sys_regmgr_call},
    {533, (void *)&null_handler}, // sys_jitshm_create
    {534, (void *)&null_handler}, // sys_jitshm_alias
    {535, (void *)&null_handler}, // sys_dl_get_list
    {536, (void *)&null_handler}, // sys_dl_get_info
    {537, (void *)&null_handler}, // sys_dl_notify_event
    {538, (void *)&null_handler}, // sys_evf_create
    {539, (void *)&null_handler}, // sys_evf_delete
    {540, (void *)&null_handler}, // sys_evf_open
    {541, (void *)&null_handler}, // sys_evf_close
    {542, (void *)&null_handler}, // sys_evf_wait
    {543, (void *)&null_handler}, // sys_evf_trywait
    {544, (void *)&null_handler}, // sys_evf_set
    {545, (void *)&null_handler}, // sys_evf_clear
    {546, (void *)&null_handler}, // sys_evf_cancel
    {547, (void *)&null_handler}, // sys_query_memory_protection
    {548, (void *)&null_handler}, // sys_batch_map
    {549, (void *)&null_handler}, // sys_osem_create
    {550, (void *)&null_handler}, // sys_osem_delete
    {551, (void *)&null_handler}, // sys_osem_open
    {552, (void *)&null_handler}, // sys_osem_close
    {553, (void *)&null_handler}, // sys_osem_wait
    {554, (void *)&null_handler}, // sys_osem_wait
    {555, (void *)&null_handler}, // sys_osem_post
    {556, (void *)&null_handler}, // sys_osem_cancel
    {557, (void *)&sys_namedobj_create},
    {558, (void *)&sys_namedobj_delete},
    {559, (void *)&null_handler}, // sys_set_vm_container
    {560, (void *)&null_handler}, // sys_debug_init
    {561, (void *)&null_handler}, // sys_suspend_process
    {562, (void *)&null_handler}, // sys_resume_process
    {563, (void *)&null_handler}, // sys_opmc_enable
    {564, (void *)&null_handler}, // sys_opmc_disable
    {565, (void *)&null_handler}, // sys_opmc_set_ctl
    {566, (void *)&null_handler}, // sys_opmc_set_ctr
    {567, (void *)&null_handler}, // sys_opmc_get_ctr
    {568, (void *)&null_handler}, // sys_budget_create
    {569, (void *)&null_handler}, // sys_budget_delete
    {570, (void *)&null_handler}, // sys_budget_get
    {571, (void *)&null_handler}, // sys_budget_set
    {572, (void *)&null_handler}, // sys_virtual_query
    {573, (void *)&null_handler}, // sys_mdbg_call
    {574, (void *)&null_handler}, // sys_sblock_create
    {575, (void *)&null_handler}, // sys_sblock_delete
    {576, (void *)&null_handler}, // sys_sblock_enter
    {577, (void *)&null_handler}, // sys_sblock_exit
    {578, (void *)&null_handler}, // sys_sblock_xenter
    {579, (void *)&null_handler}, // sys_sblock_xexit
    {580, (void *)&null_handler}, // sys_eport_create
    {581, (void *)&null_handler}, // sys_eport_delete
    {582, (void *)&null_handler}, // sys_eport_trigger
    {583, (void *)&null_handler}, // sys_eport_open
    {584, (void *)&null_handler}, // sys_eport_close
    {585, (void *)&sys_is_in_sandbox},
    {586, (void *)&sys_dmem_container},
    {587, (void *)&sys_get_authinfo},
    {588, (void *)&sys_mname},
    {589, (void *)&sys_dynlib_dlopen},
    {590, (void *)&null_handler}, // sys_dynlib_dlclose
    {591, (void *)&sys_dynlib_dlsym},
    {592, (void *)&sys_dynlib_get_list},
    {593, (void *)&sys_dynlib_get_info},
    {594, (void *)&null_handler}, // sys_dynlib_load_prx
    {595, (void *)&null_handler}, // sys_dynlib_unload_prx
    {596, (void *)&null_handler}, // sys_dynlib_do_copy_relocations
    {597, (void *)&null_handler}, // sys_dynlib_prepare_dlclose
    {598, (void *)&sys_dynlib_get_proc_param},
    {599, (void *)&sys_dynlib_process_needed_and_relocate},
    {600, (void *)&null_handler}, // sys_sandbox_path
    {601, (void *)&sys_mdbg_service},
    {602, (void *)&null_handler}, // sys_randomized_path
    {603, (void *)&null_handler}, // sys_rdup
    {604, (void *)&null_handler}, // sys_dl_get_metadata
    {605, (void *)&null_handler}, // sys_workaround8849
    {606, (void *)&null_handler}, // sys_is_development_mode
    {607, (void *)&null_handler}, // sys_get_self_auth_info
    {608, (void *)&sys_dynlib_get_info_ex},
    {609, (void *)&null_handler}, // sys_budget_getid
    {610, (void *)&sys_budget_get_ptype},
    {611, (void *)&null_handler}, // sys_get_paging_stats_of_all_threads
    {612, (void *)&sys_get_proc_type_info},
    {613, (void *)&null_handler}, // sys_get_resident_count
    {614, (void *)&null_handler}, // sys_prepare_to_suspend_process
    {615, (void *)&null_handler}, // sys_get_resident_fmem_count
    {616, (void *)&null_handler}, // sys_thr_get_name
    {617, (void *)&null_handler}, // sys_set_gpo
    {618, (void *)&null_handler}, // sys_get_paging_stats_of_all_objects
    {619, (void *)&null_handler}, // sys_test_debug_rwmem
    {620, (void *)&null_handler}, // sys_free_stack
    {621, (void *)&null_handler}, // sys_suspend_system
    {622, (void *)&null_handler}, // sys_ipmimgr_call
    {623, (void *)&null_handler}, // sys_get_gpo
    {624, (void *)&null_handler}, // sys_get_vm_map_timestamp
    {625, (void *)&null_handler}, // sys_opmc_set_hw
    {626, (void *)&null_handler}, // sys_opmc_get_hw
    {627, (void *)&null_handler}, // sys_get_cpu_usage_all
    {628, (void *)&null_handler}, // sys_mmap_dmem
    {629, (void *)&null_handler}, // sys_physhm_open
    {630, (void *)&null_handler}, // sys_physhm_unlink
    {631, (void *)&null_handler}, // sys_resume_internal_hdd
    {632, (void *)&null_handler}, // sys_thr_suspend_ucontext
    {633, (void *)&null_handler}, // sys_thr_resume_ucontext
    {634, (void *)&null_handler}, // sys_thr_get_ucontext
    {635, (void *)&null_handler}, // sys_thr_set_ucontext
    {636, (void *)&null_handler}, // sys_set_timezone_info
    {637, (void *)&null_handler}, // sys_set_phys_fmem_limit
    {638, (void *)&null_handler}, // sys_utc_to_localtime
    {639, (void *)&null_handler}, // sys_localtime_to_utc
    {640, (void *)&null_handler}, // sys_set_uevt
    {641, (void *)&null_handler}, // sys_get_cpu_usage_proc
    {642, (void *)&null_handler}, // sys_get_map_statistics
    {643, (void *)&null_handler}, // sys_set_chicken_switches
    {644, (void *)&null_handler}, // sys_extend_page_table_pool
    {645, (void *)&null_handler}, // sys_#645
    {646, (void *)&null_handler}, // sys_get_kernel_mem_statistics
    {647, (void *)&null_handler}, // sys_get_sdk_compiled_version
    {648, (void *)&null_handler}, // sys_app_state_change
    {649, (void *)&sys_dynlib_get_obj_member},
    {650, (void *)&null_handler}, // sys_budget_get_ptype_of_budget
    {651, (void *)&null_handler}, // sys_prepare_to_resume_process
    {652, (void *)&null_handler}, // sys_process_terminate
    {653, (void *)&null_handler}, // sys_blockpool_open
    {654, (void *)&null_handler}, // sys_blockpool_map
    {655, (void *)&null_handler}, // sys_blockpool_unmap
    {656, (void *)&null_handler}, // sys_dynlib_get_info_for_libdbg
    {657, (void *)&null_handler}, // sys_blockpool_batch
    {658, (void *)&null_handler}, // sys_fdatasync
    {659, (void *)&null_handler}, // sys_dynlib_get_list2
    {660, (void *)&null_handler}, // sys_dynlib_get_info2
    {661, (void *)&null_handler}, // sys_aio_submit
    {662, (void *)&null_handler}, // sys_aio_multi_delete
    {663, (void *)&null_handler}, // sys_aio_multi_wait
    {664, (void *)&null_handler}, // sys_aio_multi_poll
    {665, (void *)&null_handler}, // sys_aio_get_data
    {666, (void *)&null_handler}, // sys_aio_multi_cancel
    {667, (void *)&null_handler}, // sys_get_bio_usage_all
    {668, (void *)&null_handler}, // sys_aio_create
    {669, (void *)&null_handler}, // sys_aio_submit_cmd
    {670, (void *)&null_handler}, // sys_aio_init
    {671, (void *)&null_handler}, // sys_get_page_table_stats
    {672, (void *)&null_handler}, // sys_dynlib_get_list_for_libdbg
};

static void PS4ABI trace_syscall(const char *name, int index, void *addr) {
  std::printf("trace_syscall: %s (%d), %p\n", name, index, addr);
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

      mov(rax, dest);
      jmp(rax);
    }
  };

  callTrace *gen = new callTrace(reinterpret_cast<uintptr_t>(name), sid,
                                 reinterpret_cast<uintptr_t>(dest));

  return reinterpret_cast<uintptr_t>(gen->getCode());
}

uintptr_t lv2_get(uint32_t sid) {
  for (auto &it : syscall_dpt) {
    if (sid == it.id) {
       return reinterpret_cast<uintptr_t>(it.ptr);
      //return emit_calltrace(syscall_getname(sid), sid, it.ptr);
    }
  }

  //	LOG_WARNING("unknown syscall {}", sid);
  return reinterpret_cast<uintptr_t>(&null_handler_notable);
}
} // namespace krnl