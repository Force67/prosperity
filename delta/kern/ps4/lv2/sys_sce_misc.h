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

/* --- Sony kernel --- */
int64_t PS4ABI sys_jitshm_create(size_t len, uint32_t flags);
int PS4ABI sys_jitshm_alias();
int PS4ABI sys_dl_get_list();
int PS4ABI sys_dl_get_info();
int PS4ABI sys_dl_notify_event();
int PS4ABI sys_debug_init();
int PS4ABI sys_suspend_process();
int PS4ABI sys_resume_process();
int PS4ABI sys_prepare_to_suspend_process();
int PS4ABI sys_prepare_to_resume_process();
int PS4ABI sys_process_terminate();
int PS4ABI sys_suspend_system();
int PS4ABI sys_opmc_enable();
int PS4ABI sys_opmc_disable();
int PS4ABI sys_opmc_set_ctl();
int PS4ABI sys_opmc_set_ctr();
int PS4ABI sys_opmc_get_ctr();
int PS4ABI sys_opmc_set_hw();
int PS4ABI sys_opmc_get_hw();
int PS4ABI sys_budget_create();
int PS4ABI sys_budget_delete();
int PS4ABI sys_budget_get();
int PS4ABI sys_budget_set();
int PS4ABI sys_budget_getid();
int PS4ABI sys_budget_get_ptype_of_budget();
int PS4ABI sys_mdbg_call();
int PS4ABI sys_sblock_create();
int PS4ABI sys_sblock_delete();
int PS4ABI sys_sblock_enter();
int PS4ABI sys_sblock_exit();
int PS4ABI sys_sblock_xenter();
int PS4ABI sys_sblock_xexit();
int PS4ABI sys_eport_create();
int PS4ABI sys_eport_delete();
int PS4ABI sys_eport_trigger();
int PS4ABI sys_eport_open();
int PS4ABI sys_eport_close();
int PS4ABI sys_dynlib_dlclose();
int PS4ABI sys_dynlib_prepare_dlclose();
int PS4ABI sys_sandbox_path(char *out, size_t *len, void *, void *);
int PS4ABI sys_rdup();
int PS4ABI sys_dl_get_metadata();
int PS4ABI sys_is_development_mode();
int PS4ABI sys_get_self_auth_info(int pid, void *out);
int PS4ABI sys_get_paging_stats_of_all_threads();
int PS4ABI sys_get_paging_stats_of_all_objects();
int PS4ABI sys_get_resident_count();
int PS4ABI sys_get_resident_fmem_count();
int PS4ABI sys_set_gpo();
int PS4ABI sys_get_gpo();
int PS4ABI sys_test_debug_rwmem();
int PS4ABI sys_get_cpu_usage_all();
int PS4ABI sys_get_cpu_usage_proc();
int PS4ABI sys_physhm_open();
int PS4ABI sys_physhm_unlink();
int PS4ABI sys_resume_internal_hdd();
int PS4ABI sys_set_phys_fmem_limit();
int PS4ABI sys_set_uevt();
int PS4ABI sys_set_chicken_switches();
int PS4ABI sys_unk645();
int PS4ABI sys_get_kernel_mem_statistics(void *out);
int PS4ABI sys_get_sdk_compiled_version();
int PS4ABI sys_app_state_change();
int64_t PS4ABI sys_blockpool_map(int64_t pool, size_t len, uint32_t prot,
                                 uint32_t flags);
int PS4ABI sys_blockpool_unmap();
int64_t PS4ABI sys_blockpool_batch(uint64_t a0, uint64_t a1, uint64_t a2,
                                   uint64_t a3, uint64_t a4, uint64_t a5);
int PS4ABI sys_dynlib_get_info_for_libdbg();
int PS4ABI sys_dynlib_get_list_for_libdbg();
int PS4ABI sys_dynlib_get_list2();
int PS4ABI sys_dynlib_get_info2();
int PS4ABI sys_get_page_table_stats();

/* --- AIO (asynchronous IO, unmodeled) --- */
int PS4ABI sys_aio_unsupported();
int PS4ABI sys_get_bio_usage_all();
int PS4ABI sys_aio_init();

/* --- leftover POSIX --- */
int PS4ABI sys_getgroups(int gidsetlen, uint32_t *gidset);
int PS4ABI sys_setgroups();
int PS4ABI sys_setpriority();
int PS4ABI sys_getpriority();
int PS4ABI sys_setsockopt();
int PS4ABI sys_getsockopt(int fd, int level, int name, void *val, uint32_t *len);
int PS4ABI sys_sync();
int PS4ABI sys_getpagesize();
int PS4ABI sys_flock();
int PS4ABI sys_utimes();
int PS4ABI sys_futimes();
int PS4ABI sys_pathconf(const char *path, int name);
int PS4ABI sys_fpathconf(int fd, int name);
int PS4ABI sys_lpathconf(const char *path, int name);
int PS4ABI sys_sigqueue();
int PS4ABI sys_abort2(const char *msg, int nargs, void **args);
int PS4ABI sys_thr_sleep();
int PS4ABI sys_thr_wakeup();
int PS4ABI sys_posix_fallocate();
int PS4ABI sys_posix_fadvise();

} // namespace krnl
