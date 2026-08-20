#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Async file IO (sceKernelAio*). The requests are served synchronously inside
 * submit and handed back already complete; the API allows a request to finish
 * at any point, so a caller only learns the difference by timing.
 */

#include <base.h>
#include "base/arch.h"

namespace krnl {

// 669: sceKernelAioSubmitRead/WriteCommands[Multiple]. cmd selects read/write
// and whether each request gets its own id.
int PS4ABI sys_aio_submit_cmd(u32 cmd, void *reqs, u32 num, u32 prio, u32 *ids);
// 661: the same submit with the command implied by the caller's entry point.
int PS4ABI sys_aio_submit(u32 cmd, void *reqs, u32 num, u32 prio, u32 *ids);

// 663/664/662/666: wait / poll / delete / cancel, over an array of ids. Each
// writes one SCE error per id.
int PS4ABI sys_aio_multi_wait(u32 *ids, u32 num, int *errs, u32 mode,
                              u32 *usec);
int PS4ABI sys_aio_multi_poll(u32 *ids, u32 num, int *errs);
int PS4ABI sys_aio_multi_delete(u32 *ids, u32 num, int *errs);
int PS4ABI sys_aio_multi_cancel(u32 *ids, u32 num, int *errs);

}  // namespace krnl
