/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/gpu_perf.h"

namespace gpu {

u64 g_ns_gpu_exec = 0;
u32 g_gpu_exec_samples = 0;

u64 g_ns_draw = 0, g_ns_end = 0, g_ns_readback = 0, g_ns_tex_up = 0;
u64 g_ns_cs = 0, g_cs_bytes = 0;
u64 g_ns_cs_in = 0, g_ns_cs_gpu = 0, g_ns_cs_out = 0;
u32 g_cs_count = 0;
u64 g_win_draws = 0, g_win_declines = 0;
u64 g_ns_dr_pre = 0, g_ns_dr_pipe = 0, g_ns_dr_tex = 0, g_ns_dr_bind = 0;
u64 g_ns_tex_hash = 0, g_tex_hash_bytes = 0, g_ns_tex_probe = 0;
u64 g_tex_hash_n = 0, g_tex_probe_n = 0, g_ns_tex_lookup = 0, g_tex_lookup_n = 0;
u64 g_ns_tex_set = 0, g_tex_set_n = 0, g_ns_region = 0, g_ns_cs_flush = 0;
u64 g_ns_build_draw = 0, g_build_draw_n = 0;
u64 g_ns_pipe_build = 0, g_pipe_build_n = 0;
u64 g_ns_gfx_present = 0, g_ns_borrow_wait = 0;
u64 g_ns_submit = 0, g_ns_present = 0;
u32 g_tex_ups = 0;
u32 g_cs_stage_n = 0, g_cs_flush_n = 0;
u64 g_cs_stage_bytes = 0;
u64 g_cs_wb_bytes_written = 0, g_cs_wb_bytes_total = 0;
u64 g_fr_draw = 0, g_fr_submit = 0, g_fr_wait = 0, g_fr_present = 0,
         g_fr_tex_up = 0;

}  // namespace gpu
