/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/time/time.h"

namespace gpu {

extern u64 g_ns_draw, g_ns_end, g_ns_readback, g_ns_tex_up;
extern u64 g_ns_cs, g_cs_bytes;
extern u64 g_ns_cs_in, g_ns_cs_gpu, g_ns_cs_out;
extern u64 g_ns_submit, g_ns_present;
extern u64 g_ns_gpu_exec;
extern u32 g_cs_count, g_tex_ups;
extern u64 g_win_draws, g_win_declines;
extern u64 g_ns_dr_pre, g_ns_dr_pipe, g_ns_dr_tex, g_ns_dr_bind;
extern u64 g_ns_tex_refresh, g_tex_refresh_bytes, g_ns_tex_probe;
extern u64 g_tex_refresh_n, g_tex_probe_n, g_ns_tex_lookup, g_tex_lookup_n;
extern u64 g_ns_tex_set, g_tex_set_n, g_ns_region, g_ns_cs_flush;
extern u64 g_ns_build_draw, g_build_draw_n;
extern u64 g_ns_memory_fill, g_memory_fill_bytes, g_memory_fill_n;
extern u64 g_ns_pipe_build, g_pipe_build_n;
extern u64 g_pipe_pending_draws;
extern u64 g_ns_gfx_present, g_ns_borrow_wait;
extern u64 g_ring_vb_bytes, g_ring_ib_bytes, g_ring_cb_bytes, g_ring_raw_bytes;
extern u64 g_index_convert_n, g_index_convert_bytes, g_index_convert_copy_bytes;
extern u64 g_index_scan_n, g_index_scan_bytes, g_ns_index_scan;
extern u64 g_vb_kept_bytes;
extern u64 g_bind_bytes;
extern u32 g_bind_n;
extern u32 g_gpu_exec_samples;
extern u32 g_cs_stage_n, g_cs_flush_n;
extern u64 g_cs_stage_bytes;

extern u64 g_fr_draw, g_fr_submit, g_fr_wait, g_fr_present, g_fr_tex_up;

inline u64 NowNs() {
  return (base::TickClock::NowNs());
}

struct ScopeNs {
  u64 t0;
  u64* acc;
  explicit ScopeNs(u64* a) : t0(NowNs()), acc(a) {}
  ~ScopeNs() { *acc += NowNs() - t0; }
};

}  // namespace gpu
