/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/time/time.h"

namespace gpu {

// Counters for one fps report interval; ReportFps prints and clears them.
struct GpuStats {
  u64 ns_draw, ns_end, ns_readback, ns_tex_up;
  u64 ns_cs, cs_bytes;
  u64 ns_cs_in, ns_cs_gpu, ns_cs_out;
  u64 ns_submit, ns_present;
  u64 ns_gpu_exec;
  u32 gpu_exec_samples;
  u32 cs_count, tex_ups;
  u64 win_draws, win_declines;
  u64 ns_dr_pre, ns_dr_pipe, ns_dr_tex, ns_dr_bind;
  u64 ns_tex_refresh, tex_refresh_bytes, ns_tex_probe;
  u64 tex_refresh_n, tex_probe_n, ns_tex_lookup, tex_lookup_n;
  u64 ns_tex_set, tex_set_n, ns_region, ns_cs_flush;
  u64 ns_build_draw, build_draw_n;
  u64 ns_memory_fill, memory_fill_bytes, memory_fill_n;
  u64 ns_pipe_build, pipe_build_n;
  u64 pipe_pending_draws;
  u64 ns_gfx_present, ns_borrow_wait;
  u64 ring_vb_bytes, ring_ib_bytes, ring_cb_bytes, ring_raw_bytes;
  u64 index_convert_n, index_convert_bytes, index_convert_copy_bytes;
  u64 index_scan_n, index_scan_bytes, ns_index_scan;
  u64 vb_kept_bytes;
  u64 bind_bytes;
  u32 bind_n;
  u32 cs_stage_n, cs_flush_n;
  u64 cs_stage_bytes;
  u64 ns_recomp;
  u32 recomp_n;
  u64 ns_spv_opt;
  u32 spv_hit_n, spv_miss_n;
  u64 ns_dcb, ns_dcb_lock;
};
extern GpuStats g_stats;

// Where one frame's wall time went; the perf overlay samples and clears it.
struct FrameStageNs {
  u64 draw, submit, wait, present, tex_up;
};
extern FrameStageNs g_frame_ns;

#if defined(__aarch64__)
// The render thread times thousands of spans a frame; a vDSO clock_gettime
// per call cost ~5% of it. CLOCK_MONOTONIC on arm64 is derived from the same
// virtual counter, so reading the counter directly and adding the offset
// measured once keeps both clocks comparable.
struct CounterClock {
  u64 mul = 0, offset = 0;
  CounterClock() {
    u64 freq, count;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    mul = (u64(1000000000) << 32) / freq;
    const u64 now = base::TickClock::NowNs();
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(count));
    offset = now - Scale(count);
  }
  u64 Scale(u64 count) const {
    return static_cast<u64>((static_cast<unsigned __int128>(count) * mul) >>
                            32);
  }
};

inline u64 NowNs() {
  static const CounterClock clock;
  u64 count;
  asm volatile("isb; mrs %0, cntvct_el0" : "=r"(count));
  return clock.Scale(count) + clock.offset;
}
#else
inline u64 NowNs() {
  return (base::TickClock::NowNs());
}
#endif

struct ScopeNs {
  u64 t0;
  u64* acc;
  explicit ScopeNs(u64* a) : t0(NowNs()), acc(a) {}
  ~ScopeNs() { *acc += NowNs() - t0; }
};

}  // namespace gpu
