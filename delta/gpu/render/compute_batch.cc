/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute_internal.h"

#include <cstdio>

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/threading/lock_guard.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/guest_page_table.h"
#include "gpu/render/frame.h"
#include "gpu/render/gpu_timeline.h"
#include "gpu/render/render_target.h"
#include "guest/session.h"
#include "profile/profile.h"

namespace gpu::render::cs {

bool g_cs_batch_open = false;
u32 g_cs_batch_count = 0;
u64 g_cs_batch_id = 0;
u64 g_cs_batch_next_id = 1;
u64 g_cs_batch_done = 0;
base::Vector<PendingGuestWrite> g_pending_guest_writes;

void NotePendingGuestWrite(u64 base, u64 end) {
  g_pending_guest_writes.push_back({base, end, g_cs_batch_id});
  GuestPages().AddPendingWrite(base, end, 1);
}
u64 g_publish_batch = 0;
u32 g_cs_batch_cur = 0;
u64 g_cs_submit_n = 0;
base::Vector<BatchedDispatch> g_cs_batch_log;
rhi::TimestampPool* g_cs_timestamps = nullptr;
base::HashMap<u64, CsGpuTime> g_cs_gpu_times;
DELTA_OPTION(u32, kCsHeavyUs, "DELTA_GPU_CS_ISOLATE_HEAVY", 4000);
base::HashSet<u64> g_cs_heavy;
base::HashMap<rhi::Buffer*, ComputeBufferAccess> g_cs_batch_access;

CsBatch g_cs_batches[kCsBatchRing];

void MarkPending(CsRange& e) {
  if (!e.pending_batch)
    g_cs_pending.push_back(&e);
  e.pending_batch = true;
  e.batch_id = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id;
}

bool CsBatchInit() {
  if (g_cs_batches[0].list)
    return true;
  for (auto& b : g_cs_batches) {
    b.list = Device().CreateCommandList();
    if (b.list && kCsListProf)
      b.list->SetProfileTag(2);
    if (!b.list)
      return false;
#if defined(DELTA_TRACY)
    constexpr bool kTimeline = true;
#else
    constexpr bool kTimeline = false;
#endif
    if ((kCsSyncReport || kTimeline) && Device().caps().timestamps)
      b.timestamps = Device().CreateTimestampPool(256);
  }
  g_cs_list = g_cs_batches[0].list;
  return true;
}

void CsBatchLogFault(const CsBatch& b, const char* what) {
  BASE_LOGI("gpuvk", "cs batch DEVICE FAULT: {} failed id={} n={}", what,
            (unsigned long long)b.id, b.count);
  for (const auto& d : b.log) {
    base::String res;
    for (u32 i = 0; i < d.num_res && i < 4; i++)
      base::FormatTo(res, " [{}]{:#x}+{:#x}{}{}", i,
                     (unsigned long)d.res_base[i], (unsigned long)d.res_size[i],
                     (d.res_img >> i) & 1 ? " img" : " buf",
                     (d.res_write >> i) & 1 ? " w" : "");
    BASE_LOGI("gpuvk", "  batched cs={:#x} groups=[{} {} {}] res={}{}",
              (unsigned long)d.cs_addr, d.groups[0], d.groups[1], d.groups[2],
              d.num_res, res.c_str());
  }
  Device().ReportDeviceLoss();
  g_cs_failed = true;
}

void CsBatchFinalize(CsBatch& b) {
  if (b.timestamps && !b.log.empty()) {
    base::Array<u64, 256> stamps{};
    const u32 count = static_cast<u32>(b.log.size());
    const bool any_traced =
        base::AnyOf(b.log.begin(), b.log.end(),
                    [](const BatchedDispatch& d) { return d.traced; });
    const bool read =
        (CsSyncReporting() || any_traced || kCsHeavyUs) &&
        Device().ReadTimestamps(b.timestamps, 0, count * 2, stamps.data());
    if (read && kCsHeavyUs) {
      const rhi::Caps& caps = Device().caps();
      const u64 mask = UINT64_MAX >> (64 - caps.timestamp_bits);
      for (u32 i = 0; i < count; ++i) {
        const double us = ((stamps[i * 2 + 1] - stamps[i * 2]) & mask) *
                          caps.timestamp_period_ns / 1000.0;
        if (us > kCsHeavyUs)
          g_cs_heavy.insert(b.log[i].cs_addr);
        else
          g_cs_heavy.erase(b.log[i].cs_addr);
      }
    }
    for (u32 i = 0; i < count; ++i) {
      if (!b.log[i].traced)
        continue;
      timeline::Time(b.log[i].query, read ? stamps[i * 2] : 0);
      timeline::Time(b.log[i].query + 1, read ? stamps[i * 2 + 1] : 0);
    }
    if (read && CsSyncReporting()) {
      const rhi::Caps& caps = Device().caps();
      const u64 mask = UINT64_MAX >> (64 - caps.timestamp_bits);
      for (u32 i = 0; i < count; ++i) {
        auto& time = g_cs_gpu_times[b.log[i].cs_addr];
        time.ns += ((stamps[i * 2 + 1] - stamps[i * 2]) & mask) *
                   caps.timestamp_period_ns;
        ++time.count;
      }
    }
  }
  g_cs_batch_done = base::Max(g_cs_batch_done, b.id);
  base::EraseIf(g_pending_guest_writes, [](const PendingGuestWrite& w) {
    if (w.batch > g_cs_batch_done)
      return false;
    GuestPages().AddPendingWrite(w.base, w.end, -1);
    return true;
  });
  CheckSpeculatedReads();
  base::EraseIf(g_cs_pending, [](CsRange* range) {
    CsRange& e = *range;
    if (!e.pending_batch)
      return true;
    if (e.batch_id > g_cs_batch_done)
      return false;
    e.pending_batch = false;
    return true;
  });
  b.submitted = false;
  b.log.clear();
  b.count = 0;
  g_cs_writeback_gen++;
}

void CsBatchReap() {
  for (;;) {
    CsBatch* oldest = nullptr;
    for (auto& b : g_cs_batches)
      if (b.submitted && (!oldest || b.id < oldest->id))
        oldest = &b;
    if (!oldest || !Device().IsComplete(oldest->submission))
      return;
    CsBatchFinalize(*oldest);
  }
}

bool g_cs_chunk_needs_batch = false;
u64 g_cs_sync_n[kSyncCount] = {};
u64 g_cs_sync_ns[kSyncCount] = {};
const char* g_cs_wait_reader = "other";
base::Map<base::String, base::Pair<u64, u64>> g_cs_reader_waits;
base::Map<base::String, base::Pair<u64, u64>> g_cs_reader_wbs;

void CsBatchBeginImpl() {
  if (g_cs_batch_open) {
    if (g_cs_batches[g_cs_batch_cur].in_frame)
      EndRegion();
    return;
  }
  if (!CsBatchInit())
    return;
  CsBatchReap();
  CsBatch& b = g_cs_batches[g_cs_batch_cur];
  if (b.submitted) {
    const u64 t0 = NowNs();
    DELTA_ZONE("gpu.cs_ring_full");
    if (!Device().Wait(b.submission))
      CsBatchLogFault(b, "wait");
    CsBatchFinalize(b);
    g_ns_cs_gpu += NowNs() - t0;
    g_cs_sync_n[kSyncRingFull]++;
    g_cs_sync_ns[kSyncRingFull] += NowNs() - t0;
  }
  g_cs_timestamps = b.timestamps;
  b.id = g_cs_batch_next_id++;
  g_cs_batch_id = b.id;
  b.in_frame = g_cs_in_frame && g_frame.recording;
  if (b.in_frame) {
    EndRegion();
    g_cs_list = g_frame.list;
    if (g_cs_timestamps)
      g_cs_list->ResetTimestamps(g_cs_timestamps, 0, 256);
    g_cs_list->Barrier(kAccessAll, kAccessAll);
    g_cs_batch_open = true;
    return;
  }
  g_cs_list = b.list;
  g_cs_list->Begin();
  if (g_cs_timestamps)
    g_cs_list->ResetTimestamps(g_cs_timestamps, 0, 256);
  if (Device().caps().debug_labels) {
    char label[48];
    std::snprintf(label, sizeof(label), "cs batch (frame %llu)",
                  (unsigned long long)g_frame.num);
    g_cs_list->PushLabel(label);
  }
  if (kCsTexBridge) {
    g_cs_list->Barrier(kAccessAll, kAccessAll);
  }
  g_cs_batch_open = true;
}

u64 g_submit_why[6] = {};
base::Mutex g_batch_ids_lock;
base::ConditionVariable g_batch_ids_wake;
SubmittedBatch g_submitted[64];
u64 g_last_submitted_id = 0;

void NoteBatchSubmitted(u64 id, u64 submission) {
  {
    base::LockGuard<base::Mutex> lock(g_batch_ids_lock);
    g_submitted[id % 64] = {id, submission};
    g_last_submitted_id = base::Max(g_last_submitted_id, id);
  }
  g_batch_ids_wake.NotifyAll();
}

bool CsBatchSubmit() {
  DELTA_ZONE("gpu.cs_batch_submit");
  if (!g_cs_batch_open)
    return !g_cs_failed;
  CsBatch& b = g_cs_batches[g_cs_batch_cur];
  if (QueueCheckArmed()) {
    base::String where;
    base::FormatTo(where, "batch n={} last-cs={:#x}", g_cs_batch_count,
                   g_cs_batch_log.empty() ? 0 : g_cs_batch_log.back().cs_addr);
    if (!QueueCheck(where.c_str())) {
      g_cs_failed = true;
      return false;
    }
  }
  if (Device().caps().debug_labels && !b.in_frame)
    g_cs_list->PopLabel();
  b.count = g_cs_batch_count;
  b.log = base::move(g_cs_batch_log);
  g_cs_batch_log.clear();
  const u64 before = Device().LastSubmission();
  if (b.in_frame) {
    g_cs_batch_open = false;
    if (!SubmitFrameChunk()) {
      g_cs_failed = true;
      return false;
    }
    b.submission = g_frame.slots[g_frame.slot_idx].submission;
  } else {
    g_cs_list->End();
    b.submission = Device().Submit(g_cs_list);
  }
  g_cs_batch_open = false;
  g_cs_chunk_needs_batch = false;
  g_cs_batch_count = 0;
  g_cs_batch_access.clear();
  g_cs_submit_n++;
  if (b.submission <= before) {
    CsBatchLogFault(b, "submit");
    return false;
  }
  b.submitted = true;
  MirrorProtectSubmitted(b.submission);
  NoteBatchSubmitted(b.id, b.submission);
  g_cs_batch_cur = (g_cs_batch_cur + 1) % kCsBatchRing;
  return true;
}

bool CsBatchWaitId(u64 id, CsSyncWhy why) {
  DELTA_ZONE("gpu.cs_wait");
  if (g_cs_failed)
    return false;
  if (!id || id <= g_cs_batch_done)
    return true;
  if (g_cs_batch_open && id >= g_cs_batch_id &&
      (++g_submit_why[0], !CsBatchSubmit()))
    return false;
  const u64 t0 = NowNs();
  for (;;) {
    CsBatch* oldest = nullptr;
    for (auto& b : g_cs_batches)
      if (b.submitted && b.id <= id && (!oldest || b.id < oldest->id))
        oldest = &b;
    if (!oldest)
      break;
    if (!Device().Wait(oldest->submission)) {
      CsBatchLogFault(*oldest, "wait");
      return false;
    }
    CsBatchFinalize(*oldest);
  }
  g_ns_cs_gpu += NowNs() - t0;
  g_cs_sync_n[why]++;
  g_cs_sync_ns[why] += NowNs() - t0;
  if (CsSyncReporting()) {
    auto& waits = g_cs_reader_waits[g_cs_wait_reader];
    waits.first++;
    waits.second += NowNs() - t0;
  }
  return !g_cs_failed;
}

bool CsBatchWaitRange(const CsRange& e, CsSyncWhy why) {
  return !e.pending_batch || CsBatchWaitId(e.batch_id, why);
}

bool CsBatchFlush(CsSyncWhy why) {
  const u64 last = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id - 1;
  return CsBatchWaitId(last, why);
}

bool CsLazyKept(u64 base, const CsRange& e) {
  const bool linear_image =
      e.image_staging && gcn::TilingIsLinear(e.res.tiling_idx);
  return kCsImageLazyWb && (e.truth || linear_image) && e.gpu_dirty &&
         !e.imported && !CsAliasedBase(base);
}

base::Vector<u64> DirtyBases(bool ordered) {
  base::Vector<u64> bases;
  bases.reserve(g_cs_dirty_bases.size());
  for (const auto& [base, end] : g_cs_dirty_bases)
    bases.push_back(base);
  if (ordered)
    SortByDirtyOrder(bases);
  return bases;
}

bool CsSplitFrameChunk() {
  if (g_cs_in_frame && g_frame.recording)
    return true;
  if (g_cs_chunk_needs_batch && (++g_submit_why[1], !CsBatchSubmit()))
    return false;
  if (!SubmitFrameChunk())
    return false;
  g_cs_chunk_splits++;
  return true;
}

}  // namespace gpu::render::cs

namespace gpu::render {

bool WaitBatch(u64 batch) {
  u64 submission = 0;
  {
    base::UniqueLock<base::Mutex> lock(g_batch_ids_lock);
    if (!guest::Wait(g_batch_ids_wake, lock,
                     [&] { return g_last_submitted_id >= batch; }))
      return false;
    if (g_submitted[batch % 64].id == batch)
      submission = g_submitted[batch % 64].submission;
  }
  return Device().Wait(submission ? submission : Device().LastSubmission());
}

bool WaitSubmission(u64 submission) {
  return !submission || Device().Wait(submission);
}

void DeferLabelsToGpu() {
  g_labels_deferrable = true;
}

bool WaitPendingGuestWrites(u64 base, u64 bytes) {
  if (GuestPages().PendingWriteClear(base, base + bytes))
    return true;
  u64 wait = 0;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    if (w.base < base + bytes && base < w.end)
      wait = base::Max(wait, w.batch);
  if (!wait)
    return true;
  const WaitReaderScope reader(kMirror ? g_cs_wait_reader : "gpu-guest-write");
  return CsBatchWaitId(wait, kSyncWriteback);
}

u64 GuestWritePublishBatch() {
  u64 batch = 0;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    batch = base::Max(batch, w.batch);
  g_publish_batch = base::Max(g_publish_batch, batch);
  return batch;
}

void ReportBatchState() {
  base::String writes;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    base::FormatTo(writes, " {:#x}+{:#x}@{}", w.base, w.end - w.base, w.batch);
  BASE_LOGW("agc",
            "batches: open={} id={} next={} done={} last_submitted={} "
            "publish={} in_frame={} recording={} guest writes [{} ]",
            (int)g_cs_batch_open, g_cs_batch_id, g_cs_batch_next_id,
            g_cs_batch_done, g_last_submitted_id, g_publish_batch,
            (int)g_cs_batches[g_cs_batch_cur].in_frame, (int)g_frame.recording,
            writes.c_str());
}

u64 SubmitRecordedWork() {
  if (g_cs_batch_open && !CsBatchSubmit())
    return 0;
  if (g_frame.recording && g_frame.draws != g_frame.draws_at_chunk &&
      !SubmitFrameChunk())
    return 0;
  return Device().LastSubmission();
}

void SubmitForPublishedLabels() {
  if (g_cs_batch_open && g_publish_batch >= g_cs_batch_id) {
    ++g_submit_why[2];
    CsBatchSubmit();
  }
}

void RecordComputeInFrame() {
  g_cs_in_frame = kCsInFrame;
}

bool CsBindsImages() {
  return g_cs_in_frame && g_frame.recording && !g_cs_failed;
}

void CsFrameListChanged() {
  if (g_cs_batch_open && g_cs_batches[g_cs_batch_cur].in_frame)
    g_cs_list = g_frame.list;
}

}  // namespace gpu::render

namespace gpu::render::cs {

bool CsBatchOpenOutOfFrame() {
  return g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame;
}

}  // namespace gpu::render::cs

namespace gpu::render {

void CsSubmitBatchBeforeFrame() {
  if (CsBatchOpenOutOfFrame())
    CsBatchSubmit();
}

rhi::CommandList* FrameOrBatchList() {
  if (g_frame.recording) {
    EndRegion();
    return g_frame.list;
  }
  CsBatchBeginImpl();
  return g_cs_batch_open ? g_cs_list : nullptr;
}

}  // namespace gpu::render
