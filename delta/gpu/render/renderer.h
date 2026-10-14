#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstddef>
#include "base/arch.h"

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/render/command.h"

namespace gpu::render {

struct BackendState;

struct Renderer {
  BackendState* state = nullptr;

  // NOLINT: cheap accessor
  bool available() const { return state != nullptr; }  // NOLINT
};

bool Init(Renderer& renderer);
void StopPresentation(bool interrupt_present = true);
void ResetSession(Renderer& renderer);

void BeginFrame(Renderer& renderer);
void Draw(Renderer& renderer, const DrawInfo& d);
void EndFrame(Renderer& renderer, u64 scanout_base);

bool Dispatch(Renderer& renderer, ComputeInfo& ci);

void PrebuildComputePipeline(const base::Vector<u32>& spirv,
                             u32 num_res,
                             int guest_memory_binding);

bool FlushCsWrites(Renderer& renderer, const char* why = "all");
bool CsRangeMaybeDirty(u64 base, u64 bytes);
bool FlushCsWritesAt(Renderer& renderer, u64 base, const char* why);
bool CsBindsImages();
void RecordComputeInFrame();
bool FlushCsWritesRange(Renderer& renderer,
                        u64 base,
                        u64 bytes,
                        const char* why = "cp");
bool FlushCsWritesForGpu(Renderer& renderer,
                         u64 base,
                         u64 bytes,
                         const char* why);
bool SettleGuestWrites(u64 base, u64 bytes, const char* why);
bool SpeculateGuestRead(u64 base, u64 bytes);
bool FlushCsWritesFrameEnd(Renderer& renderer, bool writeback);
bool CsTracksGuestWrites();

bool ReadGds(Renderer& renderer, u32 offset, void* data, u32 bytes);
bool WriteGds(Renderer& renderer, u32 offset, const void* data, u32 bytes);
bool FillGds(Renderer& renderer, u32 offset, u32 bytes, u32 value);
bool CopyGdsToGuest(u32 offset, u64 dst, u32 bytes);
void DeferLabelsToGpu();
bool WaitPendingGuestWrites(u64 base, u64 bytes);
u64 GuestWritePublishBatch();
u64 SubmitRecordedWork();
void SubmitForPublishedLabels();
void ReportBatchState();
bool WaitBatch(u64 batch);
bool WaitSubmission(u64 submission);
bool DmaGds(Renderer& renderer,
            u32 src_sel,
            u32 dst_sel,
            u64 src,
            u64 dst,
            u32 bytes,
            u32 data);

u64 CsWritebackGeneration();
bool CsRangeDirtyOverlapping(u64 base, u64 bytes);
bool CanDispatchIndirect(u64 args);

void NoteMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value);

void ApplyMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value);
void ApplyMemoryFill64(Renderer& renderer, u64 base, u64 bytes, u64 value);
bool CanClearMemoryFill64(u64 base, u64 bytes);
void NoteMemoryFill64(Renderer& renderer, u64 base, u64 bytes, u64 value);
bool ApplyMemoryCopy(Renderer& renderer, u64 dst, u64 src, u64 bytes);
bool GuestWriteMaybePending(u64 base, u64 bytes);
bool GuestBytesSettled(u64 base, u64 bytes);
void MirrorNoteGuestWrites(u64 first, u64 end, bool announced);
u64 MirrorCacheBlock(u64 block);
void MirrorRefreshWritten();
void MirrorRefreshCached();
bool GpuIndirectArgs(u64 args, u64 bytes);
bool CanDrawIndirect(u32 prim_type, bool indexed);
bool CanDrawMeshIndirect();
bool BeginGuestPredicate(u64 flag);
void EndGuestPredicate();
void MirrorWritesCollected();

bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size);

void NoteGuestRemap(u64 base, u64 bytes);

Renderer& DefaultRenderer();

base::Vector<base::String> GraphicsBackends();

}  // namespace gpu::render
