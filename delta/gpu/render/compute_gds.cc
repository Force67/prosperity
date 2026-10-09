/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The GDS scratchpad and the guest pages GDS copies land in.

#include "gpu/render/compute_internal.h"

#include <cstring>

#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/texture_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"

namespace gpu::render {

static bool PrepareGdsTransfer(Renderer& renderer,
                               u32 offset,
                               u32 bytes,
                               bool read = false) {
  if (!renderer.available() || offset > GdsBuffer::kBytes ||
      bytes > GdsBuffer::kBytes - offset || !EnsureGdsBuffer())
    return false;
  if (read) {
    CsBatchBeginImpl();
    g_cs_list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                       rhi::kAccessHostRead);
  }
  const WaitReaderScope reader(read ? "gds-read" : "gds-write");
  return CsBatchFlush(kSyncWriteback);
}

bool ReadGds(Renderer& renderer, u32 offset, void* data, u32 bytes) {
  if (!PrepareGdsTransfer(renderer, offset, bytes, true))
    return false;
  std::memcpy(data, static_cast<const u8*>(g_gds.map) + offset, bytes);
  return true;
}

// A dword-aligned GDS write rides the open batch, ordered with the dispatches
// around it, instead of waiting for every one of them to finish so the CPU can
// write the mapping. UC2 resets its exposure counters this way each frame and
// paid for the whole skinning pass with it.
bool RecordGdsWrite(u32 offset,
                    u32 bytes,
                    const void* data,
                    u32 fill_value) {
  if ((offset | bytes) & 3u || !bytes || offset > GdsBuffer::kBytes ||
      bytes > GdsBuffer::kBytes - offset || !EnsureGdsBuffer())
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  if (data)
    g_cs_list->UpdateBuffer(g_gds.buf, offset, bytes, data);
  else
    g_cs_list->FillBuffer(g_gds.buf, offset, bytes, fill_value);
  g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
  return true;
}

bool WriteGds(Renderer& renderer, u32 offset, const void* data, u32 bytes) {
  if (renderer.available() && RecordGdsWrite(offset, bytes, data, 0))
    return true;
  if (!PrepareGdsTransfer(renderer, offset, bytes))
    return false;
  std::memcpy(static_cast<u8*>(g_gds.map) + offset, data, bytes);
  return true;
}

// GDS -> guest memory through the GPU, when labels can wait for it
// (DeferLabelsToGpu): the copy lands in the guest page itself, imported on
// demand, in queue order ahead of whatever reads it on the GPU. Our own CPU
// readers of the bytes wait for the batch (WaitPendingGuestWrites); the
// guest's are held back by the labels that announce them.
base::HashMap<u64, rhi::Buffer*> g_guest_page_imports;
bool g_labels_deferrable = false;

rhi::Buffer* ImportGuestPage(u64 page) {
  auto found = g_guest_page_imports.find(page);
  if (found != g_guest_page_imports.end())
    return found->second;
  const rhi::Caps& caps = Device().caps();
  rhi::Buffer* buf = nullptr;
  if (caps.host_import && caps.host_import_alignment &&
      !(kGuestPageImport % caps.host_import_alignment) &&
      gpu::IsWritableRange(page, kGuestPageImport)) {
    GuestWriteTracker().Release(page, kGuestPageImport);
    rhi::BufferDesc desc;
    desc.size = kGuestPageImport;
    desc.usage =
        rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferCopySrc;
    desc.host_pointer = reinterpret_cast<void*>(page);
    buf = Device().CreateBuffer(desc);
  }
  g_guest_page_imports[page] = buf;  // a failed import is not retried
  return buf;
}

bool CopyGdsToGuest(u32 offset, u64 dst, u32 bytes) {
  const u64 page = dst & ~(kGuestPageImport - 1);
  if (!g_labels_deferrable || g_cs_failed || ((offset | bytes | dst) & 3u) ||
      !bytes || offset > GdsBuffer::kBytes ||
      bytes > GdsBuffer::kBytes - offset || dst + bytes > page + kGuestPageImport ||
      !EnsureGdsBuffer() || CsRangeDirtyOverlapping(dst, bytes))
    return false;
  // A block the GPU owns holds the newest bytes in the mirror: written there.
  if (MirrorOwnsAny(dst, dst + bytes)) {
    if (!GpuWriteGuest(dst, bytes, g_gds.buf, offset))
      return false;
    InvalidateTexRange(dst, bytes);
    g_gds_imported_n++;
    return true;
  }
  rhi::Buffer* target = ImportGuestPage(page);
  if (!target)
    return false;
  CsBatchBeginImpl();
  if (!g_cs_batch_open)
    return false;
  g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW,
                     rhi::kAccessCopyRead | rhi::kAccessCopyWrite);
  g_cs_list->CopyBuffer(target, dst - page, g_gds.buf, offset, bytes);
  g_cs_list->Barrier(rhi::kAccessCopyWrite,
                     kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostRead);
  g_pending_guest_writes.push_back({dst, dst + bytes, g_cs_batch_id});
  NoteGuestWrite(dst, bytes);
  InvalidateTexRange(dst, bytes);
  g_gds_imported_n++;
  return true;
}

bool FillGds(Renderer& renderer, u32 offset, u32 bytes, u32 value) {
  if (renderer.available() && RecordGdsWrite(offset, bytes, nullptr, value))
    return true;
  if (!PrepareGdsTransfer(renderer, offset, bytes))
    return false;
  auto* dst = static_cast<u8*>(g_gds.map) + offset;
  for (u32 i = 0; i < bytes; i++)
    dst[i] = static_cast<u8>(value >> ((i & 3u) * 8));
  return true;
}

}  // namespace gpu::render
