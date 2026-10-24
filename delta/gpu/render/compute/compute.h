/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

bool MirrorUsesImport(rhi::Buffer* buffer);

bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     rhi::Texture* img,
                     rhi::TextureState state,
                     u64* seq,
                     bool compressed = false,
                     u32* supplied_mips = nullptr);

bool CsSupplyTextureFromBuffer(u64 base,
                               const gcn::TextureLayout32& tiled,
                               const gcn::TextureLayout32& linear,
                               u32 w,
                               u32 h,
                               rhi::Texture* img,
                               rhi::TextureState state,
                               u64* seq);

bool DetileTextureOnGpu(u64 base,
                        const gcn::TextureLayout32& tiled,
                        u32 texel_w,
                        u32 texel_h,
                        rhi::Texture* img,
                        rhi::TextureState state,
                        bool is_3d = false,
                        u32 first_mip = 0,
                        bool repack111110 = false);

bool WriteTextureToGuest(rhi::Texture* texture,
                         rhi::TextureState& state,
                         u64 base,
                         const gcn::TextureLayout32& tiled,
                         const gcn::TextureLayout32& linear,
                         bool repack111110 = false);

bool CsClearPredicate(u64 address,
                      u64 expand,
                      bool htile,
                      rhi::Buffer** predicate,
                      u64* offset);

u64 CsBufferRevision(u64 base, u64 bytes);
bool CsSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off);
bool CsGeometryBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset);
bool CsResidentGeometryBuffer(u64 base,
                              u64 bytes,
                              rhi::Buffer** buffer,
                              u64* offset);
bool CsStorageBuffer(u64 base,
                     u64 bytes,
                     rhi::Buffer** buffer,
                     u64* offset,
                     u64 alignment,
                     u64 padded_bytes = 0);
bool CsHoldsGeometry(u64 base, u64 bytes);

void PrepareDirectGuestRead();

rhi::CommandList* FrameOrBatchList();

void ReleaseRetiredCsBuffers();

bool CsRefreshRtFromTruth(u64 base);
void CsFrameListChanged();
// Bracket every submit of a list that may run mirror-owning dispatches.
void CsProtectBeforeSubmit(rhi::CommandList* list);
void CsProtectSubmitted(u64 submission);
// The guest remapped [base, base+bytes); any thread.
void CsGuestRemapped(u64 base, u64 bytes);
void CsSubmitBatchBeforeFrame();

void CsForgetGuestRange(u64 base, u64 bytes);

void CollectGuestWrites();
void CollectGuestWritesForSubmission();

void CsNoteGuestWrites(u64 first, u64 end);

bool QueueCheck(const char* where);
bool QueueCheckArmed();

rhi::Buffer* IndirectArgsBuffer(u64 args, u64* offset);
rhi::Buffer* GuestPredicate(u64* offset);

rhi::Buffer* StageIndirectArgs(rhi::CommandList* list,
                               u64 args,
                               u64 bytes,
                               u64* offset);

}  // namespace gpu::render
