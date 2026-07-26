# PS5 RDNA resource resolution
Status: implemented

## Context
PS5 shaders can select descriptor-table entries with an SMEM `SOFFSET` SGPR,
reuse the same destination SGPRs for different descriptors, and expose up to 32
user-data dwords through the command processor's register image. The RDNA path
currently records only each load's immediate offset, keys constant buffers by
their base SGPR, and defaults pixel user data to 16 dwords. MIMG lowering also
reads the obsolete GCN `DA` bit instead of RDNA's `DIM` field. These mismatches
resolve valid descriptors from the wrong table entry and can bind stale or
malformed textures.

## Decision
Decode RDNA SMEM fields in one shared helper. Replay scalar moves, integer ALU,
and scalar loads in instruction order when resolving live texture and constant-
buffer descriptors. Seed only the user SGPR window declared by the stage, at the
stage's shader-visible SGPR base. Include both `SOFFSET` and `OFFSET` in host
resolution and emitted UBO indices. Dynamic offsets use a shared 16 KiB UBO
window in shader declarations, draw-time validation, and Vulkan uploads.

Carry all 32 stage user-data dwords as per-stage Vulkan push constants. Seed GS
user data at shader SGPR `s8` and pixel user data at `s0` before executing the
translated program, so scalar address arithmetic sees the same values as
draw-time descriptor replay.

Plan SMEM constant-buffer bindings by the descriptor producer as well as its
SGPR, and retain the consuming instruction PC so draw-time scalar replay can
bind the matching live descriptor. Use RDNA MIMG `DIM` to distinguish 2D from
2D-array accesses before calling the shared GCN emitter. Reject T# descriptors
whose base or decoded geometry is invalid. Before reading guest texture memory,
reject decoded footprints that are not fully mapped and readable; live render-target
resolution does not dereference guest pixels and is exempt from this check.

For inline vertex fetches, retain each MUBUF/MTBUF consumer PC in the attribute
plan and resolve its complete V# from scalar replay at that PC. Use the static
table-slot mapping only for standalone fetch shaders, whose instructions are not
part of the main vertex program replay. Seed the vertex replay's merged-wave-info
`s3` with the same modeled value used by translated vertex shaders, and replay
the scalar bitfield, shift-add, comparison, and conditional-select operations
used to derive and patch vertex descriptors.

Apply T# `DST_SEL` component mapping to uploaded textures and sampled live
color, feedback, and depth targets. A live image is the same resource regardless
of component mapping, so cache additional Vulkan views rather than copying the
image or changing its attachment view.

Use the pixel shader's declared user-SGPR count directly, clamped to the 32-dword
register image. Do not retain the 16-dword diagnostic override.

## Alternatives
- Keep static load provenance and add the captured `s106` offset: rejected
  because any SGPR can provide `SOFFSET` and descriptor destinations are reused.
- Keep `DELTA_GPU_UDBOUND` as a default workaround: rejected because it discards
  valid inline descriptors declared by the shader.
- Assign vertex table slots by SMEM producer order: rejected for inline fetches
  because runtime `SOFFSET` selects the descriptor entry.
- Fork the full MIMG emitter for RDNA: rejected because translating `DIM` into
  the shared emitter's arrayed input fixes the supported 2D cases directly.

## Consequences
Resource resolution performs a small scalar replay per draw and can read mapped
descriptor tables while doing so. Constant-buffer plans carry a consuming PC.
Inline vertex attributes also carry a consuming PC and use the same replay,
while standalone fetch shaders retain their existing static table mapping.
Vertex replay shares the translator's modeled merged-wave launch value rather
than reconstructing hardware wave occupancy at draw time.
Unsupported image dimensions remain limited by the shared 2D emitter, but 2D
and 2D-array shaders no longer alias due to an unrelated instruction bit.

## Acceptance
- A dynamic SMEM `SOFFSET` selects the expected T# table entry.
- Scalar immediates update replayed `SOFFSET` values, while unsupported scalar
  writes invalidate prior values instead of reusing stale user data.
- Inline descriptors at `s16` resolve when the PS declares at least 24 user SGPRs
  and do not resolve when it declares only 16.
- Constant-buffer replay respects a shader-visible user-data base of `s8`.
- Dynamic SMEM `SOFFSET` selects the expected V# for each inline vertex fetch,
  even when fetch consumers are scheduled in a different order.
- Table-loaded MUBUF and MTBUF vertex fetches both retain their consumer PC.
- Vertex replay derives a table offset through `s_bfe_u32` from the modeled
  merged-wave-info `s3` value.
- Standalone fetch-shader attributes continue to resolve through their static
  table-slot plan when their pointer and descriptors fit in the declared user
  SGPR window.
- Emitted VS and PS code read the live 32-dword stage user-data windows.
- Emitted stages seed no push-constant SGPRs beyond their declared user-data
  count.
- Replayed constant buffers are bound only when their full byte range is readable,
  including dynamic offsets beyond the first 1 KiB.
- MIMG `DIM=1` and `DIM=5` produce distinct non-array and array bindings.
- Sampled live targets use the T# component mapping, including depth replication
  and BGRA selection. An all-zero mapping remains distinct from identity.
- Low T# base addresses, reversed mip ranges, and out-of-range array views are
  rejected. In-range unmapped bases remain eligible for live-target resolution.
- Texture upload rejects decoded footprints that cross into unmapped or
  `PROT_NONE` memory.
- The RDNA self-test builds, passes, and emits valid SPIR-V.
