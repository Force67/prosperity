# gpu/ps5 (AGC / RDNA2 / gfx10.3)

The PS5 GPU is an RDNA2 part (Oberon, gfx10.3-like) driven by the AGC command
format. The AGC command buffer libSceAgc builds is a **PM4 type-3 stream using
the same `IT_` opcode table as the PS4** (verified at runtime), so the packet
walk, the register-latch model, the submit bridge, and the entire Vulkan
renderer (`gpu/vk_render.*`) are reused from the PS4 path. What is genuinely new
for PS5, and lives here:

| File | Role | Mirrors (PS4) |
| --- | --- | --- |
| `agc_regs.h` | gfx10.3 register file + offsets (context 0xA000 / sh 0x2C00 / uconfig 0xC000). | `gpu/ps4/liverpool.h` |
| `cmd_processor.{h,cpp}` | AGC PM4 walk: `SET_*_REG` latch, draw/dispatch decode into `gpu::vk::DrawInfo`, completion labels, flip. | `gpu/ps4/cmd_processor.cpp` |
| `rdna/rdna_decode.{h,cpp}` | RDNA2 instruction decoder -> `gpu::gcn::Program` (shared `Inst` repr). Different encoding prefixes + opcode numbers than GFX7. | `gpu/ps4/gcn/gcn_decode.*` |
| `rdna/rdna_resource.{h,cpp}` | gfx10 128-bit V#/T#/S# descriptor decode + fetch/texture tracking. | `gpu/ps4/gcn/gcn_resource.*` |
| `rdna/rdna_translate.{h,cpp}` | RDNA2 -> SPIR-V: per-instruction dispatch that decodes RDNA2 fields, remaps opcodes to the GFX7-canonical numbers, and calls the **shared** `gpu::gcn` emitters (`EmitVop*`, `EmitSop*`, exports, cbuf). Recompile facade. | `gpu/ps4/gcn/spirv/*` |

## Reuse seam

The SPIR-V backend (`gpu/ps4/gcn/spirv/`) is split so its ALU/memory emitters
(`EmitVop1/2/3`, `EmitVopc`, `EmitSop1/2/c/k`, `EmitMimg`, `EmitCbufSmrd`) take
**pre-decoded operands + a GFX7-canonical opcode**, and the `Translator` /
`StageContext` model is ISA-neutral SPIR-V. RDNA2 keeps the same wave-lane model
(one lane == one invocation; EXEC a single bit; VCC = sgpr[106]) and the same
inline-constant operand encoding (0-255 SSRC/VSRC space). So the port is:

1. an RDNA2 front-end that produces the shared `Inst`/`Program`, and
2. an RDNA2 dispatch that reads RDNA2 field layouts, maps each RDNA2 opcode to
   its GFX7-canonical equivalent, and reuses the shared emitters for semantics.

New RDNA2-specific decode is confined to: the encoding prefixes (VOP3 is
`0b110101` not `0b110100`; SMEM replaces SMRD), the VOP3 word0 field layout
(10-bit op at [25:16], OPSEL[14:11], CLAMP[15]), the SMEM (s_load / s_buffer_load)
encoding, and the gfx10 128-bit descriptor layouts.

## Status

Milestone-0 target: a bound render target + a first triangle / clear color
through `vk_render`. The guest (Isaac, PPSA03311) does not yet submit AGC DCBs
(libSceAgc never registers its GPU context), so the stack is validated in
isolation via `tools/rdna_selftest.cpp` until submission is unblocked.

## Implemented (verified via `tools/rdna_selftest.cpp`)

- RDNA2 3-input integer ALU (address math): `v_add3_u32`, `v_lshl_or_b32`,
  `v_and_or_b32`, `v_or3_b32`, `v_lshl_add_u32`, `v_add_lshl_u32`, native
  `v_add/sub/subrev_i32`.
- MIMG texture sampling (recompiler side): `RdnaPlanMimg` + the shared `EmitMimg`.
- VOP3P packed f16 (`v_pk_mul/add/fma/min/max_f16`).

## Known gaps (loud degradation until implemented)

- MIMG draw-time binding: the cmd processor does not yet resolve the live gfx10.3
  T#/S# descriptors from user data and bind them to `vk_render` (PS4
  `TrackTextures` equivalent), so samples read an unbound texture. Resolve against
  `MimgBindingPlan.binding_srsrc`.
- MIMG `arrayed`/DIM detection and NSA (non-sequential address) sampling.
- VOP3P op_sel/neg/clamp modifiers and packed integer ops.
- f16 VOPC compares (0xC9-0xCE) collide with the GFX7 u32 integer-compare space.
- MTBUF and graphics-stage MUBUF/DS.
- RECTLIST geometry expansion (gfx10 prim type 7 vs vk_render's `primType == 17`).
- AGC input-usage-table-driven vertex fetch (the fetch pointer is currently a
  heuristic read of GS user-data[0..1]).
- gfx10 swizzle-mode de-tiling for tiled render targets / textures.
