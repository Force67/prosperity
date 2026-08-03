# GCN (GFX7 / Liverpool) shader backend

Translates PS4 guest shaders straight to SPIR-V and runs them as Vulkan
pipelines. This is the only shader execution path (no interpreter, no GLSL
intermediate).

## Layout

| File | Role |
| --- | --- |
| `gcn_decode.{h,cpp}` | Instruction decoder: bytecode -> `Program` (flat `Inst` list). Footer-aware length recovery (`OrbShdr`), plus `CachedProgram()` — a hash-validated per-address cache so per-draw analysis never re-decodes. |
| `gcn_disasm.{h,cpp}` | Full GFX7 disassembler (mnemonic tables for every encoding + operand rendering). Debug surface only. |
| `gcn_audit.{h,cpp}` | Shader translation audit: per-instruction fates (translated / unsupported / silently-dropped / dead), run-wide ranked report (`DELTA_GPU_SHAUDIT`), per-shader dump files (`DELTA_GPU_SHDUMP`). |
| `gcn_resource.{h,cpp}` | Descriptor ("sharp") decode: V#/T#/S#, and per-draw tracking of the resources a decoded shader references (`TrackTextures` / `TrackVertexBuffers`). Consumes a `Program`. |
| `gcn_detile.{h,cpp}` | 32bpp texture de-tiling (AddrLib-faithful micro/macro swizzle, mips, arrays). |
| `gcn_translate.{h,cpp}` | Public recompiler facade: `Recompile` (VS+PS) / `RecompileCompute` (CS) + the binding-plan structs the renderer consumes. |
| `spirv/spv_emit.{h,cpp}` | Minimal SPIR-V module builder (types/constants de-duped via packed-integer cache). |
| `spirv/spv_post.{h,cpp}` | SPIRV-Tools optimize (legalize + performance passes) and validation. |
| `spirv/translator.h` | Internal shared translator context (`Translator`, `StageContext`) + emitter declarations. |
| `spirv/translate_alu.cc` | Scalar + vector ALU (SOP1/SOP2/SOPC/SOPK, VOP1/VOP2/VOP3, full VOPC predicate set incl. `cmpx` -> EXEC). |
| `spirv/translate_mem.cc` | Memory ops: graphics cbuffers (SMRD) + MIMG (sample/load/gather/resinfo); compute SSBO model (SMRD/MUBUF/MTBUF/MIMG), LDS (DS). |
| `spirv/gcn_spirv.{h,cpp}` | Per-instruction dispatch, control-flow lowering (while/switch state machine), stage drivers (VS/PS/CS), RECTLIST geometry expansion, entry points. |

## Execution model

One wave lane == one SPIR-V invocation. The register file is modelled as
Private-storage `uint` arrays (`sgpr[128]`, `vgpr[256]`); EXEC is a single
"this lane active" bit, VCC (`sgpr[106]`) a 0/1 scalar. spirv-opt's SSA
rewrite (mem2reg) promotes it all out of memory. Branchy shaders are lowered
to a `while/switch` state machine over basic blocks (handles reducible and
irreducible CFGs); single-basic-block shaders emit straight-line.

Compute models guest memory as one storage buffer per descriptor (the buffer
aliases `[base, base+size)`), the 16 `COMPUTE_USER_DATA` dwords as push
constants, and LDS as a `Workgroup` array sized from `RSRC2.LDS_SIZE`
(128-dword granules). An op the model cannot express (atomics, sampling in
CS, GDS) declines the recompile — the dispatch is skipped loudly rather than
corrupting memory.

## Known gaps (declined or approximated, all warned once via `[gcnspv] UNSUPPORTED`)

- Graphics-stage MUBUF/MTBUF/DS: raw buffer loads in a VS/PS need renderer
  SSBO plumbing (guest-range aliasing like the compute path) — currently
  ignored with a warning. Vertex fetch through the Gnm fetch shader is fully
  supported (that is the common path).
- PS sampler bindings are deduplicated by descriptor identity
  (`PlanMimgBindings`, shared by the recompiler and `TrackTextures`), so a
  shader may reference at most 8 *unique* T#/S# pairs (the renderer's set-0
  layout size). A PS exceeding that declines the recompile — exceeding the
  layout instead would crash driver pipeline creation (seen with PT's FOX
  shaders before the dedupe: 44 MIMG instructions, ≤8 unique descriptors).
- `SPI_PS_INPUT_ENA` ABI VGPR seeding (frag-coord / face / barycentrics in
  v0..) is not modelled; V_INTERP results are read directly from Vulkan
  inputs instead (correct for the interpolate-then-use pattern).
- 64-bit float *arithmetic*: still evaluated on the low dwords. The 64-bit
  **compares** (VOPC f64/i64/u64 and their VOP3 forms) read the register pair
  and are assembled from 32-bit halves, so the module needs neither the
  `Int64`/`Float64` capability nor the device features behind them. Exact for
  register operands; a 64-bit compare against a float inline constant or a
  literal (the ISA gives both special 64-bit meanings — a float inline denotes
  the *double*, a literal occupies bits [63:32]) or carrying a VOP3 neg/abs
  modifier is declined rather than answered wrongly.
- The `_clamp` and `_legacy` transcendentals (`v_log_clamp_f32`,
  `v_rcp_clamp_f32`, `v_rsq_clamp_f32`, `v_rcp_legacy_f32`, `v_rsq_legacy_f32`)
  and `v_min_legacy_f32` / `v_max_legacy_f32` follow the ISA exactly, including
  the DX9 NaN rule (min/max return vsrc1 when either input is NaN) and
  `ClampInfToFltMax` / `ConvertInfToZero` touching only infinities.
- Output modifiers (OMOD) on integer results and on VOPC lane masks are
  *ignored*, which is what the hardware does — not a gap.
- LDS in a **graphics** stage is backed by `Private` storage, one array per
  invocation, because SPIR-V allows `Workgroup` only in a compute (or task/mesh)
  shader. This is exact for per-lane addressing — which is all a fragment shader
  can legally express in Vulkan, and all the Orbis compiler emits here: a spill,
  addressed by `v_mbcnt_{lo,hi}(-1)`, i.e. the lane's own slot. Genuine
  cross-lane LDS in a graphics stage has no lowering and still declines by
  construction, as do DS atomics (Vulkan forbids atomics on a `Private` pointer)
  and any access to the **global** data share (DS word 0 bit 17). Recorded as
  `ds.private` in the audit, since the translator cannot prove the addressing is
  per-lane.
- Buffer/image atomics, `s_getreg`/`s_setreg`: not modelled.
- MTBUF moves raw dwords (exact for 32-bit formats and matched load/store
  round trips; packed-format conversion is not applied).
- Tessellation (HS/DS) and real guest GS stages are not implemented (no title
  exercising them yet). The GS slot is used by the fixed RECTLIST expansion.
- Indirect descriptors: cbuffers whose V# is itself loaded from a table at
  runtime resolve through user data only (`ShaderCbuf.ud_sgpr`).

## References

- AMD Sea Islands ISA: https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
- GFX7 register/enum layouts: `drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_{enum,sh_mask}.h` (Linux)
- shadPS4 (`src/shader_recompiler/`) for opcode tables and semantics
  cross-checks.

## Diagnostics (env vars)

`DELTA_GPU_SHTRACE` (translator log), `DELTA_GPU_SHDIS` (dump first branchy
shaders), `DELTA_GPU_SPIRV` (accept/decline tally), `DELTA_GPU_SPIRV_CFG`
(force the CFG path), `DELTA_GPU_SPIRV_NOOPT` (skip spirv-opt),
`DELTA_GPU_NOKILL` (disable the PS discard lowering), `DELTA_GPU_VSFLIPZ`
(z-sign diagnostic), `DELTA_GPU_CSDUMP` (dump dispatches), `DELTA_GPU_NOCS`
(skip compute), `DELTA_GPU_TRACE`, `DELTA_GPU_TILEHIST`.

### Finding unimplemented / wrongly-implemented translations

- `DELTA_GPU_SHAUDIT=1` (or `=<report-path>`): audits every translated
  shader and prints, at exit, a ranked report: each unsupported/approximated
  op with how many shaders it appears in (fix order = report order), every
  instruction that **silently emitted no SPIR-V** (the "an ignored load is
  silently wrong" class — a non-nop instruction with 0 emitted words and no
  warning), and every declined shader, each with an example
  `stage_hash pc=` to reproduce.
- `DELTA_GPU_SHDUMP=<dir>`: per unique shader writes
  `<stage>_<hash>.txt` (annotated disassembly: real mnemonics + operands,
  per-instruction emitted-word counts, `!UNSUPPORTED` / `!SILENT` / `dead`
  markers, binding plan header), `.gcn` (raw bytecode), and `.spv` (the
  unoptimized module, where every op is preceded by an `OpLine` whose line
  number is the GCN dword pc — `spirv-dis`/RenderDoc show exactly which
  guest instruction produced which SPIR-V, for auditing suspected
  mistranslations by eye).
- SPIR-V validation failures of translator-emitted modules now always log
  (a module we emitted that fails validation is a translator bug, never a
  guest gap).
