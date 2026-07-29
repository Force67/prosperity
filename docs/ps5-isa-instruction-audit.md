# PS5 Shader ISA Implementation Audit

Date: 2026-07-29

## Scope

This audit compares the PS5 shader implementation in:

- `delta/gpu/ps5/rdna/rdna_decode.cc`
- `delta/gpu/ps5/rdna/rdna_translate.cc`
- `delta/gpu/ps5/rdna/rdna_resource.{h,cc}`
- `delta/gpu/ps5/cmd_processor.cc`
- the shared emitters under `delta/gpu/ps4/gcn/spirv/`
- `tools/rdna_selftest.cpp`

against the supplied generic RDNA2 ISA and SDK 6.00 PS5 documents:

- `/home/captainspark/Downloads/rdna2-shader-instruction-set-architecture.pdf`
- `/home/captainspark/Music/Spec/isaa5/shadercoreisaspec.md`
- `/home/captainspark/Music/Spec/isaa5/shaderinsnref.md`

The instruction-reference TOC contains 542 deduplicated records: 540
instruction, alias, or mnemonic-pattern records and two debugging macros. The
count is not an opcode count because 105 records contain placeholders or
optional suffixes.

Support is counted only when the instruction is decoded with the correct
length and operands, translated with the required architectural side effects,
and does not silently substitute unrelated behavior. Merely reaching a shared
GFX7 emitter is not sufficient.

The Markdown extraction does not include the referenced encoding PNGs. A local
LLVM gfx10 instruction table under
`/home/captainspark/Documents/Projects/aki-clang/llvm-project` was used only to
corroborate numeric opcodes that are present only in those missing images.

## Result

The current path supports a bring-up subset of VS/PS shaders. It is not a
general PS5 ISA implementation. The remediation completed with this audit makes
that subset substantially more trustworthy: reachable unsupported instructions
now reject recompilation instead of returning successful fallback SPIR-V,
decoder stream boundaries are preserved, and resource planning uses the same
reachable instruction set as translation.

The largest coverage holes are:

- all compute execution;
- all DS/LDS/GDS operations;
- all FLAT, GLOBAL, and SCRATCH operations;
- scalar stores, atomics, scratch, utility, and cache operations;
- raw/d16 buffer access, buffer stores, and buffer atomics;
- most image variants and image dimensions;
- most f16, i16, packed integer, mixed-precision, f64, and 64-bit comparison
  operations;
- complete wave, DPP, SDWA, subvector, and NGG semantics.

The major correctness blockers found during the initial review are recorded
below with their remediation status. The remaining coverage gaps are explicit
unsupported cases rather than claimed ISA support.

The RDNA2 PDF follow-up resolved the previously uncertain SMEM size and signed
offset rules and exposed additional gfx10 semantic collisions hidden by direct
GFX7 emitter reuse. The audited path now:

- decodes gfx10.3 DS, VOP3P, VOPC DPP, mandatory literal, ending, and
  reachability rules without inheriting GFX7 lengths or opcode widths;
- uses explicit supported VOP2/VOP3 mappings, including fused FMAC/FMA and
  packed-f16 inline constants, and rejects unmodeled selectors and modifiers;
- preserves NULL SGPR reads, discarded scalar destinations, and SCC/EXEC side
  effects, including 64-bit NULL operands;
- rejects unsupported signed SMEM windows, dynamic buffer offsets, MIMG status
  controls, unsupported dimensions, and resource/binding overflows;
- decodes gfx10.3 compact/full T# descriptors, R128 restrictions, packed
  formats, reserved fields, and 256-byte linear pitch without GFX7 defaults.

These corrections improve trustworthy rejection and the implemented graphics
subset; they do not add DS/LDS, FLAT/GLOBAL/SCRATCH, compute, NGG, or complete
wave semantics.

## Preserved PS5 Behavior

The generic RDNA2 PDF is not a complete PS5 AGC ABI specification. Existing
platform behavior was retained where independently documented or required by
observed PS5 shaders:

- AGC/`OrbShdr` footer-bounded shader decoding and PS5 program-ending markers;
- `S_CALL_B64`, subvector-loop, `S_ENDPGM_SAVED`, and related reachability
  classification, while unsupported executable control flow still fails closed;
- merged-wave/fetch-shader discovery, PS input VGPR seeding, direct vertex-fetch
  planning, and the launch-state EXEC move workaround;
- PS5 resource-table scalar replay and compact R128 image descriptors;
- compatibility color conversion and fallback exports remain identified as
  renderer policy rather than ISA semantics.

## Remediation Status

### 1. Fail-closed translation: fixed

`WarnUnsupported()` sets a thread-local failure flag and prints `rejected` in
`delta/gpu/ps4/gcn/spirv/gcn_spirv.cc:50-79`. The shared GCN path resets and
checks that flag at `gcn_spirv.cc:1323-1352`.

The RDNA facade now resets and checks that flag independently around VS and PS
translation. RDNA-specific dispatch also warns for unsupported families,
opcodes, modifiers, stages, dimensions, and export targets. Consequently, a
warning or a shared-emitter fallback prevents `r.ok` from becoming true.
Regression coverage verifies that an unsupported reachable VOP2 rejects
recompilation while an unreachable unsupported instruction does not.

### 2. EXEC initialization and vector predication: fixed

The translator zero-initializes SGPRs, including EXEC, in
`delta/gpu/ps4/gcn/spirv/translator.h:77-101`. `EmitVopc()` ANDs every compare
result with EXEC at `translate_alu.cc:1052-1081`.

The RDNA body now calls `SeedExec()` and enables `predicate_vector` before both
straight-line and CFG emission. `v_cmpx_*` updates EXEC while leaving VCC
unchanged, compact carry/borrow results are masked by EXEC, storage-image writes
are EXEC-guarded, and the PS kill path observes the resulting EXEC state. The
model is still scalarized to one host invocation rather than a complete wave
mask.

The spec defines compare results as predicate AND EXEC and `v_cmpx_*` as also
replacing EXEC (`shaderinsnref.md:161858-161860` and `162095-162114`).

### 3. Confirmed gfx10 VOP3 opcode collisions: fixed subset

The initial implementation handled `0x30f`, `0x310`, and `0x319` as no-carry
integer add/subtract operations. For gfx10 those opcodes are:

| Opcode | Spec operation | Current behavior |
| --- | --- | --- |
| `0x30f` | `v_add_co_u32` | Emits the sum and scalar carry mask. |
| `0x310` | `v_sub_co_u32` | Emits the difference and scalar borrow mask. |
| `0x319` | `v_subrev_co_u32` | Emits the reverse difference and scalar borrow mask. |

The scalar destination requirement is explicit in the spec at
`shadercoreisaspec.md:18355-18361` and in the operation at
`shaderinsnref.md:205151-205175`. The local LLVM table confirms these opcode
assignments at `VOP2Instructions.td:2298-2301`.

The actual gfx10 no-carry signed operations `v_sub_nc_i32 = 0x376` and
`v_add_nc_i32 = 0x37f` now have explicit mappings. Confirmed direct GFX7-remap
hazards at `0x102`, `0x10d`, `0x11e`, `0x125..0x12b` are explicitly translated
or rejected rather than entering a mismatched shared switch. This is not a
claim that the remaining gfx10 VOP3 opcode space is complete.

### 4. Compute dispatches do not execute: remaining blocker

`HandleDispatch()` reads compute state and optionally prints a decoder census,
but it never translates, submits, or emulates the dispatch. It now emits an
explicit once-per-address warning that the dispatch was skipped.

Effective PS5 compute instruction coverage is therefore zero, including
otherwise translated scalar or vector ALU instructions. This also makes the
conditional `s_barrier` emission at `rdna_translate.cc:848-852` unreachable in
real PS5 compute execution.

## Decoder Findings

### Confirmed findings

| Severity | Finding | Status |
| --- | --- | --- |
| Critical | DPP8 and DPP8FI did not consume their control dword. | Fixed. Source selectors 233 and 234 consume a typed extension dword; translation rejects their unimplemented lane semantics without desynchronizing decode. |
| High | FLAT was misclassified as MUBUF. | Fixed structurally. `Enc::kFlat` preserves the family and translation explicitly rejects it; FLAT/GLOBAL/SCRATCH semantics remain unimplemented. |
| High | NSA=2 and NSA=3 operands were lost. | Fixed. `Inst::raw` retains all five possible MIMG words, and MIMG emission receives all explicit address IDs without borrowing architectural scratch VGPRs. |
| Medium | VOP3P was represented as synthetic VOP3. | Fixed. VOP3P has a distinct family, an eight-bit opcode, and exact `0xCC` prefix validation. |
| Medium | Extension metadata was wrong. | Fixed. Literal, SDWA, DPP16, DPP8, and DPP8FI forms have distinct metadata. |
| Medium | Footer-bounded and footerless termination behavior disagreed. | Fixed. Footer-bounded decoding retains post-`endpgm` words and shared reachability filtering selects executable instructions; footerless scans remain bounded by end markers. |
| Medium | Other executable end markers were not recognized. | Fixed. `s_endpgm_ordered_ps_done` and `s_code_end` terminate translation and the appropriate decode modes. |
| Medium | Truncated multiword instructions were accepted. | Fixed. They become bounded `kUnknown` instructions and therefore reject if reachable. |
| Medium | Mandatory SOPK and VOP2 payload dwords were not all consumed. | Fixed. `s_setreg_imm32_b32`, f32 MADK/FMAK, and f16 FMAK forms retain their second dword; ordinary one-dword VOP2 opcodes do not consume false payloads. |
| Medium | DS used the GFX7 opcode bit range and VOP3P accepted an eighth opcode bit. | Fixed. DS reads bits 24:17, VOP3P enforces its seven-bit opcode plus reserved bit 23, and both have decoder regressions. |
| Medium | VOPC DPP controls and several executable endings/calls were invisible to reachability. | Fixed structurally. DPP controls remain typed extensions and reject translation; calls, subvector loops, indirect flow, and all ending forms terminate or branch conservatively instead of exposing data as instructions. |
| Medium | VOP3 literal detection examined unused fields. | Fixed. Literal consumption uses each opcode's architectural source count; the implicit VOP3 FMAC accumulator is not treated as encoded source 2. |
| Low | An undocumented prefix is treated as EXP. | Constrained documented extension. Prefix `0x31` is accepted only for observed zero-EN NGG null exports; non-null forms remain bounded unknown instructions and reject if reachable. It is not counted as spec-derived coverage. |

### SMEM length and offsets

The RDNA2 PDF confirms that SMEM uses a two-dword encoding and that the
`S_LOAD_*` immediate is signed. The decoder retains the second dword. Static
negative windows and dynamic SOFFSET forms that cannot be represented by the
current UBO model are rejected instead of wrapping to the final cbuffer element.
Observed descriptor-table SOFFSET selection remains supported by draw-time
scalar replay; this is PS5 resource resolution, not general shader-side SMEM.

## Instruction Coverage

The tables below are conservative. "Partial" means at least one operation in
the group has useful handling, but the group is not spec-complete. Any warning
fallback is counted as unsupported; the RDNA facade now rejects translations
that encounter one on a reachable path.

### Scalar ALU

| Spec group | Status | Unimplemented or incorrect instructions/features |
| --- | --- | --- |
| Arithmetic | Partial | `s_mul_hi_i32`, `s_mul_hi_u32`. Core add/sub/carry/borrow, low multiply, `s_addk_i32`, and `s_mulk_i32` exist. |
| Absolute value | Implemented core | `s_abs_i32` and `s_absdiff_i32` have direct cases. |
| Move | Partial | `s_movreld_b32/b64`, `s_movrels_b32/b64`, and `s_movrelsd_2_b32` are unsupported. `s_mov*`, `s_movk`, and conditional moves have cases. |
| Compare/select | Partial | 64-bit bit compares and `s_cmp_<compareOp>_u64` are unsupported. The 32-bit compare, immediate compare, min/max, and cselect core exists. |
| Bitwise logic | Broad core | 32/64-bit NOT, AND, OR, XOR, ANDN2, ORN2, NAND, NOR, and XNOR have shared cases. Their wave-width model is still scalarized. |
| Bit manipulation | Partial | `s_brev_b64`, 64-bit FF0/FF1/FLBIT variants, all `s_bitset0/1_b32/b64`, and `s_lshl<num>_add_u32` are unsupported. |
| Bit-field | Partial | `s_bfe_i64` and `s_bfm_b64` are unsupported. |
| Conversion | Missing | `s_sext_i32_i8`, `s_sext_i32_i16`. |
| EXEC masks | Partial/incorrect | Only a subset of 64-bit saveexec operations reaches the shared emitter; it models EXEC as one bit. All b32 saveexec forms and b32/b64 wrexec forms are missing. |
| Quad masks | Missing/approximate | `s_quadmask_b32/b64`, `s_bitreplicate_b64_b32` are missing; `s_wqm_b32/b64` is an identity approximation. |
| 16-bit pack | Missing | All `s_pack_ll/lh/hh_b32_b16`. Shared cases are gated on `IsaMode::kNeo`, while RDNA instructions remain `kBase`. |

RDNA NULL (`s125`) reads as zero and discards every dword of an encoded scalar
destination while preserving SCC and explicit EXEC side effects. Reserved and
unmodeled scalar source selectors reject translation rather than reading the
zero-initialized private register file.

### Scalar flow and special operations

| Status | Instructions |
| --- | --- |
| Implemented CFG subset | `s_branch`, `s_cbranch_scc0/1`, `s_cbranch_vccz/vccnz`, `s_cbranch_execz/execnz`, ordinary `s_endpgm`. |
| Intentional scheduling no-op | `s_nop`; `s_waitcnt` can be a no-op only while translated memory operations are synchronous. This does not cover wait-counter-dependent future asynchronous operations. |
| Additional executable end markers | `s_endpgm_ordered_ps_done` and `s_code_end` terminate decode/CFG rather than falling through. |
| Unsupported special/control operations | Calls, indirect control flow, subvector loops, sleep/wakeup, cache controls, messages, and mode controls remain unimplemented. Reachable unsupported cases reject translation; recognized waits and scheduling hints remain synchronous no-ops. |
| Missing register behavior | `s_getreg_b32` warns and returns zero; `s_setreg_b32` warns and does nothing; `s_setreg_imm32_b32` is unsupported. |
| Effectively unavailable | `s_barrier` has an SPIR-V path for compute, but PS5 compute dispatch does not execute. |

The mode-control omissions affect the semantics of otherwise implemented
floating-point instructions and cannot be treated as harmless scheduling hints.

### Scalar memory

| Spec group | Status |
| --- | --- |
| `s_load_<dword>`, `s_buffer_load_<dword>` | Partial. Only planned constant-buffer/descriptor reads are lowered. Static bounds, binding limits, unsupported negative windows, and unrepresentable SOFFSET forms fail closed. This is not general scalar memory. |
| `s_store_<dword>`, `s_buffer_store_<dword>` | Missing. |
| `s_atomic_<op>`, `s_buffer_atomic_<op>` | Missing. |
| `s_scratch_load_<dword>`, `s_scratch_store_<dword>` | Missing. |
| `s_memrealtime` | Missing. |
| `s_dcache_inv`, `s_dcache_wb`, `s_dcache_discard`, `s_dcache_discard_x2`, `s_atc_probe`, `s_atc_probe_buffer` | Missing. |

### Vector ALU

| Spec group | Status and principal gaps |
| --- | --- |
| Register moves | Only `v_mov_b32` is reliable. Relative moves and swaps are missing. |
| Lane access | Missing/fail-closed. `v_readfirstlane_b32` is explicitly rejected instead of writing a VGPR. Readlane/writelane handling remains a single-lane approximation only where a verified mapping reaches it. |
| Lane permutation and DPP | Missing. `v_permlane16_b32`, `v_permlanex16_b32`, DPP8, DPP8FI, and DPP16 lane routing/masks/inactive-lane behavior are not implemented. |
| Basic b32 logic | Partial. Basic AND/OR/XOR/NOT/XNOR and local `v_and_or_b32`/`v_or3_b32` paths exist. `v_xor3_b32`, `v_xad_u32`, and many gfx10 VOP3-numbered bitfield/permutation operations are absent or collide with GFX7 meanings. |
| Shift/rotate/bitfield | Partial. Some compact reverse shifts and six local gfx10 three-input operations exist. `v_perm_b32` and much of align, BFE/BFI/BFM, 64-bit shift, and gfx10-exclusive opcode coverage is absent or unverified. |
| Thread-mask operations | Partial. `v_cndmask_b32` now receives initialized EXEC/VCC state and CMPX leaves VCC unchanged while updating EXEC; `v_mbcnt_*` remains a pass-through single-lane approximation. |
| f32 arithmetic/rounding/transcendental | Partial. Common add/sub/mul/min/max, rounding, reciprocal, rsqrt, sqrt, exp/log, sin/cos exist. Supported FMA/FMAC forms emit fused `Fma`, including FMAC's implicit destination accumulator. Divide sequences remain simplified, and legacy/clamp/flag distinctions are rejected or incomplete. |
| f64 | Mostly missing or low-dword approximation. Arithmetic, rounding, field access, transcendental, and compare semantics are not spec-complete. |
| i32/u32 arithmetic | Partial. Compact no-carry u32, corrected carry/borrow VOP3 forms, actual signed no-carry mappings, and selected three-input operations exist. Many min3/max3/med3/multiply-high forms remain absent; known GFX7 numeric collisions are explicitly mapped or rejected. |
| i64/u64 arithmetic | Partial at best. Only narrow shared cases for multiply-add exist; required pair and status behavior is not comprehensively mapped for gfx10. |
| Comparisons | Partial. EXEC initialization and CMPX EXEC side effects are corrected for the supported f32, common i32/u32, and low-half f16 subset. Class compares, f64, i64/u64, i16/u16, and many f16 forms remain missing. |
| Conversions | Partial. A core f32/i32/u32/f16 conversion subset exists. Most packed normalization, f64, i16/u16, packed integer, and saturation conversions are missing. |
| Graphics/SAD/cube | Partial. Some cube and `v_sad_u32` shared cases exist; `v_lerp_u8`, MSAD/MQSAD/QSAD, and the remaining SAD variants are not comprehensively implemented. |
| f16 scalar operations | Mostly missing. The gfx10 VOP1/VOP2/VOP3 f16 arithmetic, rounding, field access, transcendental, and class/compare spaces are not mapped to the Neo emitters. |
| i16/u16 scalar operations | Missing except accidental/fallback behavior. |
| VOP3P packed f16 | Partial. Only `v_pk_fma_f16`, `v_pk_add_f16`, `v_pk_mul_f16`, `v_pk_min_f16`, and `v_pk_max_f16` are emitted. Packed FMA now uses fused `Fma`. |
| VOP3P packed integer/shift | Missing: all listed packed i16/u16 arithmetic and packed shift operations. |
| Mixed precision | Missing: `v_fma_mix_f32`, `v_fma_mixhi_f16`, `v_fma_mixlo_f16`. |

Direct GFX7 opcode reuse remains a risk for scalar and VOP1 operations. VOP2 and
VOP3 now use explicit supported mappings before entering shared emitters, with
regressions for known collisions; future coverage should extend that boundary
rather than broadening numeric fallthrough.

### Vector modifiers and operand modes

| Feature | Status |
| --- | --- |
| SDWA source byte/word selection | Partially applied for VOP1/VOP2. |
| SDWA destination selection | Whole-dword destination is accepted; other selections are rejected. |
| SDWA source abs/neg, CLAMP, and OMOD | Decoded and rejected rather than ignored. Reserved source selector 7 is also rejected. |
| SDWA VOPC select and arbitrary SGPR destination | Explicitly rejected. The spec describes the SD destination at `shadercoreisaspec.md:18412-18434`. |
| DPP8/DPP8FI | Control words and metadata decode correctly; lane semantics are explicitly rejected. |
| DPP16 | Control words and metadata decode correctly; lane permutation, row/bank masks, bound control, and FI are explicitly rejected. |
| VOP3 OMOD | Extracted and passed for supported floating-point operations; rejected for comparisons and unsupported integer forms. |
| VOP3 OP_SEL | Explicitly rejected when set. |
| VOP3P OP_SEL/NEG/CLAMP | Canonical componentwise defaults are accepted; unsupported modifier combinations are rejected. |
| LDS_DIRECT | Explicitly rejected through the unsupported-operand path. |
| Reserved/special selectors | Reserved ranges, trap/system values not modeled by the recompiler, context-invalid SDWA operands, and LDS direct reject instead of becoming zero. Inline constants, NULL, VCCZ, EXECZ, SCC, literals, SGPRs, and VGPRs retain their defined handling. |
| EXEC predication | Enabled for RDNA vector writes. The execution model is still scalarized rather than a complete wave implementation. |

The spec explicitly identifies VOP3 neg/abs, OMOD/CLMP, and OP_SEL at
`shadercoreisaspec.md:18329-18334`, and SDWA/DPP behavior at `18390-18461`.

### Buffer, image, data-share, and flat memory

| Spec family | Status |
| --- | --- |
| `buffer_load_format_<components>` | Partial. Four format-load opcodes can become vertex inputs or UBO reads. Unsupported dynamic SOFFSET, IDXEN, TFE/LDS, cache controls, and non-format opcodes reject. The supported direct-fetch path still does not model general descriptor stride or instancing. |
| `tbuffer_load_format_<components>` | Partial with the same addressing restrictions. Supported formats are allowlisted; narrow typed formats are not unpacked completely. |
| Raw `buffer_load_<dataType>` and d16 loads | Missing. |
| Buffer/tbuffer stores and d16 stores | Missing. |
| `buffer_atomic_<op>` | Missing. |
| `buffer_gl0_inv`, `buffer_gl1_inv` | Missing. |
| `image_load`, `image_load_mip` | Partial for a narrow 2D/2D-array path. |
| `image_store`, `image_store_mip` | Plain store has a narrow 2D/2D-array EXEC-guarded path. The mip form is rejected until mip addressing is implemented. |
| `image_sample` family | Narrow subset. Basic implicit, explicit LOD, LZ, two compare forms, and one offset form exist. Clamp, bias, gradients, many compare variants, offsets, and dimensions are missing and rejected rather than collapsed to ordinary sampling. |
| `image_gather4` family | Only one gather opcode has a path, and LZ is not represented correctly. Other gather/hsize/packed forms are missing. |
| `image_get_resinfo` | Partial. |
| `image_get_lod`, `image_msaa_load` | Missing. |
| `image_atomic_<op>` | Missing. |
| `image_<bvhPtrSize>_intersect_ray` | Missing. |
| DS/LDS/GDS | Entire family missing. No DS case exists in `RdnaEmitInst()`. This includes all reads, writes, exchange, atomics, append/consume, swizzle/permute, GWS, and ordered-count operations listed at `shaderinsnref.md:1292-1488`. |
| FLAT/GLOBAL/SCRATCH | Entire semantic family missing. The decoder now classifies it distinctly and the translator rejects it explicitly. The spec families are listed at `shaderinsnref.md:1512-1573`. |

MIMG rejects A16, D16, TFE, LWE, UNRM, unsupported R128 array use, and dimensions
beyond the implemented 2D/2D-array subset. R128 changes descriptor width and is
limited to compact descriptor resource types. GLC, DLC, SLC, and several
format/depth/storage distinctions still lack complete semantic modeling.
Complete NSA words and explicit address VGPRs are preserved.

### Interpolation and export

| Family | Status |
| --- | --- |
| `v_interp_p1_f32`/`v_interp_p2_f32` | Approximate. P1 is discarded and P2 directly loads a Vulkan varying. Barycentric modes are not modeled. |
| `v_interp_mov_f32` | One selector is handled. Other selector behavior is missing. |
| `v_interp_p1ll_f16`, `v_interp_p2_f16` | Missing. The spec requires VOP3 encoding for f16 interpolation at `shadercoreisaspec.md:18533-18546`. |
| `exp` | Partial. PS MRT0-7 color and MRTZ depth, VS POS0, and PARAM0-31 have paths. POS1-4, PRIM/NGG connectivity, non-depth MRTZ payloads, DONE/valid-mask/WQM behavior, and compressed VS exports are missing. |

The PS export path also changes values by default: it replaces NaNs and clamps
every color to the f16 range at `rdna_translate.cc:655-679`. A PS with no color
export receives an opaque-white MRT0 fallback at `1733-1737`, including a
depth-only PS. These are compatibility fallbacks, not ISA behavior.

## Stage Coverage

| Stage | Effective coverage |
| --- | --- |
| Vertex | Partial conventional VS behavior: selected ALU, cbuffer reads, heuristic vertex fetch, POS0/PARAM export. |
| Pixel | Partial: selected ALU, approximate interpolation, narrow 2D image path, MRT/depth export. |
| Compute | None; dispatches are census-only. |
| NGG/primitive shader | Not implemented as such. Merged code is treated as a VS; PRIM export, sendmsg, DS/LDS exchange, wave compaction, and topology construction are absent. |
| Geometry | No guest GS translation. Only a fixed RECTLIST expansion module is generated. |
| Tessellation/mesh | Not implemented. |

## Resource-Tracking Gaps

Instruction translation and resource discovery can disagree even where an
opcode has an emitter:

- Translation, constant-buffer planning, texture planning, and texture tracking
  now consume a shared reachable-program view, so dead instructions do not
  allocate or shift live bindings.
- `DecodeScalarWrite()` does not cover all scalar destination widths, allowing
  stale descriptor versions (`rdna_resource.h:74-102`).
- Scalar replay has gfx10 opcode-table assumptions that do not consistently
  match the translator.
- Texture format mapping remains a subset. Known linear formats derive pitch
  from element size and align rows to 256 bytes, including non-power-of-two
  RGB32 elements.
- Compact R128 descriptors reject resource types requiring omitted array/depth
  fields. Full descriptors reject reserved word4 bits while preserving the
  gfx10.3 `PITCH_MSB` extension; DCC/write-compressed resources reject because
  the upload path does not decode metadata.
- Only a small swizzle-mode subset is usable; unsupported modes use an invalid
  sentinel so layout construction rejects them instead of detiling as GFX7.
- Tiled mip chains are reduced to mip zero (`rdna_resource.cc:754-768`).
- Resource tracking recognizes only a subset of comparison/gather image
  operations, so bindings can differ from emitted operations.
- Vertex and texture table traversal is bounded and heuristic rather than full
  SGPR dataflow.
- PS shader cache identity includes `SPI_PS_INPUT_ENA`, which changes generated
  fragment input declarations and VGPR seeding.

## Test Audit

`tools/build_rdna_selftest.sh` now uses the checked-in `.cc` source names,
complete include roots, and required shared sources. It runs successfully in
the project development shell.

`delta/gpu/tests/rdna_decode_test.cc` adds focused decoder coverage for DPP8 and
DPP8FI extension consumption, SDWA metadata, distinct VOP3P/FLAT families, DS
opcode bits, exact VOP3P prefix validation, full NSA=3 preservation, bounded
truncation, and the additional end markers. It also verifies mandatory
SOPK/VOP2 payload lengths, FMAC's implicit source count, and the null-only
reserved NGG export prefix.

The standalone self-test now covers fail-closed recompilation, unreachable
unsupported instructions, corrected carry/no-carry mappings, CMPX instruction
handling, exact fused packed and scalar FMA structure, implicit FMAC
accumulation, reserved source rejection, NULL scalar replay, negative SMEM and
SOFFSET-selector rejection,
modifier/PRIM/MIMG control rejection, EXEC-guarded image stores, compact/full
T# validation including DCC rejection, format pitch, and reachable-only
texture binding allocation. It still primarily validates generated SPIR-V
structure and selected opcode
patterns; it does not execute the module against a wave-level reference model.

Missing semantic regression coverage remains for complete SDWA/DPP behavior,
OP_SEL, readfirstlane, scalarized wave operations, execution-level CMPX and
carry-mask values, buffer stores/atomics, DS, FLAT, most image variants,
interpolation modes, export controls, compute, and NGG behavior.

Verification performed for this audit:

```text
nix develop -c cmake --build /tmp/opencode/prosperity-nix-build \
  --target delta_gpu rdna_decode_test gcn_spirv_test -j2
  Passed.

nix develop -c ctest --test-dir /tmp/opencode/prosperity-nix-build \
  --output-on-failure -E '^code_lift_test$' -j2
  Passed all 18 buildable tests.

nix develop -c bash tools/build_rdna_selftest.sh
  Passed all self-test assertions.
```

Full all-target verification is blocked by unrelated baseline issues: the
AArch64 build reaches x86-only `mcontext_t` register fields in `tools/modexec`,
and the registered `code_lift_test` has unresolved `runtime::codeLift` symbols.

## Remaining Recommended Order

1. Continue replacing direct GFX7 opcode reuse with explicit gfx10 semantic
   mappings, with a regression for every known numeric collision.
2. Implement the graphics-critical subset based on real shader census: complete
   scalar/VALU mappings, buffer addressing/formats, image variants, and export
   behavior.
3. Implement DS/LDS/GDS and FLAT/GLOBAL/SCRATCH as dedicated memory-family
   milestones; retain explicit rejection until their addressing and side
   effects are complete.
4. Treat compute and NGG as separate backends/milestones; neither can be made
   correct by adding isolated opcode cases to the current VS/PS facade.
5. Add execution-based wave semantics tests instead of relying only on SPIR-V
   validation and structural opcode inspection.

## Bottom Line

The decoder recognizes most top-level binary families, but the implementation
does not handle most instructions in the supplied PS5 reference. Current useful
coverage remains a small scalar/VALU and graphics-memory VS/PS subset. The
trustworthy-rejection milestone is now in place: unsupported reachable behavior
fails compilation, structural decoder defects are covered by tests, and dead
code no longer perturbs resource bindings. Future work should expand semantics
under that fail-closed contract rather than increasing nominal opcode count.
