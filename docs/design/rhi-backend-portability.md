# RHI backend portability (D3D12 parity) and a testable renderer
Status: partially implemented

The seam, both backends and the recording device exist and are under test
(phases 2, 3's interface half, 4's device, and 6). The renderer itself has not
moved onto them yet: `gpu/vulkan` still calls Vulkan directly, which is phases
0, 1 and the rest of 3. See "Plan" below for what remains.

## Context
`delta/gpu/rhi` is described as the backend-free seam, and at the include level
it is: `check_layering.py` proves no command processor names a Vulkan type. But
the seam is in the wrong place for a second backend.

`rhi::DrawInfo` is a decoded guest draw expressed in *guest* terms: raw
`CB_BLEND0_CONTROL` dwords, `DB_DEPTH_CONTROL`, T# `dfmt`/`nfmt`/`tiling_index`,
`CB_TARGET_MASK`, scissor register pairs. `README.md` states the intent plainly:
"the backend owns every mapping decision". That is exactly what makes a second
backend expensive. Everything between "guest state" and "API call" lives in
`gpu/vulkan`:

- the address to image page table and render-target variant activation
- render-target and depth-target lifetime, footprint math, feedback copies
- the texture cache (upload, revalidation, retirement, view aliasing)
- guest format decode and detiling
- the compute working set, its staging policy and lazy writeback to guest memory
- upload-ring policy, index decoding, pipeline-state derivation
- the heuristic quad fallback and roughly forty `DELTA_GPU_*` diagnostic knobs

None of that is Vulkan. Measured: `delta/gpu/vulkan` is 17831 lines containing
459 Vulkan calls across **82 distinct entry points**. Under today's seam a D3D12
backend must reimplement ~17k lines to reach parity. Under the right seam it has
to implement 82 operations.

The same misplacement is what makes the renderer untestable. Every decision
function takes or returns a `Vk*` type, so exercising it requires a live
`VkDevice`; all state hangs off file-scope reference aliases into one
`g_backend` global (`vk_backend.cc`), so one process can hold one backend and
tests cannot be independent; and nothing can observe what the renderer *decided*,
only what it drew. Of the eleven tests in `gpu/tests`, none touch the renderer:
they cover leaf-pure helpers (detile, decode, memory span, index upload, hazard
predicate). `tools/verify.sh` exists because coverage is ~3%.

Two further facts shape the plan.

Present is already neutral. `gfx/` owns its own `VkInstance`/`VkDevice` and the
gpu-to-window handoff is a CPU pixel buffer (`gfx::present`). There is no
swapchain on the render device, so swapchain interop is not part of this work;
`gfx.h` itself names no API type, only its implementation is Vulkan.

Shaders are not. The recompiler emits SPIR-V with Vulkan descriptor-set
decorations baked in (set 0 textures, set 1 dynamic UBOs, set 2 raw buffers, set
3 LDS scratch), and `gcn_translate.h` reasons about
`maxDescriptorSetStorageBuffersDynamic`. This is the one genuinely new subsystem
D3D12 needs.

## Decision
Insert the missing layer. Keep `DrawInfo`/`ComputeInfo` exactly as they are (the
command processors do not change), and split `gpu/vulkan` into the part that
knows what the guest meant and the part that knows an API.

```
  ps4/  ps5/            command processors                     unchanged
     |                  rhi::DrawInfo / rhi::ComputeInfo       unchanged
  render/               guest render layer, API-agnostic       (from vulkan/)
     |                  rhi::Device / rhi::CommandList         the NEW seam
  vulkan/vk_rhi   d3d12/    rhi/null_device
```

`gpu/render/` owns every decision currently in `gpu/vulkan` that is not a
Vulkan call. It never names a `Vk*` or `ID3D12*` type. `gpu/rhi/` grows from a
submission seam into a real device abstraction, sized by the 82 entry points the
existing code actually uses rather than by what an API offers.

### The RHI vocabulary
Derive it, do not invent it. The 82 calls collapse to 37 operations, 23 on a
command list and 14 on the device:

| group | operations |
|---|---|
| device | `CreateDevice`, caps query, `Submit`, `WaitIdle`, fences |
| buffers | create/destroy, map/unmap, `CopyBuffer`, `FillBuffer` |
| textures | create/destroy, views (incl. format-alias and per-plane views), `CopyBufferToTexture`, `CopyTextureToBuffer`, `ClearColor` |
| binding | `BindGroupLayout`, `BindGroup` (write/update), samplers, root constants |
| pipelines | graphics/compute pipeline from a neutral state description, pipeline-cache blob save/load |
| commands | `BeginRenderPass`/`End`, bind pipeline/bind group/vertex/index, `SetViewport`, `SetScissor`, `SetBlendConstants`, `Draw`, `DrawIndexed`, `Dispatch`, `Barrier`, timestamps, debug labels |

Four decisions inside that surface will otherwise bite later:

1. Resource state rather than image layout. Today `RTarget` tracks
   `VkImageLayout` and `submitted_layout` by hand. Replace it with a neutral
   `rhi::ResourceState` (RenderTarget, ShaderRead, DepthWrite, DepthRead,
   CopySrc, CopyDst, Common), which maps to `VkImageLayout` plus an access mask
   on one side and `D3D12_RESOURCE_STATES` on the other. The backend owns
   barrier batching.
2. A render pass that carries its load and store ops. Vulkan's dynamic
   rendering needs them and D3D12 can honour or ignore them, and the lazy
   `clear_pending` mechanism in `vk_render_target` maps onto `loadOp`
   unchanged.
3. Bind groups with dynamic offsets. Set 1 is dynamic UBOs, which D3D12 has no
   equivalent for, so `BindGroup` carries an offset array: the Vulkan backend
   serves it with dynamic descriptors, the D3D12 backend with root CBVs or
   per-draw descriptor copies out of a ring heap.
4. Host import as a capability rather than an abstraction.
   `host_import_available` (`VK_EXT_external_memory_host`) aliases guest pages
   directly and is why the compute path avoids copying whole allocations. D3D12
   cannot do this; the fallback is the existing staging copy, and the cost is
   real.

### Formats
`vk_format.cc` (885 lines) maps guest encodings straight to `VkFormat`. Split it:
guest to `rhi::Format` (a closed neutral enum, lives in `render/`), then
`rhi::Format` to `VkFormat` / `DXGI_FORMAT` (small per-backend tables). The
neutral enum must be a set *both* backends can serve. DXGI has no three-channel
8/16-bit formats and no swizzled BGRA variants beyond one, so the enum stays
narrow and the texture cache's existing channel-expansion path covers the rest.
A conformance test asserts every neutral format resolves on every built backend.

### Shaders
The recompiler stays SPIR-V. For D3D12, cross-compile: SPIR-V to HLSL
(SPIRV-Cross) to DXIL (DXC), cached on disk beside the existing shader cache.
Prerequisite: stop deriving the pipeline layout from SPIR-V decorations and
derive it from `gcn::Recompiled`'s binding plan, which already carries every
set/binding. Each backend then builds its own layout (Vulkan sets, D3D12 root
signature with four descriptor tables plus root constants) from the same plan,
and the emitted decorations become one backend's detail.

### Testability
The point of the split is that once decisions return neutral data, they are
testable without a GPU. Five things follow:

1. A recording device that implements the full RHI, hands out fake handles,
   records every call into a log and serves readback deterministically.
   `Draw(DrawInfo)` becomes a pure function from guest state to a command log.
2. Golden command-log tests. `DELTA_GPU_CAPTURE` already dumps a whole guest
   frame as JSONL; reuse that as test *input*, replaying a captured frame from a
   real title through the recording device and asserting on the log. This is the
   regression net the renderer has never had, and it catches the class of bug
   `verify.sh` explicitly cannot see (the presented frame is not reproducible;
   see `frame-capture-ab-determinism`).
3. Decision-level unit tests, newly possible: RT footprint and extent math,
   page-table overlap resolution, variant activation, format mapping,
   blend/depth/raster state derivation, texture-cache revalidation and
   retirement, compute resource planning and writeback dirtiness.
4. A backend conformance suite: one binary running identical scenarios against
   every compiled-in backend and comparing readback pixels. It needs a GPU, so
   it is a developer target rather than a CI one, and it is the only thing that
   keeps D3D12 honest against Vulkan.
5. Removing the globals. `Renderer&` is already threaded through the `rhi` entry
   points; finish the job so state comes from the argument rather than
   `g_dev`/`g_frame`/`g_rts`. Two backends in one process (needed for 4) and
   independent tests both depend on this.

`check_layering.py` gains the new rules: `render/` may include `gpu/rhi/`,
`gpu/gcn/` (the same three headers as today) and the module roots, never a
backend; a backend may include `gpu/rhi/` and its own directory; only the
composition root selects a backend.

## Plan
Each phase builds, ships, and is checked with `tools/verify.sh` against a
baseline taken before it. Phases marked *done* are in the tree; the rest are
not started.

*Done, out of order:* the vocabulary (phase 2), the device abstraction and both
backends implementing it (phase 3's interface, phase 6), and the recording
device with tests on top of it (phase 4). Doing the seam and its second backend
first meant the interface was designed against two real APIs rather than
against one API and an intention, and the conformance suite existed before
anything depended on it. What is left is the move of the renderer itself, which
is the large mechanical part.

0. Freeze behaviour. Extend the frame capture to emit a canonical per-draw
record. Take goldens for Isaac (PS4), Isaac (PS5) and one 3D title. Baseline
`verify.sh`. Nothing else in this plan is safe without this.

1. De-globalize. Turn the `g_*` reference aliases into explicit
`BackendState&` parameters. Large mechanical diff, zero behaviour change.

2. Neutral vocabulary. Add `rhi/types.h` (Format, ResourceState, blend and
compare and cull enums, topology, handles). Move guest-to-neutral mapping into
`render/guest_format.cc`, leave neutral-to-`VkFormat` in the backend. Unit tests
for the mapping. No behaviour change.

3. Carve the RHI. Define `rhi/device.h` from the 82
entry points. Implement `rhi/vulkan` as a thin wrapper over today's code. Then
move `gpu/vulkan/*` to `gpu/render/*` one unit at a time, in dependency order:
`memory_span`, `memory`, `upload_ring`, `format`, `index_upload`,
`texture_cache`, `render_target`, `pipeline_cache`, `compute`, `draw_recomp`,
`draw`, `frame`. One commit per unit, `verify.sh` per commit.

4. Null backend and golden tests. Cheap now that the layer is neutral. Get
the renderer under test *before* a second real backend exists, not after.

5. Shader portability. Build layouts from the binding plan instead of the
SPIR-V decorations. Add SPIR-V to HLSL to DXIL behind a caps flag, validated by
compiling the whole on-disk shader corpus we have already collected.

6. D3D12 backend. Device, allocator, resources, command list, pipelines.
Bring up in the order the null backend's log dictates: clear, single quad,
recompiled draw, RT-as-texture, MRT, depth, compute. Conformance suite against
Vulkan at each step.

7. Present. A D3D12 `gfx` backend on Windows, so a D3D12 build does not drag
in Vulkan for the window. Optional zero-copy present is a later, separate change.

## Alternatives
- Write a D3D12 backend under today's seam: rejected because it duplicates ~17k
  lines of guest semantics, and every renderer fix afterwards has to be made
  twice. This is the whole reason the layer is being moved.
- Translate in the command processors, handing the backend an already-neutral
  draw: rejected because it puts graphics-API policy in the PM4/AGC walks, which
  the current design deliberately keeps out of them, and because it is a much
  larger change to files that are under active investigation.
- Emit DXIL directly from the recompiler: rejected for now, because DXIL is LLVM
  3.7 bitcode plus signing and a second emitter is a larger project than the
  whole D3D12 backend, while cross-compilation reuses a translator that is
  already correct for many titles. Revisit if compile time becomes the
  bottleneck (see `shader-hitch-causes`).
- Adopt an existing RHI (Dawn, wgpu-native, bgfx, nvrhi): rejected because guest
  semantics need format aliasing, layout control, host-visible imports and
  descriptor-set shapes that a portable API deliberately does not expose.
- Skip phases 0 to 5 and start with D3D12: rejected because without the
  recording device, the only way to know the D3D12 backend is correct is to boot
  a title and look at it, which is how the current renderer became untestable.

## Consequences
- Phases 1 to 4 carry most of the value and roughly 60% of the work. They are
  worth doing even if D3D12 is never shipped: they are what make the renderer
  testable.
- The D3D12 backend itself is small (order 2-3k lines) once `render/` is
  neutral. The shader path is the schedule risk.
- Every `DELTA_GPU_*` knob and the heuristic quad fallback move to `render/`, so
  they exist once and work on both backends. Any knob that stays in a backend is
  a layering bug.
- Compute-heavy titles will be measurably slower on D3D12 while host import has
  no equivalent.
- Do not run this refactor concurrently with a rendering investigation. Several
  titles are mid-bisect, and `verify.sh` cannot distinguish a refactor
  regression from a moving target.

## Acceptance
Met by what is in the tree:
- `gpu_layering` enforces the backend rule: `d3d12/` may reach `gpu/rhi/` and
  its own directory, and nothing else in the module.
- The conformance suite renders byte-identical readback for Vulkan and D3D12 on
  clear, a covering triangle, clip space orientation, interpolation, depth
  ordering, MRT, a textured quad with a dynamic constant buffer and push
  constants, and a compute dispatch.
- `rhi_null_test` asserts on the recorded command stream with no GPU.
- The existing `ps4delta` target builds and the rest of the suite is unchanged.

Still owed by the remaining phases:
- `gpu/render` exists and contains no `Vk*` or `ID3D12*` identifier.
- The recording device replays a captured Isaac frame and the command log
  matches a checked-in golden.
- A `verify.sh` baseline taken before the move matches after it, and Isaac
  still renders at parity FPS on Vulkan.
