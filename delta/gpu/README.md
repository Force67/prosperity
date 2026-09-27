# delta/gpu

Turns guest GPU command streams into rendered frames.

```
render/         the guest renderer, on the rhi alone (public: command.h, renderer.h)
rhi/            the device abstraction backends implement (types.h, device.h)
vulkan/         rhi::Device on Vulkan (SPIR-V consumed as is)
opengl/         rhi::Device on OpenGL 4.6 (headless EGL, SPIR-V lowered to GLSL)
d3d12/          rhi::Device on Direct3D 12 (vkd3d's D3D12 on Linux, SPIR-V lowered to HLSL)
gcn/            shared ISA decode + the SPIR-V translator both consoles emit through
ps4/            PM4 / Liverpool command processor + its GCN specifics
ps5/            AGC / gfx10.3 command processor + the RDNA2 decoder/emitter
shaders/        prebuilt SPIR-V for the heuristic quad path
guest_memory.h  safe reads of guest memory shared by both command processors
guest_page_table.h  per-page state of guest memory, directly indexed
gpu_check.h     GPU_BUGCHECK: always-on fail-fast checks for module invariants
gpu_perf.h      the frame-time counters and clock every unit in the module feeds
write_tracker.h which guest pages the CPU wrote (userfaultfd write protection)
tests/          unit tests + the layering check
```

Dependencies run one way: `ps4/` and `ps5/` depend on `gcn/` and `render/`;
`render/` depends on `rhi/` (plus the `gcn/` recompiled-program, resource and
detile types it consumes); each backend depends on `rhi/` alone and knows
nothing of guests. `render/backend.cc` is the one unit that names the
backends, through their factory headers. A command processor decodes guest
packets into a `render::DrawInfo` or `render::ComputeInfo` and calls the entry
points in `render/renderer.h`; nothing above `rhi/` names a graphics API type.

SPIR-V is the shader IR throughout: the recompiler emits it, Vulkan consumes
it, and the other backends lower it (SPIRV-Cross) when a pipeline is created.
`DELTA_GPU_BACKEND` (`vulkan`, `opengl`, `d3d12`) picks the backend at start.

The public surface is `render/` plus the two `cmd_processor.h` entry headers the
HLE submit paths call (`gpu/ps4/cmd_processor.h`, `gpu/ps5/cmd_processor.h`).
Everything else is internal. Note this is enforced by `tests/check_layering.py` at test time, not
by the build: every module shares one include root, so an out-of-bounds
include compiles and only `gpu_layering` rejects it.

## render/

`command.h` is the contract: one decoded draw or dispatch, expressed in guest
terms (addresses, GCN data/number formats, GNM blend words). It is deliberately
not a "translated" description: the backend owns every mapping decision, so
both command processors stay free of graphics API policy.

`renderer.h` is the operation set: a `Renderer` value (a struct with a couple
of cheap queries; all backend state hangs off its opaque `BackendState*`)
operated on by free functions: bring-up (`Init`), the frame lifecycle
(`BeginFrame` / `Draw` / `EndFrame`), compute (`Dispatch` and the guest-memory
coherency flushes), and `NoteMemoryFill` for the CP DMA fills a title uses in
place of a clear packet. `DefaultRenderer()` hands out the process-wide
instance the command processors drive (the guest-called HLE entry points
cannot thread a handle); it is the one piece of ambient state at this seam.

Behind it, one unit per decision, roughly in dependency order:

| unit | hides |
|---|---|
| `renderer_state` | the whole renderer state as one value behind `render::BackendState` |
| `backend` / `device` | backend choice and bring-up, the frame's command list, checkpoints |
| `guest_format` | every guest encoding -> `rhi` mapping (surface, vertex, blend, topology) |
| `hash` | key mixing and the guest-memory content fingerprint |
| `guest_memory_table` | which guest pages are imported as GPU memory |
| `index_upload` | guest index decoding (8-bit widened to 16) and the upload element policy |
| `upload_ring` | how per-draw vertices, indices and constants reach the GPU each frame |
| `texture_cache` | guest textures: descriptors, upload, revalidation, retirement |
| `render_target` | render targets keyed by guest address, the address -> image page table, the rendering region |
| `pipeline_cache` | which pipeline a given piece of guest state needs |
| `compute` | the GPU-resident compute working set and lazy writeback to guest memory |
| `compute_hazard` | the buffer-hazard predicate deciding when a dispatch batch needs a barrier |
| `draw_recomp` | running the game's own recompiled VS/PS for a draw |
| `draw` | the draw entry point and the heuristic quad fallback |
| `frame` | the two-slot frame ring, readback and presentation of a finished frame |
| `perf` | reporting `gpu_perf.h`'s counters: the FPS line and the on-screen overlay |
| `labels` / `trace` | debug labels and names for capture tools, the barrier and message trace |
| `capture` / `present` / `png` | frames out to disk / to the window |

Rendering is offscreen: there is no swapchain on the device. Each draw renders
into the image for its `rt_base`, and `EndFrame` reads back the target at the
scanout address and hands the pixels to the window (or to a PPM, headless).

The heuristic quad path in `draw` predates the recompiler and is still the
fallback for draws `draw_recomp` declines; `DELTA_GPU_DECLINES=1` reports why
draws are still landing there.

## rhi/

A Vulkan-shaped device interface: explicit texture states and barriers, bind
groups with dynamic offsets and push layouts, push constants, dynamic
rendering, and submission ids that retire in order. `Caps` reports what a
backend can do; the renderer checks it rather than the backend.
`tests/rhi_conformance_test` runs every scenario against each compiled backend.

## vulkan/

| unit | hides |
|---|---|
| `vk_rhi_device` | instance/device selection, features, memory, resources, pipelines, submission |
| `vk_rhi_command` | command lists: barriers, rendering, binding, copies, draws |
| `vk_rhi_format` | `rhi::Format` <-> Vulkan |
| `vk_memory_span` | aligned free-span suballocation of pooled image memory |

## opengl/

An `rhi::Device` on desktop GL 4.6 core, built when libepoxy and glvnd's
libEGL are found (`DELTA_GPU_OPENGL`).

- Threads: a render thread owns the context that replays command lists and
  the objects GL does not share (framebuffers, vertex arrays, queries); a
  resource thread creates buffers, textures and samplers, compile threads
  link programs, and a waiter thread retires the per-submission fences, all
  on shared contexts. A command list records a CPU-side stream with every
  binding resolved to GL names and slots.
- Conventions match Vulkan: row 0 is the top in memory, clip z is [0, 1], and
  a negative viewport height (y-up) becomes an upper-left clip origin, with
  the front face flipped to keep the framebuffer-space winding.
- Shaders: `gl_shader_lowering` turns SPIR-V into GLSL 4.60 with SPIRV-Cross.
  Each program gets slots only for the resources it uses; push constants are
  a flattened uniform array; uniform blocks std140 cannot express become
  read-only storage blocks; storage buffers past the per-stage limit are read
  through GPU addresses (`GL_NV_shader_buffer_load`).
- `tests/rhi_lowering_corpus` runs the lowering over the recompiler's SPIR-V
  cache and compiles every module with the driver. The driver's compiler can
  take several GB on one large shader, so the tool defaults to one thread.

## d3d12/

`rhi::Device` on Direct3D 12, written as Windows D3D12 and built on Linux
against vkd3d (D3D12 over Vulkan), which is where `rhi_conformance_test` runs
it. Optional: `DELTA_GPU_D3D12`, on when vkd3d and DXC are found.

| unit | hides |
|---|---|
| `d3d12_device` | device, queue and fence, resources, views, samplers, bind groups |
| `d3d12_pipeline` | root signatures, pipeline states, the shader cache, the blit pass |
| `d3d12_command` | command lists: state tracking, binding, copies, clears, draws |
| `d3d12_shader` | SPIR-V -> HLSL (SPIRV-Cross) and its fix-ups, HLSL -> DXIL (`d3d12_dxc`) |
| `d3d12_spirv_patch` | SPIR-V rewrites for what SPIRV-Cross cannot express in HLSL |
| `d3d12_format` | `rhi::Format` -> DXGI |

What differs from Vulkan and how it is bridged:

- Viewports: a D3D12 viewport is Vulkan's y-up one. A negative Vulkan height
  maps straight across; a positive one flips clip-space y in the last vertex
  stage through a root constant. Screen positions, and so facing, match.
- Binding: one root signature per pipeline layout, a descriptor table (plus a
  sampler table) per group with register = binding and space = set, dynamic
  uniform buffers as root CBVs, push constants as root constants. Groups are
  copied into a shader-visible ring owned in chunks by each command list.
- State: textures take the caller's before/after states; buffers are tracked
  per command list from COMMON, since they decay there between
  `ExecuteCommandLists` calls and each list is executed in its own.
- Stage linkage: D3D12 matches signatures by register, so a consumer's inputs
  are rewritten to the producer's element order.
- Below shader model 6.1 barycentrics come from a generated geometry shader;
  below 6.7 integer textures are sampled by gather. `tests/rhi_hlsl_corpus`
  runs the recompiler's SPIR-V cache through all of this.

## ps4/

The same one-unit-per-decision split, from the packet stream inwards:

| unit | hides |
|---|---|
| `pm4` | PM4 packet framing (shared with `ps5/`, whose AGC streams are PM4-framed) |
| `liverpool` | the Liverpool register file and the offsets worth naming |
| `guest_address` | which addresses a packet may be believed when it points at one |
| `cmd_processor` | the DE/CE walks, the state they latch, and the fence labels they write |
| `draw_state` | how register state plus tracked shader resources become one `render::DrawInfo` |
| `compute_dispatch` | how COMPUTE_* registers plus a CS's descriptors become one `render::ComputeInfo` |
| `shader_cache` | which recompiled module a given piece of guest state needs |
| `cmd_trace` | the `DELTA_GPU_*` instrumentation of the command stream |

`cmd_processor.h` is the only header of these the rest of the emulator may
include; the walk hands each packet to the unit that owns the decision it
carries and holds no decoding of its own beyond the framing.

`cmd_trace` is deliberately a wide, shallow surface (one function per knob,
each taking exactly what it reports on). Keeping it out of the decode path is
what lets the units above read as the decisions they make rather than as the
probes that were needed to find them; the frame debugger below is the better
tool for anything that fits in one frame.

Its output goes through `BASE_LOGI` on a channel named after the tag the probe
has always printed (`[drawpkt]`, `[csres]`), so lines grep as before,
`base::SetChannelMinLevel` can silence one probe, and delivery is the async
sink `utl::routeBaseLogging` installs at startup. Tracing on the submit thread
with `fprintf` distorted the frame timings the traces exist to explain.

## ps5/

The same split, one unit per decision. AGC command buffers are PM4-framed with
the PS4's `IT_` opcode table, so `pm4` is shared and the walk is the PS4 walk;
everything below it is gfx10.3 and RDNA2:

| unit | hides |
|---|---|
| `agc_regs` | the gfx10.3 register file and the offsets worth naming |
| `guest_address` | the two address windows a packet field may be believed in |
| `cmd_processor` | the AGC walk, the state it latches, the fence labels it writes |
| `reg_state` | what a register write arriving through guest memory means |
| `draw_state` | how register state plus tracked shader resources become one `render::DrawInfo` |
| `compute_dispatch` | how COMPUTE_* registers plus a CS's descriptors become one `render::ComputeInfo` |
| `shader_cache` | which recompiled module a given piece of guest state needs |
| `cmd_trace` | the `DELTA_AGC_*` / `DELTA_GPU_*` instrumentation of the command stream |
| `rdna/` | the RDNA2 decoder, descriptor decode and SPIR-V translator (see `ps5/README.md`) |

`reg_state` has no PS4 counterpart because the PS4 has nothing to hide there:
Gnm puts register values in the packet. AGC mostly does not: it restores
shadow images and submits blocks of (offset, value) entries whose layout is not
documented anywhere and was read back out of the command stream, so that
guesswork is one unit rather than a third of the walk.

## Debugging a frame

`DEBUGGER.md` documents the built-in frame debugger: `DELTA_GPU_CAPTURE=<frame>`
records one complete guest frame (every region, draw, dispatch, barrier and
resolved descriptor) as JSONL plus PNG resource dumps, and
`tools/gpu_capture.py` queries it. It needs no capture layer and no GUI, which
is what the `DELTA_GPU_*` printf switches were standing in for.

## Debugging a frame in RenderDoc

Launch the emulator under RenderDoc with `DELTA_RDOC_FRAME=N`: rendering is
offscreen (no swapchain on the device), so `render/frame` brackets guest
frame N with an explicit capture instead of relying on a present boundary.
With a capture tool attached debug labels light up (`render/labels`) and the
capture is self-describing, everything keyed by guest addresses so it lines up
with the `DELTA_GPU_*` logs:

- The event browser nests `frame N` > `region rt=... / cs batch` >
  `recomp vs=... ps=... / quad ... / dispatch cs=...` markers.
- Resources are named after what they cache: `rt 0x... WxH`, `depth 0x...`,
  `tex 0x...`, `csbuf 0x...`, the upload rings, pipelines by shader address.
- The recompiled SPIR-V carries `OpName`s (`sgpr`/`vgpr`, `cbufN`, `texN`,
  `bufN`, `user_data`, `v_attrN`, `in_attrN`/`out_paramN`, `mrtN`, `lds`), so
  the shader viewer reads like the GCN it came from rather than anonymous ids.

## Conventions

The module follows the repo-wide [conventions](../../docs/conventions.md),
plus `tests/check_layering.py` (registered with CTest as `gpu_layering`):
directory dependencies inside the module are one-way, and its public surface
is `render/` plus the two `cmd_processor.h` entry headers. Everything else is
internal: nothing outside `delta/gpu` may include it.
