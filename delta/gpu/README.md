# delta/gpu

Turns guest GPU command streams into rendered frames.

```
rhi/            the renderer as its callers see it (command.h, renderer.h), and
                the device abstraction its backends implement (types.h, device.h)
vulkan/         the guest renderer, and the Vulkan RHI backend (vk_rhi)
d3d12/          the D3D12 RHI backend
gcn/            shared ISA decode + the SPIR-V translator both consoles emit through
ps4/            PM4 / Liverpool command processor + its GCN specifics
ps5/            AGC / gfx10.3 command processor + the RDNA2 decoder/emitter
shaders/        prebuilt SPIR-V for the heuristic quad path
guest_memory.h  safe reads of guest memory shared by both command processors
gpu_check.h     GPU_BUGCHECK: always-on fail-fast checks for module invariants
gpu_perf.h      the frame-time counters and clock every unit in the module feeds
tests/          unit tests + the layering check
```

Dependencies run one way: `ps4/` and `ps5/` depend on `gcn/` and `rhi/`, `vulkan/` depends
on `rhi/` (plus the `gcn/` recompiled-program and detile types it consumes),
and `rhi/` includes nothing in this module -- though `command.h` does
forward-declare `gcn::Recompiled`/`gcn::RecompiledCs`, so the seam is
backend-free, not recompiler-free. A command processor decodes guest packets
into an `rhi::DrawInfo` or `rhi::ComputeInfo` and calls the entry points in
`rhi/renderer.h`; it never names a graphics API type, and never includes
anything from `vulkan/`.

The public surface is `rhi/` plus the two `cmd_processor.h` entry headers the
HLE submit paths call (`gpu/ps4/cmd_processor.h`, `gpu/ps5/cmd_processor.h`).
Everything else is internal, so a second backend can be added without touching
a caller. Note this is enforced by `tests/check_layering.py` at test time, not
by the build: every module shares one include root, so an out-of-bounds
include compiles and only `gpu_layering` rejects it.

## rhi/

`command.h` is the contract: one decoded draw or dispatch, expressed in guest
terms (addresses, GCN data/number formats, GNM blend words). It is deliberately
not a "translated" description -- the backend owns every mapping decision, so
both command processors stay free of graphics API policy.

`renderer.h` is the operation set: a `Renderer` value (a struct with a couple
of cheap queries; all backend state hangs off its opaque `BackendState*`)
operated on by free functions -- bring-up (`Init`), the frame lifecycle
(`BeginFrame` / `Draw` / `EndFrame`), compute (`Dispatch` and the guest-memory
coherency flushes), and `NoteMemoryFill` for the CP DMA fills a title uses in
place of a clear packet. `DefaultRenderer()` hands out the process-wide
instance the command processors drive (the guest-called HLE entry points
cannot thread a handle); it is the one piece of ambient state at this seam.

`types.h` and `device.h` are the *device* abstraction: what a graphics API has
to provide for the renderer to run on it, in a vocabulary of its own (formats,
resource states, bind groups, render passes) rather than either API's. It is
sized by what the renderer calls -- 23 command-list operations and 14 device
ones, which is what the 17k lines of `vulkan/` reduce to once the guest
semantics are taken out of them -- so a second backend implements an API rather
than reimplementing the renderer.
`docs/design/rhi-backend-portability.md` is why it looks like this.

Resource state is explicit but tracked: every buffer and texture knows the state
the recording has left it in, so `Transition` to a state it already holds costs
nothing, and for unordered access it is the request to make earlier writes
visible. Command lists must therefore be submitted in the order they were
recorded.

Clip space has +Y up, which is D3D12's convention; the Vulkan backend reaches it
with a negative-height viewport, the same flip the guest renderer already
applies so that a target sampled as a texture lines up with the one it was
rasterized into. Face winding then means the same thing on both, and geometry
needs no per-backend flip.

A binding index is unique within its group across register classes, which is
what lets one number serve as both a Vulkan binding and an HLSL register slot. A
group is a descriptor set on Vulkan and a descriptor table on D3D12; push
constants are root constants at `b0` in space 8.

`null_device.h` is a device that records instead of rendering: every operation
appends a line to a log, so what the renderer decided can be asserted without a
GPU. It is the only way this module's decision-making has ever been testable.

## vulkan/

One unit per decision, roughly in dependency order:

| unit | hides |
|---|---|
| `vk_backend` | the whole backend state as one value behind `rhi::BackendState` |
| `vk_device` | instance/adapter/queue selection, memory types, barriers, shader modules |
| `vk_format` | every guest encoding -> Vulkan mapping (surface, vertex, blend, topology, readback) |
| `vk_hash` | key mixing and the guest-memory content fingerprint |
| `vk_memory_span` / `vk_memory` | aligned free-span suballocation, and device-local image memory pooled with it |
| `vk_index_upload` | guest index decoding (8-bit widened to 16) and the upload element policy |
| `vk_upload_ring` | how per-draw vertices, indices and constants reach the GPU each frame |
| `vk_texture_cache` | guest textures as images: descriptors, upload, revalidation, retirement |
| `vk_render_target` | render targets keyed by guest address, the address -> image page table, the rendering region |
| `vk_pipeline_cache` | which pipeline a given piece of guest state needs |
| `vk_compute` | the GPU-resident compute working set and lazy writeback to guest memory |
| `vk_compute_hazard` | the buffer-hazard predicate deciding when a dispatch batch needs a barrier |
| `vk_draw_recomp` | running the game's own recompiled VS/PS for a draw |
| `vk_draw` | the draw entry point and the heuristic quad fallback |
| `vk_frame` | the two-slot frame ring, readback and presentation of a finished frame |
| `vk_perf` | reporting `gpu_perf.h`'s counters: the FPS line and the on-screen overlay |
| `vk_capture` / `vk_present` | frames out to disk / to the window |
| `vk_rhi` | Vulkan behind `rhi::Device`: its own instance, device and queue, so a test can create one |

Rendering is offscreen: there is no swapchain on this device. Each draw renders
into the image for its `rt_base`, and `EndFrame` reads back the target at the
scanout address and hands the pixels to the window (or to a PPM, headless).

The heuristic quad path in `vk_draw` predates the recompiler and is still the
fallback for draws `vk_draw_recomp` declines; `DELTA_GPU_DECLINES=1` reports why
draws are still landing there.

Everything above `vk_rhi` in that table still calls Vulkan directly. Moving it
onto the device abstraction (and out to a backend-free directory with it) is the
next phase of the plan in `docs/design/rhi-backend-portability.md`.

## d3d12/

`d3d12_rhi` is the second backend: Windows D3D12 code, compiled as such. On
Linux it builds against vkd3d, which implements the same API over Vulkan, so it
is developed and tested on the machine it is written on rather than only on the
one it ships to. The build finds either through `DELTA_HAVE_D3D12`; without it
the factory returns null and nothing else changes.

D3D12 has no dynamic descriptors, so a bind group stages its descriptors in a
CPU heap and `SetBindGroup` copies them into the command list's shader-visible
ring, writing any dynamic binding's descriptor fresh with the per-draw offset
applied. It has no load ops, so a pass that wants a cleared attachment clears it
at `BeginRenderPass`, which is the work the driver does behind Vulkan's
`loadOp`. It has no fill command, so `FillBuffer` clears through an unordered
access view, and the buffer has to carry `kBufferStorage` as well as
`kBufferCopyDst`.

There is no equivalent of `VK_EXT_external_memory_host`, so
`DeviceCaps::host_import` is false here and guest pages are copied rather than
aliased.

## ps4/

The same one-unit-per-decision split, from the packet stream inwards:

| unit | hides |
|---|---|
| `pm4` | PM4 packet framing (shared with `ps5/`, whose AGC streams are PM4-framed) |
| `liverpool` | the Liverpool register file and the offsets worth naming |
| `guest_address` | which addresses a packet may be believed when it points at one |
| `cmd_processor` | the DE/CE walks, the state they latch, and the fence labels they write |
| `draw_state` | how register state plus tracked shader resources become one `rhi::DrawInfo` |
| `compute_dispatch` | how COMPUTE_* registers plus a CS's descriptors become one `rhi::ComputeInfo` |
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
| `draw_state` | how register state plus tracked shader resources become one `rhi::DrawInfo` |
| `compute_dispatch` | how COMPUTE_* registers plus a CS's descriptors become one `rhi::ComputeInfo` |
| `shader_cache` | which recompiled module a given piece of guest state needs |
| `cmd_trace` | the `DELTA_AGC_*` / `DELTA_GPU_*` instrumentation of the command stream |
| `rdna/` | the RDNA2 decoder, descriptor decode and SPIR-V translator (see `ps5/README.md`) |

`reg_state` has no PS4 counterpart because the PS4 has nothing to hide there:
Gnm puts register values in the packet. AGC mostly does not -- it restores
shadow images and submits blocks of (offset, value) entries whose layout is not
documented anywhere and was read back out of the command stream -- so that
guesswork is one unit rather than a third of the walk.

## Testing the RHI

Two tests, with different requirements:

- `rhi_null_test` runs against the recording device and needs no GPU, so it runs
  wherever the rest of the unit tests do. It asserts on the command stream: that
  a repeated transition emits nothing, that a pass transitions its own
  attachments, that every dynamic offset reaches its binding.
- `rhi_conformance_test` runs the same work through every backend the build has
  and compares the results pixel for pixel: clear, a covering triangle, clip
  space orientation, interpolation, depth ordering, MRT, a textured quad with a
  dynamic constant buffer and push constants, and a compute dispatch. Each
  scenario also states what the image should be, so one backend failing alone is
  caught as well as the two disagreeing. It needs a device and skips without
  one.

The shaders are one HLSL file (`tests/rhi_conformance.hlsl`), compiled to SPIR-V
by glslc at build time and to DXBC by `D3DCompile` at run time. A pixel
difference is therefore a backend difference rather than a shader difference.
Without glslc the conformance test is not built at all, rather than quietly
testing one backend.

## Debugging a frame

`DEBUGGER.md` documents the built-in frame debugger: `DELTA_GPU_CAPTURE=<frame>`
records one complete guest frame (every region, draw, dispatch, barrier and
resolved descriptor) as JSONL plus PNG resource dumps, and
`tools/gpu_capture.py` queries it. It needs no capture layer and no GUI, which
is what the `DELTA_GPU_*` printf switches were standing in for.

## Debugging a frame in RenderDoc

Launch the emulator under RenderDoc with `DELTA_RDOC_FRAME=N`: rendering is
offscreen (no swapchain on this device), so vk_frame brackets guest frame N
with an explicit capture instead of relying on a present boundary. With a
capture tool attached `VK_EXT_debug_utils` lights up (`vk_debug`) and the
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

The module follows [Chromium C++ style](https://chromium.googlesource.com/chromium/src/+/main/styleguide/c++/c++.md),
enforced by the local `.clang-format` / `.clang-tidy` (naming) and by
`tests/check_layering.py` (dependencies, registered with CTest as
`gpu_layering`):

- Types `CamelCase`; functions `CamelCase()`; variables, struct members and
  parameters `snake_case` (private class members would take a trailing `_`);
  constants `kCamelCase`; mutable globals `g_snake_case`; macros `UPPER_CASE`.
- Every include is spelled from the delta root (`gpu/vulkan/vk_device.h`),
  including inside the module. No extra include roots.
- Directory dependencies are one-way and machine-checked; the module's public
  surface is `rhi/` plus the two `cmd_processor.h` entry headers. Everything
  else is internal: nothing outside `delta/gpu` may include it.

Deliberate deviations from Chromium:

- (The module uses Chromium's `.cc` extension; the rest of the repo stays
  `.cpp` — the shared `add_delta_module` glob accepts both.)
- Unit tests live in `tests/`, not next to the code (repo-wide
  `add_delta_module`/CTest wiring).
- Hardware mnemonics keep AMD's canonical spelling (`IT_DRAW_INDEX_2`,
  register names) so they can be grepped against cikd.h and the ISA docs;
  such enums carry `NOLINT` guards.
