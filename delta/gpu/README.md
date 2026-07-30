# delta/gpu

Turns guest GPU command streams into rendered frames.

```
rhi/            the renderer as its callers see it (command.h, renderer.h)
vulkan/         the only backend implementing it
gcn/            shared ISA decode + the SPIR-V translator both consoles emit through
ps4/            PM4 / Liverpool command processor + its GCN specifics
ps5/            AGC / gfx10.3 command processor + the RDNA2 decoder/emitter
shaders/        prebuilt SPIR-V for the heuristic quad path
guest_memory.h  safe reads of guest memory shared by both command processors
gpu_check.h     GPU_BUGCHECK: always-on fail-fast checks for module invariants
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
| `vk_perf` | where frame time goes, and the on-screen overlay |
| `vk_capture` / `vk_present` | frames out to disk / to the window |

Rendering is offscreen: there is no swapchain on this device. Each draw renders
into the image for its `rt_base`, and `EndFrame` reads back the target at the
scanout address and hands the pixels to the window (or to a PPM, headless).

The heuristic quad path in `vk_draw` predates the recompiler and is still the
fallback for draws `vk_draw_recomp` declines; `DELTA_GPU_DECLINES=1` reports why
draws are still landing there.

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
