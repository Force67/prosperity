# delta/gpu

Turns guest GPU command streams into rendered frames.

```
rhi/        the renderer as its callers see it (command.h, renderer.h)
vulkan/     the only backend implementing it
ps4/        PM4 / Liverpool command processor + the GCN -> SPIR-V recompiler
ps5/        AGC / gfx10.3 command processor + the RDNA2 -> SPIR-V recompiler
shaders/    prebuilt SPIR-V for the heuristic quad path
```

Dependencies run one way: `ps4/` and `ps5/` depend on `rhi/`, `vulkan/` depends
on `rhi/`, and `rhi/` depends on nothing in this module. A command processor
decodes guest packets into an `rhi::DrawInfo` or `rhi::ComputeInfo` and calls
the entry points in `rhi/renderer.h`; it never names a graphics API type, and
never includes anything from `vulkan/`.

Only `rhi/` is reachable from outside the module (`gpu/rhi/renderer.h` from the
delta root). `vulkan/` and the platform directories are on the module's private
include path, so a second backend can be added without touching a caller.

## rhi/

`command.h` is the contract: one decoded draw or dispatch, expressed in guest
terms (addresses, GCN data/number formats, GNM blend words). It is deliberately
not a "translated" description -- the backend owns every mapping decision, so
both command processors stay free of graphics API policy.

`renderer.h` is the operation set: bring-up, the frame lifecycle
(`BeginFrame` / `Draw` / `EndFrame`), compute (`Dispatch` and the guest-memory
coherency flushes), and `NoteMemoryFill` for the CP DMA fills a title uses in
place of a clear packet.

## vulkan/

One unit per decision, roughly in dependency order:

| unit | hides |
|---|---|
| `vk_device` | instance/adapter/queue selection, memory types, barriers, shader modules |
| `vk_format` | every guest encoding -> Vulkan mapping (surface, vertex, blend, topology, readback) |
| `vk_hash` | key mixing and the guest-memory content fingerprint |
| `vk_upload_ring` | how per-draw vertices, indices and constants reach the GPU each frame |
| `vk_texture_cache` | guest textures as images: descriptors, upload, revalidation, retirement |
| `vk_render_target` | render targets keyed by guest address, the address -> image page table, the rendering region |
| `vk_pipeline_cache` | which pipeline a given piece of guest state needs |
| `vk_compute` | the GPU-resident compute working set and lazy writeback to guest memory |
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
