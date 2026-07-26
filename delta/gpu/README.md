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
(`beginFrame` / `draw` / `endFrame`), compute (`dispatch` and the guest-memory
coherency flushes), and `noteMemoryFill` for the CP DMA fills a title uses in
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
into the image for its `rtBase`, and `endFrame` reads back the target at the
scanout address and hands the pixels to the window (or to a PPM, headless).

The heuristic quad path in `vk_draw` predates the recompiler and is still the
fallback for draws `vk_draw_recomp` declines; `DELTA_GPU_DECLINES=1` reports why
draws are still landing there.
