# The GPU frame debugger

One armed guest frame, recorded completely, into one machine-readable file.

The `DELTA_GPU_*` printf switches each answer one question, cost a rebuild when
the question changes, and cap themselves at an arbitrary line count. This
records the whole frame instead — every region, draw, dispatch, barrier, memory
fill, decline and validation message with its full state — so a new question is
a query over an existing capture rather than a new build.

Implementation: `vulkan/vk_trace.{h,cc}` (the recorder) and `vulkan/vk_png.{h,cc}`
(a self-contained PNG writer). The query tool is `tools/gpu_capture.py`.

Everything is keyed by **guest addresses**, so a capture lines up by eye with
every other diagnostic in the module.

## Arming a capture

Set exactly one trigger. Everything else has a default.

| variable | meaning |
|---|---|
| `DELTA_GPU_CAPTURE=N` | capture guest frame N |
| `DELTA_GPU_CAPTURE_AFTER=T` | capture the first frame after T seconds of run time |
| `DELTA_GPU_CAPTURE_BUSY=N` | capture the first frame that follows a frame with ≥ N draws (for a screen whose frame number moves between runs) |

Then, optionally:

| variable | default | meaning |
|---|---|---|
| `DELTA_GPU_CAPTURE_COUNT=K` | `1` | capture K consecutive frames |
| `DELTA_GPU_CAPTURE_DIR=<dir>` | `<DELTA_GPU_DUMP_DIR or /tmp>/gpucap` | where the capture and its PNGs go |
| `DELTA_GPU_CAPTURE_DUMP=<list>` | `rt,depth` | what to read back at frame end: any of `rt`, `depth`, `tex`, `all`, `none` |
| `DELTA_GPU_CAPTURE_AT=<spec>` | off | mid-frame snapshots: a comma list of draw indices, `every:N`, or `all` |
| `DELTA_GPU_CAPTURE_EXPOSURE=<f>` | `1.0` | multiplier applied to float (HDR) targets before the PNG |
| `DELTA_GPU_CAPTURE_GAMMA=<f>` | `2.2` | gamma applied to float (HDR) targets (`1` = linear) |
| `DELTA_GPU_CAPTURE_CBUF_BYTES=<n>` | `256` | bytes of each constant buffer recorded (`0` = the whole binding) |
| `DELTA_GPU_CAPTURE_RAW=1` | off | also write the untouched readback bytes next to each PNG |
| `DELTA_GPU_CAPTURE_EXIT=1` | off | exit as soon as the capture is written |
| `DELTA_GPU_VALIDATE=1` | off | enable the Khronos validation layers and route their messages into the capture |

Disarmed, all of this costs one predictable branch on a global bool per draw,
region and barrier. Nothing is allocated and no file is opened.

Example — the frame after 40 seconds, with the depth buffer and every guest
texture, plus a snapshot of the bound targets after draws 12 and 40:

```
DELTA_GPU_CAPTURE_AFTER=40 \
DELTA_GPU_CAPTURE_DUMP=rt,depth,tex \
DELTA_GPU_CAPTURE_AT=12,40 \
DELTA_GPU_CAPTURE_DIR=/tmp/cap \
  ./run.sh -T 60
```

That writes `/tmp/cap/frame_<N>.jsonl` plus `frame_<N>_*.png`.

## The capture file

JSONL: one JSON object per line, `"t"` names the record type, `"seq"` orders
them. Records appear in the order the backend recorded them.

| `t` | contents |
|---|---|
| `capture` | file header: frame, exposure/gamma, what was dumped |
| `frame_begin` / `frame_end` | frame number, draw counts, scanout address |
| `region_begin` / `region_end` | every colour attachment (guest base, `CB_COLORn_INFO`, resolved Vulkan format, loadOp, current layout), the depth/stencil base, the render area |
| `draw` | the complete draw — see below |
| `decline` | the recompiled path refusing a draw, with the reason |
| `dispatch` | CS guest address + SPIR-V hash, group counts, every resolved resource range with its guest-memory statistics, the raw user data |
| `barrier` | every image layout transition, named after the guest resource the image holds |
| `memory_fill` | a CP DMA fill (this hardware's clear) |
| `validation` | a validation-layer message with the debug-utils label stack that names the draw |
| `dump` | a written PNG with its per-channel min/max/mean, non-zero and NaN counts |

A `draw` record carries: the render targets (guest base, format, size, per-MRT
blend word and clear word), `CB_TARGET_MASK` / `CB_SHADER_MASK` /
`CB_COLOR_CONTROL`, the depth and stencil state, the viewport (both the guest
scale/offset and the rectangle the backend sets), cull state, primitive type,
the index buffer, every vertex binding and attribute, every resolved texture
descriptor `T#` (**including the guest address the descriptor itself was
`s_load`ed from**, and **how the binding actually resolved** — to a live render
target, a feedback copy, a depth target, a storage image, a guest upload, or
the 1x1 default), every constant buffer with its bytes, every set-2 raw buffer,
and the VS/PS guest addresses with both a guest-code hash and a SPIR-V hash.

Non-finite floats are emitted as the bare `NaN` / `Infinity` literals, which
Python's `json` reads back unchanged — a NaN in a viewport or a constant is
exactly what a capture exists to show.

## Querying a capture

`tools/gpu_capture.py <command> <capture>`, where `<capture>` is a `.jsonl` file
or a directory (the newest `frame_*.jsonl` in it is used).

```
summary     frame overview: draws, regions, every RT written and sampled,
            with its end-of-frame mean/non-zero/NaN statistics
draws       one line per draw (filter with --rt ADDR, --ps ADDR, --grep TEXT)
draw N      the complete state of draw N as JSON
wrote ADDR  what wrote render target ADDR (regions, draws, dispatches, fills)
sampled ADDR   which draws sampled ADDR, through which binding, resolved how
graph       the producer/consumer graph of the frame, plus the targets nothing
            sampled and the targets sampled but never written
textures    every distinct T#, how it resolved, and its guest memory
            (--zero: only the ones reading nothing)
cbufs       constant buffers with their first floats (--draw N)
barriers    every layout transition (--resource TEXT)
dumps       every written PNG with its statistics
validation  validation-layer messages
zero        everything the frame is silently reading as zero: bindings that
            fell back to the default, guest memory that is all zero, constant
            and raw buffers that never staged, targets that ended empty
timeline    the raw event stream, one line each
diff A B    two captures compared: draw counts, per-shader-pair counts,
            per-draw state deltas, end-of-frame target means
```

`zero` is the fastest first move on "the screen is black": it names every
binding whose data never arrived, which is otherwise indistinguishable in the
output from data that is genuinely zero.

## Resource dumps

PNG, not PPM, so an agent can open them directly.

- **Colour targets** are decoded from their real Vulkan format. Float (HDR)
  targets go through `DELTA_GPU_CAPTURE_EXPOSURE` then
  `DELTA_GPU_CAPTURE_GAMMA`; a NaN texel is written as white so it cannot be
  mistaken for black.
- **Depth targets** are normalised against their own extent (a reversed-Z
  target lives in `[0.996, 1]`); the extent is recorded in the `dump` record.
- **Guest textures** are read out of guest memory, de-tiled through the same
  layout the GPU stores (`gcn/gcn_detile.h`, so every tiling mode, mip and
  `POW2_PAD` case is the one the hardware uses) and decoded. Uncompressed
  formats and BC1–BC5 are written; anything else records a `skipped` reason
  rather than guessing.
- `DELTA_GPU_CAPTURE_AT` snapshots the attachments of the region that is open
  after the named draw. A copy cannot be recorded inside dynamic rendering, so
  the region is closed first; the next draw reopens it with `loadOp = LOAD`,
  which is what it would have done anyway.

## Vulkan-level visibility

`vk_debug` already names every object it creates after the guest resource it
represents (`rt 0x8142f00000 1920x1080`, `tex 0x...`, `cs 0x...`) and brackets
the command stream in `frame N` > `region rt=...` > `recomp vs=... ps=...`
labels. With the debugger armed those names are also kept host-side, so a
`barrier` record says *which guest target* moved layout with no capture tool
attached.

`DELTA_GPU_VALIDATE=1` enables `VK_LAYER_KHRONOS_validation`, forces the
debug-utils labels on so every message carries the label stack that names the
guest draw, prints each message to stderr as `[vkval] ...` and, during an armed
frame, writes it into the capture. The loader has to be able to find the layer
(the dev shell's `vulkan-validation-layers` puts it on `XDG_DATA_DIRS`).

## Relation to RenderDoc

RenderDoc (`DELTA_RDOC_FRAME=N`, see `README.md`) is still the right tool for
stepping a draw's pixel history and reading disassembled SPIR-V interactively.
This is the complementary one: complete, uncapped, diffable text an agent can
query without a GUI, and it works on a box with no capture layer installed.
