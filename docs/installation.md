# Installation & running

After [building](building.md), the emulator binary is at
`build/delta/main/ps4delta`. It takes a `.pkg` as its first argument; any further
arguments are passed through to the guest:

```bash
./build/delta/main/ps4delta /path/to/game.pkg [guest args...]
```

The window stays up until you close it. Prosperity ships no Sony code, so you
have to supply the game and (optionally) the system modules yourself.

## Games

Point the emulator at a PS4 `.pkg`. The title's own modules are loaded straight
out of the package (SDK PRX from `/app0/sce_module`, the game's own PRX at the
app root), so a single `.pkg` is enough to attempt a boot.

## System modules (firmware)

For better compatibility you can provide decrypted PS4 system modules. The loader
looks for them as `.sprx` files in a `modules/` directory next to the binary and
prefers those over the copies inside the pkg:

```
build/delta/main/
├── ps4delta
└── modules/
    ├── libkernel.sprx
    ├── libSceLibcInternal.sprx
    └── ...
```

These must be decrypted modules dumped from hardware you own; Prosperity cannot
decrypt retail firmware for you. Without them, the emulator falls back to the
modules bundled in the pkg.

Set `DELTA_DATA_DIR=/some/dir` to look for `modules/` (and other host assets)
under that directory instead of next to the binary.

PS4 hardware mode defaults to Base. Set `DELTA_PS4_NEO=1` to emulate Neo-capable
hardware. A title enters enhanced Neo mode only when its `param.sfo` also marks
Neo support; that mode selects `modules/libSceGnmDriverForNeoMode.sprx` and the
Neo shader ISA. Other titles continue to use the Base driver and ISA.

## Options

Every `DELTA_*` knob is a named option (`base::Option`), settable three ways.
Later sources win:

1. the environment: `DELTA_GPU_SNAP=120 ./ps4delta game.pkg`
2. an options file: `./ps4delta game.pkg --options=minecraft.txt` (or
   `DELTA_OPTIONS=minecraft.txt`, several files separated by commas)
3. the command line: `./ps4delta game.pkg +DELTA_GPU_SNAP=120`

An options file is one entry per line, which keeps a long debugging set in the
repo instead of in shell history:

```
// Minecraft: what the renderer did on the way to the world screen.
+DELTA_GPU_SNAP=600          // set a value
+DELTA_GPU_SNAP_EXIT         // no value means 1 / on
-DELTA_GPU_VSYNC             // back to the compiled-in default
+DELTA_GPU_DUMP_DIR="/tmp/mc shots"
```

Lines starting with `//` or `#` are comments, the leading `+` is optional on an
assignment, and a name nothing recognises is reported at startup rather than
ignored silently.

`--dump-options=all.txt` writes every registered option with its current value
and description in exactly that format, so a run can be captured as a file and
edited from there. `--dump-options` alone logs the same to stdout.

## Headless / no display

When no display is available (no `DISPLAY` / `WAYLAND_DISPLAY`), run with SDL's
offscreen driver; the GPU still renders through Vulkan (lavapipe in the Nix
shell on machines without a GPU):

```bash
SDL_VIDEODRIVER=offscreen ./build/delta/main/ps4delta /path/to/game.pkg
```

## run.sh helper

`run.sh` wraps the aarch64 / FEX build: it launches a pkg inside `nix develop`
(so Vulkan/SDL3 match) with optional rebuild, syscall tracing, a timeout, and
log teeing. See `./run.sh -h`.
