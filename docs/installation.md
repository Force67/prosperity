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
