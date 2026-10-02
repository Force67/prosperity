# Install modules and run

Build first: [building.md](building.md). Commands below run from the repository
root, inside `nix develop` if you built with Nix.

## Home screen (Linux)

Start without a game path:

```bash
./build/delta/main/ps4delta
```

Choose a recent game and press **Play**, or use **Open game** / **Open folder**.
Each card shows a PS4 or PS5 badge from the game's metadata.
Games without background art use an animated Delta backdrop chosen at random.
A banner lists missing firmware modules and the setup commands to import them.
Arrow keys browse the cards, Enter plays, O opens a file, and Esc exits.
A controller's D-pad browses; Cross plays, Triangle opens, and Circle exits.
Game paths on the command line still boot directly.

The last 12 launches, titles, and artwork are saved in
`~/.local/share/prosperity/recent-games` and `covers/` (under
`$XDG_DATA_HOME/prosperity/` when set). Games at missing paths stay visible but
cannot be played. Native file pickers use the desktop's XDG portal.

## Import firmware modules once

Use a separate setup switch for each console:

```bash
./build/delta/main/ps4delta --configure-ps4-fw /dumps/ps4/modules
./build/delta/main/ps4delta --configure-ps5-fw /dumps/ps5/common/lib:/dumps/ps5/priv/lib
```

These copy `.sprx` files into:

```text
~/.local/share/prosperity/modules/
  ps4/
    libkernel.sprx
    libSceLibcInternal.sprx
    ...
  ps5/
    libkernel.native.sprx
    libSceLibcInternal.sprx
    ...
```

If `XDG_DATA_HOME` is set, storage uses `$XDG_DATA_HOME/prosperity/modules/`
instead. Subsequent launches use these folders automatically:

```bash
./build/delta/main/ps4delta /games/game.pkg
./build/delta/main/ps4delta /games/game.ffpkg
```

Run either setup switch again to replace only that console's stored modules.
The other console's folder and the original dump remain untouched. Import is
flat, without recursion; for PS5, pass each directory containing `.sprx` files.
The first source directory wins duplicate filenames. `.native.sprx` names are
preserved.

Setup checks the directories and the decrypted x86-64 ELF headers for
`libkernel` and `libSceLibcInternal`. It does not verify the complete firmware's
compatibility. Invalid input or a failed copy keeps the previous module folder.
Without a game, setup imports and exits. Include a game path to launch immediately.

For a temporary override, without replacing stored modules:

```bash
./build/delta/main/ps4delta --ps4-modules /other/ps4/modules /games/game.pkg
./build/delta/main/ps4delta --ps5-modules /other/ps5/modules /games/game.ffpkg
```

`DELTA_PS4_MODULES` and `DELTA_PS5_MODULES` environment variables also override
the stored folders. To forget a platform's setup, remove its `ps4/` or `ps5/`
folder.

## PS4 system modules

Supply decrypted `.sprx` files from your own console dump. Copy them directly
into `modules/` beside the executable if you have no imported PS4 modules,
without the dump's parent directories:

```bash
mkdir -p build/delta/main/modules
cp /path/to/ps4-decrypted-modules/*.sprx build/delta/main/modules/
```

The layout should look like this:

```text
build/delta/main/
  ps4delta
  game_profiles/
  modules/
    libkernel.sprx
    libkernel_sys.sprx
    libSceLibcInternal.sprx
    libSceGnmDriver.sprx
    ...
```

Copy the full decrypted module set. Those filenames are examples, not a complete
list. Renaming encrypted files does not decrypt them.

To keep modules outside the build directory:

```bash
mkdir -p ~/.local/share/prosperity/modules
cp /path/to/ps4-decrypted-modules/*.sprx ~/.local/share/prosperity/modules/
cp -r game_profiles ~/.local/share/prosperity/
./build/delta/main/ps4delta --data-dir ~/.local/share/prosperity /games/game.pkg
```

`--data-dir` changes both the module and game profile location. The equivalent
environment variable is `DELTA_DATA_DIR`.

## PS5 system modules

Keep PS5 modules separate from PS4 modules. Point at directories containing the
decrypted PS5 `.sprx` files:

```bash
./build/delta/main/ps4delta \
  --ps5-modules /dumps/ps5/system/common/lib:/dumps/ps5/system/priv/lib \
  /games/game.ffpkg
```

Use your dump's actual directories. Keep `.native.sprx` filenames intact, the
loader prefers those over compatibility `.sprx` files. The equivalent environment
variable is `DELTA_PS5_MODULES`. PS5 titles never use the PS4 `modules/` directory.

## How module discovery works

Lookup uses exact, case-sensitive filenames on Linux. It does not scan
subdirectories or flatten a firmware dump for you.

| Platform | Lookup order |
| --- | --- |
| PS4 | `<selected folder>/<name>.sprx` (default: `modules/` beside the executable), then the game's `sce_module/<name>.prx`, then the game's root `<name>.prx` |
| PS5 | Each `--ps5-modules` directory in order, trying `<name>.native.sprx` before `<name>.sprx`; then game `.prx` files in `decrypted/sce_module/`, `decrypted/`, `sce_module/`, and the app root |

PS4 exceptions: `libc` and `libSceFios2` prefer the game's `.prx` files.
Neo mode selects `libSceGnmDriverForNeoMode.sprx` for supported PS4 titles.
Both platforms need `libkernel` and `libSceLibcInternal` to start.

A full PS5 dump can keep its directory tree:

```text
ps5/
  system/common/lib/
    libkernel.sprx
    libSceLibcInternal.sprx
    libSceExample.native.sprx
    ...
  system/priv/lib/
    ...
```

Pass both `lib/` directories, not `ps5/`. The example filenames illustrate the
layout; your dump determines which directory contains each module.

Host `.sprx` files must be decrypted x86-64 ELF files. Check a few before copying:

```bash
file /path/to/modules/libkernel.sprx
od -An -tx1 -N4 /path/to/modules/libkernel.sprx # ELF begins 7f 45 4c 46
```

Discovery is predictable for a clean dump, but it does not verify that firmware
versions or module ABIs match. Keep one coherent module set per console.
On PS4, an existing but invalid host `.sprx` blocks the game's `.prx` fallback.
On PS5, an invalid candidate falls through to the next candidate.

## Start from a terminal

```bash
./build/delta/main/ps4delta /games/game.pkg      # PS4 package
./build/delta/main/ps4delta /games/game.ffpkg    # PS5 backup
./build/delta/main/ps4delta /games/app           # extracted app with sce_sys/
./build/delta/main/ps4delta /games/app/eboot.bin  # executable
./build/delta/main/ps4delta '/games/My Game.zip' # ZIP or RAR app dump
```

Games must contain executable code the loader can read. PS5 `.ffpkg` backups can
include a `decrypted/` tree, which the loader uses when available.

In a desktop terminal, the game opens an SDL window. A text-only TTY or SSH
session has no desktop display; for a headless run with frame captures:

```bash
mkdir -p /tmp/prosperity-frames
./build/delta/main/ps4delta --headless --dump-frames /tmp/prosperity-frames \
  /games/game.pkg
```

Headless runs still need Vulkan. Stop with `Ctrl+C`.

## Useful flags

```bash
./build/delta/main/ps4delta --help
./build/delta/main/ps4delta --version
./build/delta/main/ps4delta --dump-options=delta.txt
./build/delta/main/ps4delta --options delta.txt /games/game.pkg
./build/delta/main/ps4delta --profile off /games/game.pkg
./build/delta/main/ps4delta /games/game.pkg -- -debug # guest arguments
```

`--data-dir`, `--ps4-modules`, `--ps5-modules`, `--configure-ps4-fw`,
`--configure-ps5-fw`, `--profile`, `--options`, and `--dump-frames` accept either `--flag value` or `--flag=value`. Runtime settings also accept
`+DELTA_NAME=VALUE` or environment variables.

Saved firmware settings apply first, then the environment, then `DELTA_OPTIONS`
files, then command-line
settings from left to right. Game profiles fill unset settings only. Put `--`
before guest arguments that resemble launcher options.

## If it fails

| Symptom | Check |
| --- | --- |
| `unable to load module` | Decryption, filename, and module directory |
| Missing PS5 exports | PS5 native modules, separate from PS4 modules |
| Missing profiles after `--data-dir` | Copy `game_profiles/` into that directory |
| SDL window cannot open | Run in a desktop session, or use headless mode above |
| Vulkan device unavailable | GPU driver and Vulkan loader; try the Nix shell |

Saves live in `~/.prosperity/savedata/`; downloaded game data lives in
`~/.prosperity/download/`.
