# Prosperity <img src="https://i.imgur.com/zOaZAH2.png" width="40" height="40" />

What comes after Delta? Prosperity. 

Propserity (formerly started as PS4Delta in 2019) is a dual PlayStation 4 and 5 emulator for Linux and Android (Windows support coming later).

The CPU is not emulated for amd64 hosts, on ARM FEX is used to run.
The GPU is entirely emulated, and recompiled to SPIRV.

Everything else (loader, kernel HLE, devices) is as host-native as possible.
Graphics are presented with Vulkan + SDL3.

## Documentation
* [Building](docs/building.md)
* [Installation & running](docs/installation.md)

## Quick start (Linux)

Prosperity builds inside a [Nix](https://nixos.org/download) dev shell that pins
the exact Vulkan / SDL3 / shaderc toolchain it needs:

```bash
git clone --recursive https://github.com/Force67/prosperity.git
cd prosperity
nix develop --command bash -c 'cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build'
./build/delta/main/ps4delta /path/to/game.pkg
```

See [docs/building.md](docs/building.md) for the non-Nix build, Android, and the
available CMake options.

## Requirements

### On linux
* __Processor__: x86-64 (made in the last 10 years) with AVX, SSE4.2 and BMI1, or an aarch64 host.
* __RAM__: 16 GB of RAM (8 may work, depending on the type of game you want to run).
* __Graphics__: a GPU with support for Vulkan 1.4+, and a minimum of 6-8 GB vram (the more, the better).

### On android
* __Processor__: Preferably something new, like one of those fancy new Snapdragons.
* __RAM__: 12 GB of RAM (8 may work, depending on the type of game you want to run).
* __Graphics__: a GPU with support for Vulkan 1.4+.

## Legal

Prosperity ships no Sony code. You must supply your own decrypted system modules
and games, dumped from hardware you own. See
[docs/installation.md](docs/installation.md).
