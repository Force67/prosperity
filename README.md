# Prosperity <img src="https://i.imgur.com/zOaZAH2.png" width="40" height="40" />

What comes after Delta? Prosperity.

An experimental PlayStation 4 emulator for Linux and Android. Guest x86-64 code
runs directly on x86-64 hosts and through an embedded FEX JIT on aarch64 (arm64)
hosts; everything else (loader, kernel HLE, devices) is host-native. Graphics are
presented with Vulkan + SDL3.

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
* __Graphics__: a GPU with support for Vulkan 1.4+.

### On android
* __Processor__: Preferably something new, like one of those fancy new Snapdragons.
* __RAM__: 12 GB of RAM (8 may work, depending on the type of game you want to run).
* __Graphics__: a GPU with support for Vulkan 1.4+.

## Legal

Prosperity ships no Sony code. You must supply your own decrypted system modules
and games, dumped from hardware you own. See
[docs/installation.md](docs/installation.md).
