# Prosperity <img src="https://i.imgur.com/zOaZAH2.png" width="40" height="40" />

What comes after Delta? Prosperity. 

Prosperity, originally launched as PS4Delta in 2019, is a PlayStation 4 and PlayStation 5 emulator for Linux and Android. Windows support is planned for a later release.

On AMD64 hosts, the CPU runs without emulation. On ARM hosts, FEX is used to execute the CPU code.

The GPU is fully emulated, with graphics code recompiled to SPIR-V.

Everything else, including the loader, kernel HLE, and devices, runs as natively on the host as possible. Graphics output is handled through Vulkan and SDL3.


|  |  |
|:------------:|:------------:|
| <img width="3826" height="2085" alt="deadcellsps5" src="https://github.com/user-attachments/assets/3144bc0a-32d8-4a2b-8f20-b46e54749b87" /> | <img width="2048" height="1118" alt="signal-2026-07-28-09-50-25-554_002" src="https://github.com/user-attachments/assets/a48e9f5c-ba61-4735-932e-0ed14ad7c88c" /> |
| **Dead Cells**<br>Goes ingame | **Skyrim**<br>Some games don't render nicely yet |
| <img width="2048" height="940" alt="signal-2026-07-28-09-50-07-032_002" src="https://github.com/user-attachments/assets/3516e741-8df2-4db4-a78d-4a1ac58926ce" /> | <img width="2048" height="940" alt="signal-2026-07-28-09-49-56-587_002" src="https://github.com/user-attachments/assets/ac02716f-4d7d-4eb3-8df3-3468fe929d35" /> |
| **Undertale**<br>Undertale runs nicely on android | **The binding of isaac (PS5)**<br>Runs nicely on android too |




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

## Thanks and credits
- zecoaxco
- anon (You know who you are)
- GPCS4
- idc/uplift (original inspiration for PS4Delta)

## Legal

Prosperity ships no Sony code. You must supply your own decrypted system modules
and games, dumped from hardware you own. See
[docs/installation.md](docs/installation.md).
