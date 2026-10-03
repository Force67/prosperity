# Build

Linux: x86-64 uses the native CPU backend. ARM64 uses FEX.

## Linux with Nix

Install [Nix](https://nixos.org/download) with flakes enabled, then:

```bash
git clone --recursive https://github.com/Force67/prosperity.git
cd prosperity
nix develop
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Already cloned? Run `git submodule update --init --recursive` first.
Already configured `build/` with another generator? Leave out `-G Ninja`.

Binary: `build/delta/main/ps4delta`.
Next: [install system modules and run a game](installation.md).
Keep the Nix shell open when running the binary.

## Ubuntu CI and release packages

CI builds natively on Ubuntu 24.04 with GCC 14 and Ubuntu's development packages.
GCC 13 miscompiles the packed buffer format conversion at release optimization.
SDL3 3.4.2 is built from a pinned commit and linked statically. Vulkan and OpenGL
are enabled; D3D12 is disabled. The shader recompiler and FFmpeg video decoder
use Ubuntu's SPIRV-Tools and FFmpeg packages.

Every successful build produces a `prosperity-ubuntu-24.04-amd64` artifact with a
`.deb` package. CI installs it in a clean Ubuntu 24.04 container and checks the
CLI and installed files. Pushing a `v*` tag publishes the tested package as a
GitHub release. Tag versions must be valid Debian versions, for example `v0.1.0`.

Nix remains available for local development.

## Linux without Nix

Install a C++20 compiler, CMake 3.20+, Ninja, pkg-config, and development
packages for Vulkan, SDL3, shaderc, SPIRV-Tools, SPIRV-Headers, and FFmpeg.
OpenGL/EGL + libepoxy and vkd3d + DXC enable optional graphics backends.

Use the same CMake commands above. If a dependency is hard to obtain on your
distro, use the Nix shell.

## Android

Use the NDK toolchain and an ARM64 target:

```bash
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DDELTA_BUILD_TESTS=OFF
cmake --build build-android --target ps4delta
```

Set `ANDROID_NDK` to your NDK directory. This builds the headless executable
for `adb shell`. Add `-DDELTA_ANDROID_APP=ON` and build `ps4delta_app` for the
NativeActivity library. APK packaging uses local, gitignored helper scripts.

## Build options

| CMake option | Default | Purpose |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` for debugging |
| `DELTA_BUILD_TESTS` | `ON` | Unit tests |
| `DELTA_BACKEND` | Host architecture | `NATIVE` (x86-64), `FEX` (ARM64) |
| `DELTA_ANDROID_APP` | `OFF` | Android app library |
| `DELTA_TRACY` | `ON`, Android `OFF` | Tracy profiling |

## Tests and profiling

```bash
ctest --test-dir build --output-on-failure
tracy                                      # connect to a running ps4delta
tools/drun.py uc2 -t 130 --tracy 100:20     # capture 20 seconds at t=100
tools/drun.py uc2 -t 130 --perf 100:10:NdJob # sample the named guest thread
```
