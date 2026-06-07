# Building

Prosperity targets Linux (x86-64 and aarch64) and Android. The CPU backend is
chosen automatically from the host architecture:

* **x86-64 host** -> `NATIVE`: guest x86-64 runs directly on the CPU.
* **aarch64 host** -> `FEX`: guest x86-64 runs through an embedded FEX JIT.

The graphics layer needs Vulkan, SDL3 and shaderc. The supported way to get a
matching toolchain is the Nix dev shell defined in `flake.nix`.

## Get the source

All third-party code (capstone, fmtlib, zlib, xbyak, FEX, mbedtls, equilibrium,
googletest) is vendored as git submodules, so clone recursively:

```bash
git clone --recursive https://github.com/Force67/prosperity.git
cd prosperity
# already cloned without --recursive?
git submodule update --init --recursive
```

## Linux (Nix, recommended)

Install Nix (with flakes enabled) from <https://nixos.org/download>, then:

```bash
nix develop                                 # enter the dev shell
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build                         # or: ninja -C build
```

The dev shell provides cmake, ninja, the compiler, Vulkan, SDL3, shaderc, and
mesa's lavapipe (software Vulkan) for headless/GPU-less machines.

To run everything in one shot without entering the shell interactively:

```bash
nix develop --command bash -c 'cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build'
```

The resulting binary is `build/delta/main/ps4delta`. See
[installation.md](installation.md) for how to run it.

## Linux (without Nix)

You need a C++20 toolchain (GCC 13+ or Clang 18+) plus:

* `cmake` >= 3.20 and `ninja`
* Vulkan headers + loader (`libvulkan`)
* SDL3 (note: not packaged on Ubuntu 24.04 yet; build it from source)
* shaderc (`glslc` / `libshaderc`)

Then configure and build the same way:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If `find_package(Vulkan)` or `find_package(SDL3)` fails, the corresponding
dependency is missing. Nix is recommended precisely because SDL3 and shaderc are
awkward to obtain on stable distros.

## Android

Android builds use the NDK toolchain (clang) and run on aarch64 via the FEX
backend. There are two flavours:

* **Headless adb-shell binary** (default): dumps frames, driven over `adb`.
* **On-screen NativeActivity APK**: configure with `-DDELTA_ANDROID_APP=ON`
  (requires the NDK; builds `libps4delta_app.so` and presents to the device).

The exact NDK/SDK wiring lives in the (gitignored) `build-apk.sh` /
`run-android.sh` helper scripts.

## CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | standard CMake build type |
| `DELTA_BUILD_TESTS` | `ON` | build the unit tests (`ctest`) |
| `DELTA_BACKEND` | auto | `NATIVE` or `FEX`; auto-selected from host arch |
| `DELTA_ANDROID_APP` | `OFF` | build the on-screen Android app (needs the NDK) |

Host-only dev tools (`tools/modload`, `modexec`, `pkg_check`, `gfx_test`) are
built automatically on non-Android targets.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

CI (`.github/workflows/cibuild.yml`) runs exactly this configure/build/test flow
inside the same Nix dev shell.
