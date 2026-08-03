{
  description = "PS4Delta - PS4 emulation and research project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        renderdocPython = pkgs.renderdoc.overrideAttrs (old: {
          cmakeFlags = (old.cmakeFlags or [ ]) ++ [
            "-DENABLE_PYRENDERDOC=TRUE"
          ];
          postInstall = (old.postInstall or "") + ''
            install -Dm755 lib/renderdoc.so $out/lib/renderdoc.so
          '';
        });

        ps4delta = pkgs.stdenv.mkDerivation {
          pname = "ps4delta";
          version = "0.0.0";

          src = self;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
          ];

          # capstone, fmtlib, zlib, xbyak, equilibrium are vendored as
          # submodules, so we do not bring system copies into the closure.
          # The graphics layer needs Vulkan + SDL3 (window/present).
          buildInputs = with pkgs; [
            vulkan-headers
            vulkan-loader
            sdl3
            shaderc          # runtime GLSL -> SPIR-V for the shader recompiler
          ];

          cmakeFlags = [
            "-GNinja"
            "-DCMAKE_BUILD_TYPE=Release"
          ];

          installPhase = ''
            mkdir -p $out/bin
            cp ps4delta $out/bin/
          '';

          meta = with pkgs.lib; {
            description = "Early high-level PlayStation 4 emulator";
            license = licenses.gpl2Plus;
            platforms = platforms.linux;
            mainProgram = "ps4delta";
          };
        };
      in {
        packages.default = ps4delta;
        packages.ps4delta = ps4delta;

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            pkg-config
            ccache          # compiler cache (CI warms it across runs)
            clang_18
            clang-tools
            gdb
            git
            vulkan-headers
            vulkan-loader
            vulkan-validation-layers
            mesa            # lavapipe: software Vulkan for headless testing
            sdl3
            shaderc         # runtime GLSL -> SPIR-V for the shader recompiler
            renderdocPython # frame capture: UI, CLI, and Python replay API
          ];

          shellHook = ''
            echo "ps4delta dev shell, run:  cmake -G Ninja -B build && ninja -C build"
            # RenderDoc's capture layer is implicit: the loader picks it up from
            # this shell's XDG_DATA_DIRS. A stale renderdoc_capture.json in
            # ~/.local/share (pointing at a collected store path) shadows it and
            # makes capture silently fail, so pin the good one for capture runs.
            export DELTA_RDOC_LAYER_PATH="${renderdocPython}/share/vulkan/implicit_layer.d"
            export RENDERDOC_PYTHON_PATH="${renderdocPython}/lib"
            export PYTHONPATH="$RENDERDOC_PYTHON_PATH''${PYTHONPATH:+:$PYTHONPATH}"
            # Vulkan ICD: prefer the system GPU driver. Hardware ICDs live under
            # /run/opengl-driver on NixOS and /usr/share/vulkan/icd.d (or
            # /etc/vulkan/icd.d) on FHS distros. When one exists, point
            # VK_ICD_FILENAMES at it explicitly (plus nix lavapipe as fallback):
            # the nix vulkan-loader does not scan the system data dirs, so
            # leaving it unset used to silently select lavapipe and run the
            # whole renderer on a ~30ms/frame software rasteriser even on a box
            # with a real GPU. Only pin lavapipe alone when there is no system
            # driver at all (headless / GPU-less CI).
            if [ -z "$VK_ICD_FILENAMES" ]; then
              icds=""
              for d in /run/opengl-driver/share/vulkan/icd.d \
                       /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
                for f in "$d"/nvidia_icd*.json "$d"/radeon_icd*.json \
                         "$d"/intel_icd*.json; do
                  [ -e "$f" ] && icds="$icds''${icds:+:}$f"
                done
              done
              for f in ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json; do
                [ -e "$f" ] && icds="$icds''${icds:+:}$f" && break
              done
              [ -n "$icds" ] && export VK_ICD_FILENAMES="$icds"
              # nix glibc ignores /etc/ld.so.cache, so dlopen of a system ICD
              # (libGLX_nvidia.so.0 etc.) cannot find the driver library. The
              # FHS lib dirs must NOT go on LD_LIBRARY_PATH (it outranks nix
              # RUNPATHs, so the system glibc would shadow the nix one and break
              # every binary in the shell). Instead expose ONLY the driver libs
              # through a symlink dir: nix binaries never request these names,
              # and the driver's own libc/libX11 deps still resolve to nix libs.
              # The farm is shared by every shell, so it must never be empty at
              # any instant: entering a second `nix develop` while the emulator
              # runs used to `rm -f` the farm out from under it, the ICD then
              # failed to dlopen, and the renderer silently fell back to
              # llvmpipe mid-session. Link in place (ln -sf overwrites one link
              # at a time, so a concurrent reader always sees a usable farm) and
              # prune only links whose target is gone -- never a resolving one.
              nvdir="/tmp/ps4delta-nvlibs-$(id -u)"
              mkdir -p "$nvdir"
              for l in /usr/lib/aarch64-linux-gnu/libGLX_nvidia.so* \
                       /usr/lib/aarch64-linux-gnu/libnvidia*.so* \
                       /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
                       /usr/lib/x86_64-linux-gnu/libnvidia*.so*; do
                [ -e "$l" ] && ln -sf "$l" "$nvdir/"
              done
              find "$nvdir" -maxdepth 1 -xtype l -delete 2>/dev/null
              if ls "$nvdir"/*.so* >/dev/null 2>&1; then
                # The driver's DT_NEEDED includes X11 libs that nothing else has
                # loaded yet when the renderer (windowless) dlopens the ICD, so
                # give it the nix X11 libs too (X-only dirs: no libc shadowing).
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH''${LD_LIBRARY_PATH:+:}$nvdir:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.xorg.libxcb}/lib:${pkgs.libglvnd}/lib"
              fi
            fi
          '';
        };
      });
}
