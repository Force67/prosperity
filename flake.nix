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
            clang_18
            clang-tools
            gdb
            git
            vulkan-headers
            vulkan-loader
            vulkan-validation-layers
            mesa            # lavapipe: software Vulkan for headless testing
            sdl3
          ];

          shellHook = ''
            echo "ps4delta dev shell, run:  cmake -G Ninja -B build && ninja -C build"
            # Default to nix lavapipe (software Vulkan) so the GPU backend works
            # headless / without a GPU driver. Override VK_ICD_FILENAMES to use a
            # real GPU + display.
            if [ -z "$VK_ICD_FILENAMES" ]; then
              for f in ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.*.json; do
                [ -e "$f" ] && export VK_ICD_FILENAMES="$f" && break
              done
            fi
          '';
        };
      });
}
