#!/usr/bin/env bash
# Build Prosperity for the Steam Deck (Zen 2, SteamOS) and push it to one.
#
#   tools/deploy_deck.sh [deck@host]     build + sync to ~/prosperity
#   tools/deploy_deck.sh --shortcut ...  also add the Steam shortcut
#
# The nix toolchain's glibc is older than SteamOS's, so the binary runs on the
# Deck's own loader and libraries once the nix interpreter and RUNPATH are
# stripped. SteamOS ships SDL3, FFmpeg, libepoxy and the Vulkan loader, so the
# Deck's RADV and gamescope's WSI layer apply as they would to a Steam game.
# The Deck has no vkd3d, so the D3D12 backend is off.
#
# DECK_SSH_KEY picks the identity (default ~/.ssh/steamdeck_devkit).
set -euo pipefail
cd "$(dirname "$0")/.."

SHORTCUT=0
if [ "${1:-}" = --shortcut ]; then
  SHORTCUT=1
  shift
fi
TARGET="${1:-deck@steamdeck}"
KEY="${DECK_SSH_KEY:-$HOME/.ssh/steamdeck_devkit}"
BUILD=build-deck
STAGE="$BUILD/stage"

nix develop -c bash -c "
  set -euo pipefail
  if [ ! -f $BUILD/build.ninja ]; then
    cmake -S . -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DDELTA_BUILD_TESTS=OFF -DDELTA_GPU_D3D12=OFF -DDELTA_TRACY=OFF \
      -DCMAKE_C_FLAGS='-march=znver2 -mtune=znver2' \
      -DCMAKE_CXX_FLAGS='-march=znver2 -mtune=znver2'
  fi
  ninja -C $BUILD -j6 ps4delta
"

rm -rf "$STAGE"
mkdir -p "$STAGE"
nix shell nixpkgs#patchelf -c patchelf \
  --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath \
  --output "$STAGE/ps4delta" "$BUILD/delta/main/ps4delta"
cp -rL "$BUILD/delta/main/game_profiles" "$STAGE/"
cp delta/main/_res/logo.png tools/deck/prosperity.sh tools/deck/add_shortcut.py \
  "$STAGE/"

rsync -az --delete --exclude last.log -e "ssh -i $KEY" "$STAGE/" \
  "$TARGET:prosperity/"
echo "deployed to $TARGET:~/prosperity (run ~/prosperity/prosperity.sh)"

if [ "$SHORTCUT" = 1 ]; then
  # Steam rewrites shortcuts.vdf on exit; the session restarts it afterwards.
  ssh -i "$KEY" "$TARGET" '
    export XDG_RUNTIME_DIR=/run/user/1000
    steam -shutdown >/dev/null 2>&1 || true
    while pgrep -x steam >/dev/null; do sleep 1; done
    python3 ~/prosperity/add_shortcut.py'
fi
