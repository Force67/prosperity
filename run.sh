#!/usr/bin/env bash
#
# run.sh: launch a PS4 pkg under the FEX (aarch64) build of ps4delta.
#
# The graphics layer needs Vulkan + SDL3, so the binary must be built AND run
# inside the nix dev shell (`nix develop`). This wrapper does that for you.
#
# Usage:
#   ./run.sh [pkg]                run a pkg (defaults to the Isaac pkg below)
#   ./run.sh -b [pkg]            (re)build first, then run
#   ./run.sh -t [pkg]            run with FEX_SCTRACE=1 (per-syscall trace)
#   ./run.sh -T 30 [pkg]         kill after N seconds
#   ./run.sh -l out.txt [pkg]    tee all output to a log file
#
# Flags combine, e.g.  ./run.sh -b -t -T 60 -l /tmp/run.txt
# Anything after `--` is passed straight to ps4delta.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"                 # nix-toolchain build (consistent libstdc++/glibc)
BIN="$BUILD_DIR/delta/main/ps4delta"
#DEFAULT_PKG="/home/vince/Documents/dumps/The.Binding.of.Isaac.Rebirth.v1.16.PS4-CUSA00792.pkg"
#DEFAULT_PKG="/home/vince/Documents/dumps/Undertale-CUSA09415.pkg"
DEFAULT_PKG="/home/vince/Documents/dumps/gtasa.pkg"
#DEFAULT_PKG="/home/vince/Documents/dumps/uncharted2.pkg"
#DEFAULT_PKG="/home/vince/Documents/dumps/shadowofthecolossus.pkg"
#DEFAULT_PKG="/home/vince/Documents/dumps/doom64.pkg"
#DEFAULT_PKG="/home/vince/Documents/dumps/tr.pkg"
#
# /home/vince/Documents/dumps/doom64.pkg

DO_BUILD=0; TRACE=0; TIMEOUT=""; LOG=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build)   DO_BUILD=1; shift ;;
    -t|--trace)   TRACE=1; shift ;;
    -T|--timeout) TIMEOUT="$2"; shift 2 ;;
    -l|--log)     LOG="$2"; shift 2 ;;
    --)           shift; EXTRA+=("$@"); break ;;
    -h|--help)    sed -n '3,18p' "$0"; exit 0 ;;
    *)            PKG="$1"; shift ;;
  esac
done
PKG="${PKG:-$DEFAULT_PKG}"

# modules/ lives next to the binary; the nix build dir symlinks it from build-arm.
if [[ ! -e "$BUILD_DIR/delta/main/modules" && -d "$ROOT/build-arm/delta/main/modules" ]]; then
  ln -sfn "$ROOT/build-arm/delta/main/modules" "$BUILD_DIR/delta/main/modules"
fi

# Everything runs inside `nix develop` so Vulkan/SDL3 + the nix toolchain match.
INNER='set -e'
if [[ $DO_BUILD -eq 1 ]]; then
  INNER+='
    CLANGDIR=$(dirname "$(command -v clang)")
    if [[ -f "'"$BUILD_DIR"'/CMakeCache.txt" ]]; then
      cmake -B "'"$BUILD_DIR"'" -DCMAKE_BUILD_TYPE=Release -DDELTA_BUILD_TESTS=OFF -DDELTA_FEX_CLANG="$CLANGDIR" >/dev/null
    else
      cmake -GNinja -B "'"$BUILD_DIR"'" -DCMAKE_BUILD_TYPE=Release -DDELTA_BUILD_TESTS=OFF -DDELTA_FEX_CLANG="$CLANGDIR" >/dev/null
    fi
    cmake --build "'"$BUILD_DIR"'" --target ps4delta --parallel'
fi
INNER+='
  [[ -x "'"$BIN"'" ]] || { echo "missing binary: '"$BIN"' (run with -b)"; exit 1; }
  [[ -f "'"$PKG"'" ]] || { echo "missing pkg: '"$PKG"'"; exit 1; }
  cd "$(dirname "'"$BIN"'")"
  cmd=(stdbuf -o0 -e0)'
# -k: the emulator ignores SIGTERM (FEX guest threads), which leaves runaway
# processes burning every core after the timeout; escalate to SIGKILL.
[[ -n "$TIMEOUT" ]] && INNER+='
  cmd+=(timeout -k 5 --signal=KILL "'"$TIMEOUT"'")'
INNER+='
  cmd+=("'"$BIN"'" "'"$PKG"'" '"${EXTRA[*]:-}"')'
[[ $TRACE -eq 1 ]] && INNER+='
  export FEX_SCTRACE=1'
INNER+='
  # Show a real window when a display is available; fall back to SDL'"'"'s offscreen
  # driver only when headless (no X / Wayland), where the GPU still renders and
  # dumps frames (DELTA_GPU_DUMP=1) for verification.
  if [[ -z "${SDL_VIDEODRIVER:-}" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER=offscreen
  fi
  echo ">> ${cmd[*]}"'
if [[ -n "$LOG" ]]; then
  INNER+='
  "${cmd[@]}" 2>&1 | tee "'"$LOG"'"'
else
  INNER+='
  "${cmd[@]}"'
fi

cd "$ROOT"
exec nix develop --command bash -c "$INNER"
