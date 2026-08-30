#!/usr/bin/env bash
#
# verify.sh: the regression net for structural refactors.
#
# Test coverage is ~3%, so the proof that a move changed nothing is a
# fingerprint of observable behaviour taken before and after it:
#
#   tools/verify.sh baseline      # writes .verify/baseline.txt
#   ... refactor ...
#   tools/verify.sh               # writes .verify/current.txt and diffs
#
# What it fingerprints, and why each line is here:
#
#   layering / ctest    the only machine-checked invariants that exist:
#                       gpu-internal directory rules, module-level edges,
#                       and the unit tests.
#   isaac-boot          a digest of Isaac's boot log with timestamps, addresses
#                       and line numbers normalised away, then sorted. Measured
#                       stable across repeated runs of the same binary, and it
#                       covers module loading, syscall dispatch and HLE
#                       bring-up -- the parts a kern refactor can break.
#                       NOTE: the presented FRAME is not usable here. Three runs
#                       of one binary produced three different frames (the
#                       capture lands on whichever frame the wall clock reaches
#                       first), so hashing it reports noise as regression.
#   isaac-armed         the same boot with debug probes ARMED. Every probe knob
#                       defaults off, so a default run executes none of the
#                       instrumentation in kern/probe and would pass unchanged
#                       if it were deleted outright. This is the only check
#                       that sees that code at all. The specs are
#                       title-independent (a dmem range, an eboot offset), and
#                       DELTA_GUEST_WPROT deliberately routes faults through
#                       the SIGSEGV path the fatal crash handler shares.
#                       Recorded as the SET of probes that reported: the
#                       payloads and the resulting frame are timing-perturbed
#                       and are not stable enough to hash.
#   crash-smoke         the crash handler still resolves a guest address to
#                       <module>+offset.
#   undertale           a second title, to a boot milestone rather than a hash.
#
# Set VERIFY_CLEAN=1 to rebuild from scratch. Off by default because the tree
# builds incrementally in seconds; on because `make` has been observed here
# linking stale objects, and an A/B against a stale binary invents regressions.
# The verify tree uses Ninja for that reason.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$ROOT/.verify
BUILD=$OUT/build
mkdir -p "$OUT"

case ${1:-check} in
  baseline) FP=$OUT/baseline.txt ;;
  check)    FP=$OUT/current.txt ;;
  *) echo "usage: $0 [baseline|check]" >&2; exit 2 ;;
esac
: > "$FP"

DUMPS=${DUMPS:-/home/vince/Documents/dumps}
ISAAC=${ISAAC:-$DUMPS/The.Binding.of.Isaac.Rebirth.v1.16.PS4-CUSA00792.pkg}
UNDERTALE=${UNDERTALE:-$DUMPS/Undertale-CUSA09415.pkg}

say() { printf '%-20s %s\n' "$1" "$2" >> "$FP"; printf '%-20s %s\n' "$1" "$2"; }

# --- layering ---------------------------------------------------------------
{ python3 delta/gpu/tests/check_layering.py . &&
  python3 delta/tests/check_module_layering.py .; } >"$OUT/layering.log" 2>&1 &&
  say layering OK || say layering "VIOLATIONS ($(wc -l <"$OUT/layering.log"))"

# --- build + tests ----------------------------------------------------------
[ "${VERIFY_CLEAN:-0}" = 1 ] && rm -rf "$BUILD"
if nix develop --command cmake -GNinja -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
     >"$OUT/cmake.log" 2>&1 &&
   nix develop --command cmake --build "$BUILD" --parallel 8 \
     >"$OUT/build.log" 2>&1; then
  say build OK
else
  say build FAILED; tail -40 "$OUT/build.log"; exit 1
fi
say ctest "$(nix develop --command ctest --test-dir "$BUILD" 2>&1 |
             grep -oE '[0-9]+ tests failed out of [0-9]+' | head -1)"

BIN=$BUILD/delta/main/ps4delta
[ -e "$BUILD/delta/main/modules" ] ||
  ln -sfn "$ROOT/build/delta/main/modules" "$BUILD/delta/main/modules" 2>/dev/null

# run <label> <seconds> <pkg> <env...>
#
# Returns 0 whenever the run happened at all. The emulator ignores SIGTERM, so
# a run that does not exit itself is SIGKILLed by the timeout and reports a
# non-zero status; treating that as failure would silently drop the check from
# the fingerprint, which is worse than a wrong value.
run() {
  local label=$1 secs=$2 pkg=$3; shift 3
  local dir=$OUT/$label
  [ -f "$pkg" ] || { say "$label" "SKIP (missing pkg)"; return 1; }
  rm -rf "$dir"; mkdir -p "$dir"
  ( cd "$BUILD/delta/main" &&
    env SDL_VIDEODRIVER=offscreen DELTA_GPU_DUMP_DIR="$dir" "$@" \
      timeout -k 5 --signal=KILL "$secs" "$BIN" "$pkg" ) >"$OUT/$label.log" 2>&1
  return 0
}

# Normalise away everything that is allowed to move, then sort (the log is
# written by several threads, so line order is not stable but content is):
#
#   ANSI colouring, the timestamp, and build/dump paths.
#   Hex values and decimal runs -- the latter covers the func:line prefix, so
#     moving code between files is not reported as a behaviour change.
#   "sp+" lines. guestStackTrace SCANS the stack window and prints every slot
#     that still looks like a code pointer, so it reports stale residue left by
#     earlier calls. Relocating code shifts that residue without changing what
#     the guest did; measured on the kern/probe extraction, these were the only
#     lines that moved while the rest of the log stayed byte-identical.
boot_digest() {
  sed -E 's/\x1b\[[0-9;]*m//g; s/^\[[^]]*\] //' "$OUT/$1.log" |
    grep -v 'sp+' |
    sed -E 's#/[^ ]*/(build|\.verify)[^ ]*#PATH#g; s#/tmp/[^ ]*#PATH#g;
            s/0x[0-9a-fA-F]+/X/g; s/\b[0-9a-fA-F]{3,}\b/X/g; s/[0-9]+/N/g' |
    sort -u | sha256sum | cut -c1-16
}

# The stack dumps still have to be there, just not compared slot by slot.
stack_dumps() { grep -c 'guest stack:' "$OUT/$1.log"; }

# --- PS4 Isaac -------------------------------------------------------------
run isaac-snap 120 "$ISAAC" DELTA_GPU_SNAP=120 DELTA_GPU_SNAP_EXIT=1 &&
  { say isaac-boot "$(boot_digest isaac-snap)"
    say isaac-stacks "$(stack_dumps isaac-snap)"; }

# --- the same boot with the instrumentation armed --------------------------
if run isaac-armed 120 "$ISAAC" \
     DELTA_GPU_SNAP=120 DELTA_GPU_SNAP_EXIT=1 \
     DELTA_GUEST_WPROT=200000000:4000:200 \
     DELTA_POOLMAP=all:2000 \
     DELTA_FNWATCH=0x1000:probe; then
  say isaac-armed "$(grep -ohE '\[(wprot|poolmap|census|fnwatch|heapprof|memdump|whist|sumwatch|popcnt)\]' \
                       "$OUT/isaac-armed.log" | sort -u | tr -d '[]' | tr '\n' ',')"
  # One-shot arming lines are deterministic even though the periodic payloads
  # are not; they prove the parse and install paths still run.
  say isaac-armed-install "$(grep -cE 'fnwatch: eboot\+0x1000|watching .* re-armed' \
                               "$OUT/isaac-armed.log")"
fi

# --- the LLE audio daemon actually reaches its sink ------------------------
# A default boot never gets there (the HLE shim answers instead), so the
# daemon's four calls into the host sink would otherwise be uncovered.
# `dropped` is the useful number: it counts blocks the sink refused.
if run lle-audio 45 "$ISAAC" DELTA_LLE=libSceAudioOut DELTA_AUDIO_TRACE=1; then
  say lle-audio "sinks=$(grep -cE '\[audiod\] port [0-9]+ -> sink' "$OUT/lle-audio.log"),$(grep -oE "dropped=[0-9]+" "$OUT/lle-audio.log" | tail -1)"
fi

# --- the fatal path still reports ------------------------------------------
# Inject a real SIGSEGV into a running guest thread. Nothing is armed, so
# probe::onSignal declines it and the crash reporter must produce its dump --
# which is the half of the handler no other check here reaches. An earlier
# version of this check grepped for "<module>+0x..." after arming a breakpoint
# that never executed, and matched ordinary module-load lines instead: it
# passed on every build, including ones where nothing faulted at all.
if [ -f "$ISAAC" ]; then
  rm -rf "$OUT/crash-smoke"; mkdir -p "$OUT/crash-smoke"
  ( cd "$BUILD/delta/main" &&
    env SDL_VIDEODRIVER=offscreen DELTA_GPU_DUMP_DIR="$OUT/crash-smoke" \
      timeout -k 5 --signal=KILL 40 "$BIN" "$ISAAC" ) >"$OUT/crash-smoke.log" 2>&1 &
  emu=$!
  sleep 12
  kill -SEGV "$(ls /proc/$emu/task 2>/dev/null | tail -1)" 2>/dev/null
  sleep 5
  kill -9 $emu 2>/dev/null; wait $emu 2>/dev/null
  grep -q 'GUEST FAULT' "$OUT/crash-smoke.log" &&
    say crash-smoke "DUMPED,maps=$(grep -c 'maps ' "$OUT/crash-smoke.log")" ||
    say crash-smoke NO-DUMP
else
  say crash-smoke "SKIP (missing pkg)"
fi

# --- a second title, to a milestone rather than a hash ---------------------
run undertale 120 "$UNDERTALE" DELTA_GPU_SNAP=200 DELTA_GPU_SNAP_EXIT=1 &&
  say undertale-boot "$(boot_digest undertale)"

echo
if [ "$FP" = "$OUT/current.txt" ] && [ -f "$OUT/baseline.txt" ]; then
  echo "--- baseline vs current ---"
  diff -u "$OUT/baseline.txt" "$FP" && echo IDENTICAL ||
    { echo "DIFFERENCES ABOVE -- explain every one before landing"; exit 1; }
fi
