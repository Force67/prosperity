#!/usr/bin/env bash
#
# boot_sweep.sh <binary> <outdir>
#
# Boot every title we have and record how far each got. Complements
# tools/verify.sh, which only exercises two PS4 pkgs: this covers all three
# container providers (pkg / ffpkg / archive) and both consoles, which is what a
# change to the loader, the VFS or the process model can break without any of
# the fast checks noticing.
#
# Prints one line per title: modules loaded, whether the GPU produced a frame,
# and a digest of the boot log with everything volatile normalised away.
set -uo pipefail

BIN=$(readlink -f "$1")   # the runs cd to the binary dir, so this must be absolute
OUT=$(readlink -f "$2")
mkdir -p "$OUT"

DUMPS=${DUMPS:-/home/vince/Documents/dumps}
FW="${FW:-$DUMPS/PS5/sys_sys_ex/system_system_ex_database/1/Firmware 01.14.00/PS5UPDATE.PUP_out/PROSPEROUPDATE1.PUP/dev/ssd0.system_b_out}"
SECS=${SECS:-60}

# label:path:extra-env
TITLES=(
  "isaac:$DUMPS/The.Binding.of.Isaac.Rebirth.v1.16.PS4-CUSA00792.pkg:"
  "undertale:$DUMPS/Undertale-CUSA09415.pkg:"
  "doom64:$DUMPS/doom64.pkg:"
  "doom_bp:$DUMPS/doom_bp.pkg:"
  "pt:$DUMPS/PT.pkg:"
  "gtasa:$DUMPS/gtasa.pkg:"
  "sotc:$DUMPS/shadowofthecolossus.pkg:"
  "tombraider:$DUMPS/tr.pkg:"
  "uncharted2:$DUMPS/uncharted2.pkg:"
  "bloodborne:$DUMPS/bloodborne.pkg:"
  "isaac_zip:$DUMPS/ps4spec/ISAAAA4.zip:"
  "isaac_ps5:$DUMPS/PPSA03311.ffpkg:PS5"
  "astrobot:$DUMPS/PS5/astrobot.rar:PS5"
  "demonssouls:$DUMPS/PS5/PPSA01342 (1.05).rar:PS5"
)

# Same normalisation as tools/verify.sh: drop the stack-scan residue, then
# every timestamp, path, hex value and digit run (which covers the func:line
# prefix, so relocating code is not reported as a behaviour change).
digest() {
  sed -E 's/\x1b\[[0-9;]*m//g; s/^\[[^]]*\] //' "$1" |
    grep -v 'sp+' |
    sed -E 's#/[^ ]*/(build|\.verify|base_build)[^ ]*#PATH#g; s#/tmp/[^ ]*#PATH#g;
            s/0x[0-9a-fA-F]+/X/g; s/\b[0-9a-fA-F]{3,}\b/X/g; s/[0-9]+/N/g' |
    sort -u | sha256sum | cut -c1-12
}

printf '%-14s %7s %7s %6s %s\n' title modules frames fps digest
for entry in "${TITLES[@]}"; do
  label=${entry%%:*}; rest=${entry#*:}
  path=${rest%:*}; kind=${rest##*:}
  log=$OUT/$label.log
  if [ ! -e "$path" ]; then
    printf '%-14s %7s\n' "$label" MISSING; continue
  fi
  env_extra=()
  [ "$kind" = PS5 ] && env_extra+=("DELTA_PS5_MODULES=$FW/common/lib:$FW/priv/lib")
  ( cd "$(dirname "$BIN")" &&
    env SDL_VIDEODRIVER=offscreen DELTA_GPU_DUMP_DIR="$OUT/$label.d" \
      "${env_extra[@]}" \
      timeout -k 5 --signal=KILL "$SECS" "$BIN" "$path" ) >"$log" 2>&1
  pkill -9 -x ps4delta 2>/dev/null   # ignores SIGTERM; -x so the sweep does not match itself
  mods=$(grep -c 'get_info_ex' "$log")
  # "did the GPU produce anything" -- a flip or a presented frame.
  frames=$(grep -cE '\[(flip|gpuvk|snap)\]' "$log")
  fps=$(grep -oE '[0-9]+\.[0-9]+ fps' "$log" | tail -1 | cut -d' ' -f1)
  printf '%-14s %7s %7s %6s %s\n' "$label" "$mods" "$frames" "${fps:--}" "$(digest "$log")"
done
