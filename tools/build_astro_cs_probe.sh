#!/usr/bin/env bash
# Build + run the Astro Bot CS probe (tools/astro_cs_probe.cc) against a freshly
# dumped eboot (/tmp/dumped_module.elf), linking a configured build tree's
# libraries. Run like:
#   nix develop -c bash tools/build_astro_cs_probe.sh 0x201408e6aa00
# BUILD: the build tree to link (default: build).
set -e
cd "$(dirname "$0")/.."

B=${BUILD:-build}
OUT=/tmp/astro_cs_probe
c++ -std=c++20 -DDELTA_HAVE_SPIRV_BACKEND=1 \
  -Ivendor/SPIRV-Headers/include \
  -Idelta -Ishared -Ivendor/equilibrium \
  tools/astro_cs_probe.cc \
  "$B/delta/gpu/libdelta_gpu.a" "$B/shared/libshared.a" \
  "$B/vendor/libeq_base.a" -lpthread \
  -o "$OUT"

exec "$OUT" "$@"
