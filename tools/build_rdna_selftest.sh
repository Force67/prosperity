#!/usr/bin/env bash
# Build + run the RDNA2 recompiler self-test (tools/rdna_selftest.cc) against
# a configured build tree's libraries. Run inside the nix dev shell:
#   nix develop -c bash tools/build_rdna_selftest.sh
# BUILD: the build tree to link (default: build).
set -e
cd "$(dirname "$0")/.."

B=${BUILD:-build}
OUT=/tmp/rdna_selftest
c++ -std=c++20 -DDELTA_HAVE_SPIRV_BACKEND=1 \
  $(pkg-config --cflags SPIRV-Headers SPIRV-Tools) \
  -Idelta -Ishared -Ivendor/equilibrium \
  tools/rdna_selftest.cc \
  "$B/delta/gpu/libdelta_gpu.a" "$B/shared/libshared.a" \
  "$B/vendor/libeq_base.a" -lpthread \
  $(pkg-config --libs SPIRV-Tools) \
  -o "$OUT"
echo "built $OUT"
exec "$OUT"
