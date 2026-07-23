#!/usr/bin/env bash
# Build + run the RDNA2 recompiler self-test (tools/rdna_selftest.cpp). Run
# inside the nix dev shell:  nix develop -c bash tools/build_rdna_selftest.sh
set -e
cd "$(dirname "$0")/.."

G=delta/gpu
OUT=/tmp/rdna_selftest
c++ -std=c++20 -DDELTA_HAVE_SPIRV_BACKEND=1 \
  $(pkg-config --cflags SPIRV-Headers SPIRV-Tools) \
  -I"$G" -I"$G/ps4" -Ishared \
  tools/rdna_selftest.cpp \
  "$G/ps5/rdna/rdna_decode.cpp" \
  "$G/ps5/rdna/rdna_translate.cpp" \
  "$G/ps4/gcn/gcn_decode.cpp" \
  "$G/ps4/gcn/gcn_resource.cpp" \
  shared/utl/mem_posix.cpp \
  "$G/ps4/gcn/spirv/gcn_spirv.cpp" \
  "$G/ps4/gcn/spirv/translate_alu.cpp" \
  "$G/ps4/gcn/spirv/translate_mem.cpp" \
  "$G/ps4/gcn/spirv/spv_emit.cpp" \
  "$G/ps4/gcn/spirv/spv_post.cpp" \
  $(pkg-config --libs SPIRV-Tools) \
  -o "$OUT"
echo "built $OUT"
exec "$OUT"
