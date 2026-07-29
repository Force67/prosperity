#!/usr/bin/env bash
# Build + run the RDNA2 recompiler self-test (tools/rdna_selftest.cpp). Run
# inside the nix dev shell:  nix develop -c bash tools/build_rdna_selftest.sh
set -e
cd "$(dirname "$0")/.."

G=delta/gpu
OUT=/tmp/rdna_selftest
c++ -std=c++20 -DDELTA_HAVE_SPIRV_BACKEND=1 \
  $(pkg-config --cflags SPIRV-Headers SPIRV-Tools) \
  -Idelta -I"$G" -I"$G/ps4" -Ishared \
  tools/rdna_selftest.cpp \
  "$G/ps5/rdna/rdna_decode.cc" \
  "$G/ps5/rdna/rdna_translate.cc" \
  "$G/ps5/rdna/rdna_resource.cc" \
  "$G/ps4/gcn/gcn_decode.cc" \
  "$G/ps4/gcn/gcn_resource.cc" \
  "$G/ps4/gcn/gcn_disasm.cc" \
  "$G/ps4/gcn/gcn_audit.cc" \
  shared/utl/mem_posix.cpp \
  "$G/ps4/gcn/spirv/gcn_spirv.cc" \
  "$G/ps4/gcn/spirv/translate_alu.cc" \
  "$G/ps4/gcn/spirv/translate_neo.cc" \
  "$G/ps4/gcn/spirv/translate_mem.cc" \
  "$G/ps4/gcn/spirv/spv_emit.cc" \
  "$G/ps4/gcn/spirv/spv_post.cc" \
  $(pkg-config --libs SPIRV-Tools) \
  -o "$OUT"
echo "built $OUT"
exec "$OUT"
