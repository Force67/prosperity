#!/usr/bin/env bash
# Build + run the Astro Bot CS probe (tools/astro_cs_probe.cc) against a freshly
# dumped eboot (/tmp/dumped_module.elf). Run like:
#   nix develop -c bash tools/build_astro_cs_probe.sh 0x201408e6aa00
set -e
cd "$(dirname "$0")/.."

G=delta/gpu
OUT=/tmp/astro_cs_probe
c++ -std=c++20 -DDELTA_HAVE_SPIRV_BACKEND=1 \
  $(pkg-config --cflags SPIRV-Headers SPIRV-Tools) \
  -Idelta -I"$G" -I"$G/ps4" -Ishared -Ivendor/equilibrium \
  tools/astro_cs_probe.cc \
  "$G/ps5/rdna/rdna_decode.cc" \
  "$G/ps5/rdna/rdna_translate.cc" \
  "$G/ps5/rdna/rdna_resource.cc" \
  "$G/ps5/rdna/rdna_compute.cc" \
  "$G/gcn/gcn_decode.cc" \
  "$G/gcn/gcn_resource.cc" \
  "$G/gcn/gcn_disasm.cc" \
  "$G/gcn/gcn_audit.cc" \
  shared/utl/mem_posix.cpp \
  "$G/gcn/spirv/gcn_spirv.cc" \
  "$G/gcn/spirv/translate_alu.cc" \
  "$G/gcn/spirv/translate_neo.cc" \
  "$G/gcn/spirv/translate_mem.cc" \
  "$G/gcn/spirv/spv_emit.cc" \
  "$G/gcn/spirv/spv_post.cc" \
  "$G/gcn/gcn_translate.cc" \
  vendor/equilibrium/base/logging.cc \
  vendor/equilibrium/base/strings/format.cc \
  vendor/equilibrium/base/strings/base_string.cc \
  $(pkg-config --libs SPIRV-Tools) \
  -o "$OUT"

exec "$OUT" "$@"
