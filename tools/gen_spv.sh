#!/usr/bin/env bash
# Compile a GLSL shader to a C header (uint32 SPIR-V array). Run inside the
# glslang nix shell. usage: gen_spv.sh <shader.(vert|frag)> <varname> <out.h>
set -e
SRC="$1"; VAR="$2"; OUT="$3"
TMP=$(mktemp --suffix=.spv)
STAGE="${SRC##*.}"
glslangValidator -V -S "$STAGE" "$SRC" -o "$TMP"
{
  echo "// Generated."
  echo "#pragma once"
  echo "#include <cstdint>"
  printf 'static const uint32_t %s[]={' "$VAR"
  python3 - "$TMP" <<'PY'
import sys,struct
d=open(sys.argv[1],'rb').read()
n=len(d)//4
vals=struct.unpack('<%dI'%n,d)
sys.stdout.write(','.join('0x%08x'%v for v in vals))
PY
  echo "};"
} > "$OUT"
echo "wrote $OUT"
