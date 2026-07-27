#!/usr/bin/env python3
"""Layering check for delta/gpu (a scaled-down Chromium DEPS check).

Rules enforced (see delta/gpu/README.md):
  rhi/     may include only gpu/rhi/           (the backend-free seam)
  ps4/     may include gpu/ps4/, gpu/rhi/, gpu/guest_memory.h
  ps5/     may include gpu/ps5/, gpu/ps4/gcn/, gpu/rhi/, gpu/guest_memory.h
           (the RDNA2 path reuses the GCN->SPIR-V translator infrastructure)
  vulkan/  may include gpu/vulkan/, gpu/rhi/, gpu/shaders/, gpu/guest_memory.h,
           gpu/ps4/gcn/ (recompiled-program types only) -- never a command
           processor or register file
  tests/   may include anything in gpu/
Outside delta/gpu, only the public surface is reachable:
  gpu/rhi/*, gpu/ps4/cmd_processor.h, gpu/ps5/cmd_processor.h

Run from the repo root (or pass it as argv[1]). Exits non-zero on violation.
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
GPU = os.path.join(ROOT, 'delta', 'gpu')
DELTA = os.path.join(ROOT, 'delta')

ALLOWED = {
    'rhi': ('gpu/rhi/',),
    'ps4': ('gpu/ps4/', 'gpu/rhi/', 'gpu/guest_memory.h'),
    # gpu/ps4/pm4.h: AGC command streams are PM4-framed, the packet framing
    # header is shared with the PS4 path.
    'ps5': ('gpu/ps5/', 'gpu/ps4/gcn/', 'gpu/ps4/pm4.h', 'gpu/rhi/',
            'gpu/guest_memory.h'),
    'vulkan': ('gpu/vulkan/', 'gpu/rhi/', 'gpu/shaders/',
               'gpu/guest_memory.h', 'gpu/ps4/gcn/'),
    'tests': ('gpu/',),
    'shaders': (),
    '': (),  # module-root headers depend on nothing in the module
}

PUBLIC = ('gpu/rhi/', 'gpu/ps4/cmd_processor.h', 'gpu/ps5/cmd_processor.h')

INCLUDE = re.compile(r'^\s*#include "(gpu/[^"]+)"', re.M)

failures = []

for cur, _, names in os.walk(GPU):
    rel = os.path.relpath(cur, GPU)
    top = '' if rel == '.' else rel.split(os.sep)[0]
    allowed = ALLOWED.get(top)
    if allowed is None:
        failures.append(f'{cur}: directory not covered by layering rules; '
                        f'add it to check_layering.py')
        continue
    for n in names:
        if not n.endswith(('.cc', '.cpp', '.h')):
            continue
        path = os.path.join(cur, n)
        text = open(path, encoding='utf-8', errors='replace').read()
        for inc in INCLUDE.findall(text):
            if not any(inc.startswith(a) for a in allowed):
                failures.append(f'{path}: includes "{inc}" '
                                f'(not allowed from {top or "module root"}/)')

for cur, dirs, names in os.walk(DELTA):
    if cur.startswith(GPU):
        dirs[:] = []
        continue
    for n in names:
        if not n.endswith(('.cc', '.cpp', '.h')):
            continue
        path = os.path.join(cur, n)
        text = open(path, encoding='utf-8', errors='replace').read()
        for inc in INCLUDE.findall(text):
            if not any(inc.startswith(p) for p in PUBLIC):
                failures.append(f'{path}: includes gpu-internal "{inc}" '
                                f'(public surface is {", ".join(PUBLIC)})')

if failures:
    print(f'{len(failures)} layering violation(s):')
    for f in failures:
        print(' ', f)
    sys.exit(1)
print('gpu layering: OK')
