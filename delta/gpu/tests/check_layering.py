#!/usr/bin/env python3
"""Layering check for delta/gpu (a scaled-down Chromium DEPS check).

Rules enforced (see delta/gpu/README.md):
  rhi/     may include only gpu/rhi/           (the backend-free seam)
  ps4/     may include gpu/ps4/, gpu/rhi/, gpu/guest_memory.h
  ps5/     may include gpu/ps5/, gpu/gcn/, gpu/rhi/, gpu/guest_memory.h
           (the RDNA2 path reuses the GCN->SPIR-V translator infrastructure)
  vulkan/  may include gpu/vulkan/, gpu/rhi/, gpu/shaders/, gpu/guest_memory.h,
           gpu/gcn/ (recompiled-program types only) -- never a command
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
    # The shared ISA decode + SPIR-V translator both consoles emit through. It
    # is the bottom of the recompiler stack, so it may not reach back up into a
    # console's command processor.
    'gcn': ('gpu/gcn/', 'gpu/guest_memory.h', 'gpu/gpu_check.h'),
    'ps4': ('gpu/ps4/', 'gpu/gcn/', 'gpu/rhi/', 'gpu/guest_memory.h'),
    # gpu/ps4/pm4.h: AGC command streams are PM4-framed, the packet framing
    # header is shared with the PS4 path.
    'ps5': ('gpu/ps5/', 'gpu/gcn/', 'gpu/ps4/pm4.h', 'gpu/rhi/',
            'gpu/guest_memory.h'),
    # The gcn allowance is the two headers the backend actually consumes (the
    # recompiled-program types and the detiler), not the directory: a backend
    # reaching into the decoder or spirv/ internals is a layering bug.
    'vulkan': ('gpu/vulkan/', 'gpu/rhi/', 'gpu/shaders/',
               'gpu/guest_memory.h', 'gpu/gpu_check.h',
               'gpu/gcn/gcn_translate.h', 'gpu/gcn/gcn_detile.h'),
    'tests': ('gpu/',),
    'shaders': (),
    '': (),  # module-root headers depend on nothing in the module
}

PUBLIC = ('gpu/rhi/', 'gpu/ps4/cmd_processor.h', 'gpu/ps5/cmd_processor.h')

# Developer harnesses that test a gpu internal directly, the same role as
# gpu/tests/. They are not emulator code and nothing links them, so the public
# surface does not apply.
INTERNAL_HARNESSES = (os.path.join('tools', 'spv_selftest.cpp'),)

# Quoted or angle form: DELTA_ROOT is a plain -I directory, so both spellings
# resolve and both must be policed.
INCLUDE = re.compile(r'^\s*#include ["<](gpu/[^">]+)[">]', re.M)
# Any quoted include: used to catch the includer-relative bypasses (a bare
# "vk_device.h" or a "../vulkan/..." path compiles without ever naming gpu/).
ANY_QUOTED = re.compile(r'^\s*#include "([^"]+)"', re.M)

EXTS = ('.cc', '.cpp', '.h', '.hpp', '.inl')

failures = []


def gpu_includes(path, text):
    """Yield every include of a gpu header, canonicalised to its gpu/... form.

    Catches the quoted/angle "gpu/..." spellings plus the includer-relative
    forms (bare filename, ../ paths) that quoted lookup resolves without the
    gpu/ prefix; those are reported as violations of the spelling rule."""
    for inc in INCLUDE.findall(text):
        yield inc
    for inc in ANY_QUOTED.findall(text):
        if inc.startswith('gpu/'):
            continue  # already handled above
        resolved = os.path.normpath(os.path.join(os.path.dirname(path), inc))
        if os.path.abspath(resolved).startswith(os.path.abspath(GPU) + os.sep) \
                and os.path.exists(resolved):
            failures.append(f'{path}: includes "{inc}" relative to the '
                            f'includer; spell gpu includes from the delta '
                            f'root ("gpu/...")')


for cur, _, names in os.walk(GPU):
    rel = os.path.relpath(cur, GPU)
    top = '' if rel == '.' else rel.split(os.sep)[0]
    allowed = ALLOWED.get(top)
    if allowed is None:
        failures.append(f'{cur}: directory not covered by layering rules; '
                        f'add it to check_layering.py')
        continue
    for n in names:
        if not n.endswith(EXTS):
            continue
        path = os.path.join(cur, n)
        text = open(path, encoding='utf-8', errors='replace').read()
        for inc in gpu_includes(path, text):
            if not any(inc.startswith(a) for a in allowed):
                failures.append(f'{path}: includes "{inc}" '
                                f'(not allowed from {top or "module root"}/)')

for outside in (DELTA, os.path.join(ROOT, 'shared'), os.path.join(ROOT, 'tools')):
    for cur, dirs, names in os.walk(outside):
        if cur.startswith(GPU):
            dirs[:] = []
            continue
        for n in names:
            if not n.endswith(EXTS):
                continue
            path = os.path.join(cur, n)
            if any(os.path.normpath(path).endswith(h)
                   for h in INTERNAL_HARNESSES):
                continue
            text = open(path, encoding='utf-8', errors='replace').read()
            for inc in gpu_includes(path, text):
                if not any(inc.startswith(p) for p in PUBLIC):
                    failures.append(f'{path}: includes gpu-internal "{inc}" '
                                    f'(public surface is {", ".join(PUBLIC)})')

if failures:
    print(f'{len(failures)} layering violation(s):')
    for f in failures:
        print(' ', f)
    sys.exit(1)
print('gpu layering: OK')
