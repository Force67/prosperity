#!/usr/bin/env python3
"""Module-level layering check for delta/ (a scaled-down Chromium DEPS check).

Every module under delta/ shares one include root, so any header of any module
is reachable from any file and a wrong-way dependency compiles silently. This
records which edges exist and fails when a new one appears.

It is a RATCHET, not a description of the intended design. Edges that should
not exist are listed under GRANDFATHERED with the reason and what would remove
them; the file only ever shrinks. Adding an entry to GRANDFATHERED to make a
build pass defeats the point -- fix the direction instead.

delta/gpu additionally polices its own internals; see gpu/tests/check_layering.py.

Run from the repo root (or pass it as argv[1]). Exits non-zero on violation.
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
DELTA = os.path.join(ROOT, 'delta')

MODULES = ('cpu', 'crypto', 'formats', 'gfx', 'gpu', 'kern', 'main', 'runtime')

# The one-way design. Read down: a module may only include modules below it.
#
#   main        composition root; wires everything, owns no policy
#   runtime     the HLE modules a title imports (vprx) + the code lifter
#   kern        the process model, lv2 syscalls, devices, VFS
#   gpu         guest command streams -> rendered frames
#   gfx         the host platform shell (window, input, audio, overlay)
#   formats     container parsers (pkg, pup, ufs2, archive) -- depends on nothing
#   crypto      primitives -- depends on nothing
ALLOWED = {
    'crypto': (),
    'formats': (),
    'gfx': (),
    'gpu': ('gfx',),
    'kern': ('crypto', 'formats', 'gpu', 'gfx'),
    'cpu': (),
    'runtime': ('kern', 'cpu', 'gpu', 'gfx', 'crypto', 'formats'),
    'main': MODULES,
}

# Wrong-way edges that exist today. Each entry is a debt, not a permission.
GRANDFATHERED = {
    # delta/CMakeLists.txt declares the static-library cycle these create.
    #
    # kern -> cpu is proc/module/threads calling cpu::backend() to run guest
    # code; cpu -> kern is the FEX backend calling back for syscall dispatch,
    # module info and the fatal handler. That is one interface (a host-services
    # client the backend is handed) inverted the wrong way. Removing it means
    # declaring that client in cpu/ and having kern install it, the same shape
    # as krnl::setCsRangeDescriber and krnl::ps4::setAudioSink.
    ('kern', 'cpu'),
    ('cpu', 'kern'),
    # kern -> runtime is the loader asking vprx to resolve an HLE import and
    # the lifter rewriting guest code; runtime -> kern is every HLE module
    # calling the kernel it sits on. Either invert the resolver (kern declares
    # it, runtime registers into it) or declare kern+runtime one layer and say
    # so here -- but decide, because right now neither is written down.
    ('kern', 'runtime'),
}

INCLUDE = re.compile(
    r'^\s*#include ["<](%s)/[^">]+[">]' % '|'.join(MODULES), re.M)
EXTS = ('.cc', '.cpp', '.h', '.hpp', '.inl')

failures = []
seen = set()

for module in MODULES:
    top = os.path.join(DELTA, module)
    for cur, _, names in os.walk(top):
        for n in names:
            if not n.endswith(EXTS):
                continue
            path = os.path.join(cur, n)
            text = open(path, encoding='utf-8', errors='replace').read()
            for dep in INCLUDE.findall(text):
                if dep == module:
                    continue
                seen.add((module, dep))
                if dep in ALLOWED[module] or (module, dep) in GRANDFATHERED:
                    continue
                failures.append(
                    f'{os.path.relpath(path, ROOT)}: {module} -> {dep} '
                    f'({module} may include {", ".join(ALLOWED[module]) or "nothing"})')

# A debt that has been paid should stop being carried, or the file rots into a
# list of edges nobody has checked in a year.
for edge in sorted(GRANDFATHERED - seen):
    failures.append(f'{edge[0]} -> {edge[1]} is listed as grandfathered but no '
                    f'longer exists; delete it from check_module_layering.py')

if failures:
    print(f'{len(failures)} module layering violation(s):')
    for f in failures:
        print(' ', f)
    sys.exit(1)
print(f'module layering: OK ({len(seen)} edges, '
      f'{len(GRANDFATHERED)} grandfathered)')
