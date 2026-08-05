#!/usr/bin/env python3
"""Query a GPU frame capture written by delta/gpu (DELTA_GPU_CAPTURE=<frame>).

A capture is one guest frame as JSONL: every region, draw, dispatch, barrier,
memory fill, decline, validation message and resource dump, in order and
uncapped. This reads one and answers the questions that actually come up while
debugging a renderer, as plain greppable text.

  gpu_capture.py summary    <capture>
  gpu_capture.py draws      <capture> [--rt ADDR] [--ps ADDR] [--grep TEXT]
  gpu_capture.py draw       <capture> N
  gpu_capture.py wrote      <capture> ADDR      what wrote render target ADDR
  gpu_capture.py sampled    <capture> ADDR      which draws sampled ADDR
  gpu_capture.py graph      <capture>           producer/consumer graph
  gpu_capture.py textures   <capture> [--zero]  every distinct T# + its memory
  gpu_capture.py cbufs      <capture> [--draw N]
  gpu_capture.py barriers   <capture> [--resource TEXT]
  gpu_capture.py dumps      <capture>           the PNGs and their statistics
  gpu_capture.py validation <capture>
  gpu_capture.py zero       <capture>           everything silently reading 0
  gpu_capture.py timeline   <capture>           the raw event stream, one line each
  gpu_capture.py diff       <capture-a> <capture-b>

<capture> is a .jsonl file, or a directory (the newest frame_*.jsonl in it).
"""

import argparse
import collections
import glob
import json
import os
import sys


# --- loading ---------------------------------------------------------------

def resolve(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'frame_*.jsonl')),
                       key=os.path.getmtime)
        if not files:
            sys.exit(f'no frame_*.jsonl under {path}')
        return files[-1]
    return path


def load(path):
    events = []
    with open(resolve(path)) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'{path}:{n}: unparsable ({e})', file=sys.stderr)
    return events


def of(events, kind):
    return [e for e in events if e.get('t') == kind]


def addr(text):
    """Accept 0x8142f00000, 8142f00000 or a decimal, return the canonical hex."""
    if isinstance(text, int):
        return hex(text)
    text = text.strip()
    try:
        return hex(int(text, 16 if text.lower().startswith('0x') else 0))
    except ValueError:
        try:
            return hex(int(text, 16))
        except ValueError:
            return text


def same(a, b):
    return a is not None and b is not None and addr(a) == addr(b)


# --- helpers ---------------------------------------------------------------

def draw_targets(d):
    """Every colour target the draw writes."""
    return [ct['base'] for ct in d.get('color_targets', [])
            if ct.get('base') and addr(ct['base']) != '0x0']


def draw_sources(d):
    """(binding, address, how) for every sampler binding that resolved to an
    image the frame itself produced."""
    out = []
    for t in d.get('textures', []):
        how = t.get('resolved', 'unknown')
        base = t.get('resolved_base')
        if not base or addr(base) == '0x0':
            base = t.get('base')
        if base and addr(base) != '0x0':
            out.append((t.get('i'), base, how))
    return out


def shader_id(d, stage):
    s = d.get(stage, {}) or {}
    return s.get('spirv_hash') or s.get('guest_hash') or s.get('addr') or '?'


def fmt_draw_line(d):
    tex = d.get('textures', [])
    resolved = collections.Counter(t.get('resolved', '?') for t in tex)
    how = ' '.join(f'{k}={v}' for k, v in sorted(resolved.items()))
    n = d.get('index_count') if d.get('indexed') else d.get('vertex_count')
    return (f"draw {d.get('draw'):4d} rt={d.get('rt')} {d.get('rt_w')}x"
            f"{d.get('rt_h')} {d.get('path'):6s} "
            f"{'idx' if d.get('indexed') else 'seq'}={n} "
            f"mrt={len(d.get('color_targets', []))} "
            f"vs={d.get('vs', {}).get('addr')} ps={d.get('ps', {}).get('addr')} "
            f"tex={len(tex)}{(' [' + how + ']') if how else ''}")


# --- commands --------------------------------------------------------------

def cmd_summary(args):
    events = load(args.capture)
    head = (of(events, 'capture') or [{}])[0]
    end = (of(events, 'frame_end') or [{}])[0]
    draws = of(events, 'draw')
    print(f"capture      {resolve(args.capture)}")
    print(f"frame        {head.get('frame')}  (armed at {head.get('armed_frame')})")
    print(f"exposure     {head.get('exposure')}  gamma {head.get('gamma')}")
    print(f"events       {len(events)}")
    print(f"draws        {len(draws)}  "
          f"(recomp {sum(1 for d in draws if d.get('path') == 'recomp')}, "
          f"quad {sum(1 for d in draws if d.get('path') == 'quad')})")
    print(f"regions      {len(of(events, 'region_begin'))}")
    print(f"dispatches   {len(of(events, 'dispatch'))}")
    print(f"barriers     {len(of(events, 'barrier'))}")
    print(f"memory fills {len(of(events, 'memory_fill'))}")
    dec = collections.Counter(e['reason'] for e in of(events, 'decline'))
    if dec:
        print("declines     " + ', '.join(f'{k}={v}' for k, v in dec.most_common()))
    val = collections.Counter(e['severity'] for e in of(events, 'validation'))
    if val:
        print("validation   " + ', '.join(f'{k}={v}' for k, v in val.most_common()))
    print(f"scanout      {end.get('scanout')}")

    print("\nrender targets written this frame:")
    per_rt = collections.Counter()
    for d in draws:
        for base in draw_targets(d):
            per_rt[base] += 1
    dumps = {e['base']: e for e in of(events, 'dump')
             if e.get('when') == 'frame-end' and e.get('kind') == 'rt'}
    for base, n in sorted(per_rt.items(), key=lambda kv: -kv[1]):
        dump = dumps.get(base, {})
        stats = dump.get('stats', {})
        extra = ''
        if stats:
            mean = stats.get('mean', [0, 0, 0, 0])
            extra = (f"  mean=({mean[0]:.4g},{mean[1]:.4g},{mean[2]:.4g}) "
                     f"nonzero={stats.get('nonzero')}/{stats.get('texels')} "
                     f"nan={stats.get('nan')}")
        print(f"  {base} draws={n:4d} {dump.get('format', '')}{extra}")
    print("\nrender targets SAMPLED this frame:")
    per_src = collections.Counter()
    for d in draws:
        for _, base, how in draw_sources(d):
            if how in ('rt', 'depth', 'feedback', 'storage'):
                per_src[(base, how)] += 1
    for (base, how), n in sorted(per_src.items(), key=lambda kv: -kv[1]):
        print(f"  {base} as {how:8s} by {n} draws")


def cmd_draws(args):
    for d in of(load(args.capture), 'draw'):
        if args.rt and not any(same(t, args.rt) for t in draw_targets(d)):
            continue
        if args.ps and not same(d.get('ps', {}).get('addr'), args.ps):
            continue
        line = fmt_draw_line(d)
        if args.grep and args.grep not in json.dumps(d):
            continue
        print(line)


def cmd_draw(args):
    for d in of(load(args.capture), 'draw'):
        if d.get('draw') == args.n:
            print(json.dumps(d, indent=2))
            return
    sys.exit(f'no draw {args.n} in the capture')


def cmd_wrote(args):
    events = load(args.capture)
    want = args.address
    print(f'writers of {addr(want)}:')
    for e in events:
        if e.get('t') == 'region_begin':
            for c in e.get('color', []):
                if same(c.get('base'), want):
                    print(f"  region  after_draw={e.get('after_draw')} "
                          f"{e.get('w')}x{e.get('h')} loadOp={c.get('load_op')} "
                          f"format={c.get('format')}")
        elif e.get('t') == 'draw':
            if any(same(t, want) for t in draw_targets(e)):
                print('  ' + fmt_draw_line(e))
        elif e.get('t') == 'dispatch':
            for r in e.get('resources', []):
                if same(r.get('base'), want) and r.get('written'):
                    print(f"  dispatch cs={e.get('cs', {}).get('addr')} "
                          f"binding={r.get('binding')} size={r.get('size')}")
        elif e.get('t') == 'memory_fill':
            if same(e.get('base'), want):
                print(f"  cp-dma fill after_draw={e.get('after_draw')} "
                      f"bytes={e.get('bytes')} value={e.get('value')}")


def cmd_sampled(args):
    want = args.address
    print(f'readers of {addr(want)}:')
    for d in of(load(args.capture), 'draw'):
        for t in d.get('textures', []):
            if same(t.get('base'), want) or same(t.get('resolved_base'), want):
                g = t.get('guest', {})
                print(f"  draw {d.get('draw'):4d} binding {t.get('i')} "
                      f"as {t.get('resolved')} rt={d.get('rt')} "
                      f"{t.get('w')}x{t.get('h')} {t.get('format')} "
                      f"src={t.get('src')} "
                      f"guest_readable={g.get('readable')} "
                      f"nonzero={g.get('nonzero')}/{g.get('sampled')}")


def cmd_graph(args):
    events = load(args.capture)
    draws = of(events, 'draw')
    edges = collections.Counter()
    produced = collections.Counter()
    for d in draws:
        targets = draw_targets(d)
        for t in targets:
            produced[t] += 1
        for _, base, how in draw_sources(d):
            if how in ('rt', 'depth', 'feedback', 'storage'):
                for t in targets:
                    edges[(base, t, how)] += 1
    print('producer -> consumer (one line per distinct edge):')
    for (src, dst, how), n in sorted(edges.items(), key=lambda kv: -kv[1]):
        print(f'  {src} --{how}--> {dst}   ({n} draws)')
    sources = {src for src, _, _ in edges}
    print('\ntargets nothing in this frame sampled (leaves / scanout):')
    for t in sorted(set(produced) - sources):
        print(f'  {t}  ({produced[t]} draws)')
    print('\ntargets sampled but never written in this frame:')
    for s in sorted(sources - set(produced)):
        print(f'  {s}')


def cmd_textures(args):
    events = load(args.capture)
    seen = {}
    for d in of(events, 'draw'):
        for t in d.get('textures', []):
            key = (t.get('base'), t.get('w'), t.get('h'), t.get('dfmt'),
                   t.get('nfmt'))
            entry = seen.setdefault(key, {'tex': t, 'draws': [], 'how':
                                          collections.Counter()})
            entry['draws'].append(d.get('draw'))
            entry['how'][t.get('resolved', '?')] += 1
    dumps = {(e.get('base'), e.get('w'), e.get('h')): e
             for e in of(events, 'dump') if e.get('kind') == 'tex'}
    for key, entry in sorted(seen.items(), key=lambda kv: -len(kv[1]['draws'])):
        t = entry['tex']
        g = t.get('guest', {})
        zero = g.get('readable') and g.get('nonzero') == 0
        if args.zero and not (zero or not g.get('readable')
                              or addr(t.get('base') or '0') == '0x0'):
            continue
        flag = ''
        if addr(t.get('base') or '0') == '0x0':
            flag = '  <- NULL DESCRIPTOR'
        elif not g.get('readable'):
            flag = '  <- UNREADABLE GUEST MEMORY'
        elif zero:
            flag = '  <- ALL ZERO IN GUEST MEMORY'
        how = ' '.join(f'{k}={v}' for k, v in entry['how'].most_common())
        d = dumps.get((t.get('base'), t.get('w'), t.get('h')), {})
        png = d.get('file') or (f"(skipped: {d.get('skipped')})"
                                if d.get('skipped') else '')
        print(f"{t.get('base')} {t.get('w')}x{t.get('h')} "
              f"dfmt={t.get('dfmt')} nfmt={t.get('nfmt')} "
              f"tiling={t.get('tiling')} mips={t.get('mip_levels')} "
              f"layers={t.get('layers')} src={t.get('src')} "
              f"[{how}] draws={len(entry['draws'])} "
              f"nonzero={g.get('nonzero')}/{g.get('sampled')}{flag}")
        if png:
            print(f'    {png}')


def cmd_cbufs(args):
    for d in of(load(args.capture), 'draw'):
        if args.draw is not None and d.get('draw') != args.draw:
            continue
        for c in d.get('cbufs', []):
            g = c.get('guest', {})
            data = c.get('data', '')
            floats = ''
            if data:
                import struct
                words = bytes.fromhex(data[:8 * 8 * 2])
                vals = struct.unpack(f'<{len(words) // 4}f', words[:len(words) // 4 * 4])
                floats = ' '.join(f'{v:.6g}' for v in vals)
            print(f"draw {d.get('draw'):4d} cb{c.get('i')} {c.get('base')} "
                  f"size={c.get('size')} staged={c.get('staged')} "
                  f"readable={g.get('readable')} "
                  f"nonzero={g.get('nonzero')}/{g.get('sampled')}")
            if floats:
                print(f'    f32: {floats}')


def cmd_barriers(args):
    for e in of(load(args.capture), 'barrier'):
        if args.resource and args.resource not in (e.get('resource') or ''):
            continue
        print(f"after_draw={e.get('after_draw'):4d} {e.get('aspect'):7s} "
              f"{e.get('resource') or e.get('image')}: "
              f"{e.get('from')} -> {e.get('to')} "
              f"src={e.get('src_access')} dst={e.get('dst_access')}")


def cmd_dumps(args):
    for e in of(load(args.capture), 'dump'):
        s = e.get('stats', {})
        extra = ''
        if s:
            mean = s.get('mean', [0, 0, 0, 0])
            extra = (f" mean=({mean[0]:.4g},{mean[1]:.4g},{mean[2]:.4g},"
                     f"{mean[3]:.4g}) nonzero={s.get('nonzero')}/"
                     f"{s.get('texels')} nan={s.get('nan')}")
        when = e.get('when')
        at = f" after_draw={e.get('after_draw')}" if e.get('after_draw') is not None else ''
        print(f"{e.get('kind'):5s} {when}{at} {e.get('base')} "
              f"{e.get('w')}x{e.get('h')} {e.get('format')}{extra}")
        print(f"    {e.get('file') or '(skipped: ' + str(e.get('skipped')) + ')'}")


def cmd_validation(args):
    for e in of(load(args.capture), 'validation'):
        print(f"[{e.get('severity')}] draw={e.get('draw')} {e.get('id')}\n"
              f"    labels: {e.get('labels')}\n"
              f"    {e.get('message')}")


def cmd_zero(args):
    """Everything the frame is silently reading as zero.

    A binding that does not resolve is bound to a zero window or a 1x1 default,
    which is indistinguishable in the output from data that is genuinely zero --
    so it has to be named."""
    events = load(args.capture)
    print('sampler bindings that resolved to the default (1x1 white/zero):')
    for d in of(events, 'draw'):
        for t in d.get('textures', []):
            if t.get('resolved') == 'default':
                print(f"  draw {d.get('draw'):4d} binding {t.get('i')} "
                      f"base={t.get('base')} src={t.get('src')} "
                      f"{t.get('w')}x{t.get('h')} {t.get('format')} "
                      f"ps={d.get('ps', {}).get('addr')}")
    print('\nsampler bindings whose guest memory is all zero:')
    for d in of(events, 'draw'):
        for t in d.get('textures', []):
            g = t.get('guest', {})
            if t.get('resolved') == 'guest' and g.get('readable') \
                    and g.get('nonzero') == 0:
                print(f"  draw {d.get('draw'):4d} binding {t.get('i')} "
                      f"base={t.get('base')} {t.get('w')}x{t.get('h')} "
                      f"{t.get('format')}")
    print('\nconstant buffers that did NOT stage real guest memory:')
    for d in of(events, 'draw'):
        for c in d.get('cbufs', []):
            if not c.get('staged'):
                print(f"  draw {d.get('draw'):4d} cb{c.get('i')} "
                      f"{c.get('base')} size={c.get('size')} "
                      f"readable={c.get('guest', {}).get('readable')}")
    print('\nraw (set-2) buffers that did NOT stage real guest memory:')
    for d in of(events, 'draw'):
        for b in d.get('bufs', []):
            if not b.get('staged'):
                print(f"  draw {d.get('draw'):4d} buf{b.get('i')} "
                      f"{b.get('base')} size={b.get('size')} "
                      f"readable={b.get('guest', {}).get('readable')}")
    print('\nrender targets that ended the frame empty:')
    for e in of(events, 'dump'):
        s = e.get('stats') or {}
        if e.get('kind') == 'rt' and e.get('when') == 'frame-end' \
                and s.get('nonzero') == 0:
            print(f"  {e.get('base')} {e.get('w')}x{e.get('h')} "
                  f"{e.get('format')}")


def cmd_timeline(args):
    for e in load(args.capture):
        t = e.get('t')
        if t == 'draw':
            print(f"{e.get('seq'):5d} " + fmt_draw_line(e))
        elif t == 'region_begin':
            targets = ','.join(c.get('base', '?') for c in e.get('color', []))
            print(f"{e.get('seq'):5d} region_begin {e.get('w')}x{e.get('h')} "
                  f"color=[{targets}] depth={e.get('depth')} "
                  f"loadOps=[{','.join(c.get('load_op', '?') for c in e.get('color', []))}]")
        elif t == 'region_end':
            print(f"{e.get('seq'):5d} region_end")
        elif t == 'barrier':
            print(f"{e.get('seq'):5d} barrier {e.get('aspect')} "
                  f"{e.get('resource') or e.get('image')} "
                  f"{e.get('from')} -> {e.get('to')}")
        elif t == 'dispatch':
            print(f"{e.get('seq'):5d} dispatch cs={e.get('cs', {}).get('addr')} "
                  f"groups={e.get('groups')} res={len(e.get('resources', []))}")
        elif t == 'memory_fill':
            print(f"{e.get('seq'):5d} memory_fill {e.get('base')} "
                  f"bytes={e.get('bytes')} value={e.get('value')}")
        elif t == 'decline':
            print(f"{e.get('seq'):5d} decline {e.get('reason')}")
        elif t == 'validation':
            print(f"{e.get('seq'):5d} validation [{e.get('severity')}] "
                  f"{e.get('id')}")
        elif t == 'dump':
            print(f"{e.get('seq'):5d} dump {e.get('kind')} {e.get('base')} "
                  f"-> {e.get('file')}")
        else:
            print(f"{e.get('seq', 0):5d} {t} {json.dumps(e)}")


def cmd_diff(args):
    a, b = load(args.a), load(args.b)
    da, db = of(a, 'draw'), of(b, 'draw')
    print(f'draws: {len(da)} -> {len(db)}')

    def key(d):
        return (shader_id(d, 'vs'), shader_id(d, 'ps'))
    ka = collections.Counter(key(d) for d in da)
    kb = collections.Counter(key(d) for d in db)
    for k in sorted(set(ka) | set(kb)):
        if ka[k] != kb[k]:
            print(f'  shader pair vs={k[0]} ps={k[1]}: {ka[k]} -> {kb[k]}')

    fields = ['rt', 'rt_w', 'rt_h', 'path', 'target_mask', 'shader_mask',
              'blend_enable', 'blend_control', 'indexed', 'index_count',
              'vertex_count', 'instance_count', 'prim_type', 'cull_mode',
              'clear_rect']
    changed = 0
    for x, y in zip(da, db):
        deltas = [f'{f}: {x.get(f)} -> {y.get(f)}'
                  for f in fields if x.get(f) != y.get(f)]
        tx = {t.get('i'): t for t in x.get('textures', [])}
        ty = {t.get('i'): t for t in y.get('textures', [])}
        for i in sorted(set(tx) | set(ty)):
            u, v = tx.get(i, {}), ty.get(i, {})
            for f in ('base', 'resolved', 'resolved_base', 'format'):
                if u.get(f) != v.get(f):
                    deltas.append(f'tex{i}.{f}: {u.get(f)} -> {v.get(f)}')
        if deltas:
            changed += 1
            print(f"draw {x.get('draw')}:")
            for d in deltas:
                print(f'    {d}')
    print(f'{changed} draws differ in state')

    for kind in ('dump',):
        sa = {(e.get('kind'), e.get('base')): e.get('stats', {})
              for e in of(a, kind) if e.get('when') == 'frame-end'}
        sb = {(e.get('kind'), e.get('base')): e.get('stats', {})
              for e in of(b, kind) if e.get('when') == 'frame-end'}
        for k in sorted(set(sa) | set(sb)):
            ma = (sa.get(k) or {}).get('mean')
            mb = (sb.get(k) or {}).get('mean')
            if ma != mb:
                print(f'{k[0]} {k[1]} mean {ma} -> {mb}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def add(name, fn, *, address=False, n=False):
        s = sub.add_parser(name)
        s.add_argument('capture')
        if address:
            s.add_argument('address')
        if n:
            s.add_argument('n', type=int)
        s.set_defaults(fn=fn)
        return s

    add('summary', cmd_summary)
    s = add('draws', cmd_draws)
    s.add_argument('--rt')
    s.add_argument('--ps')
    s.add_argument('--grep')
    add('draw', cmd_draw, n=True)
    add('wrote', cmd_wrote, address=True)
    add('sampled', cmd_sampled, address=True)
    add('graph', cmd_graph)
    s = add('textures', cmd_textures)
    s.add_argument('--zero', action='store_true',
                   help='only bindings reading nothing')
    s = add('cbufs', cmd_cbufs)
    s.add_argument('--draw', type=int)
    s = add('barriers', cmd_barriers)
    s.add_argument('--resource')
    add('dumps', cmd_dumps)
    add('validation', cmd_validation)
    add('zero', cmd_zero)
    add('timeline', cmd_timeline)
    sd = sub.add_parser('diff')
    sd.add_argument('a')
    sd.add_argument('b')
    sd.set_defaults(fn=cmd_diff)

    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
