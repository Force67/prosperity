#!/usr/bin/env python3
"""Replay vertex shader invocations from a GPU capture on the CPU and diff
them against what the recompiled shader computed on the GPU.

The GPU side comes from the VS register probe: it writes four registers after
one instruction for every vertex of the probed shader. Capture a frame with it:

  tools/drun.py dc -t 120 --capture t=80 -o DELTA_GPU_SHADER_CACHE=0 \\
      -o DELTA_GPU_GCN_VSVGPR_VS=0x50030e3900 \\
      -o DELTA_GPU_GCN_VSVGPR_PC=0xf1 -o DELTA_GPU_GCN_VSVGPR=30

(PC is the instruction index from the capture's listing, the probe records the
registers right after it; registers >= 1000 are SGPRs. PC 0xffff records the
exported position instead.) Then:

  tools/gcn_vs_oracle.py /tmp/drun/cap --pc 0xf1 --reg 30

Draws of the probed shader carry full buffer contents and the fetch shader
listing, so the replay needs nothing else. It runs one lane with 1-bit
EXEC/VCC, decodes vertex attributes like the fetch shader does, and reads every
scalar and buffer load from the captured memory. Texture samples cannot be
replayed: they return --tex. Captures taken with DELTA_GPU_RENDER_THREAD=0
read their tables at the same moment the draw was resolved; with the render
thread on, a ring the title rewrote in between shows up as a mismatch.
"""
import argparse
import glob
import json
import math
import os
import re
import struct
import sys



def f32(f):
    try:
        return struct.unpack('<f', struct.pack('<f', f))[0]
    except OverflowError:
        return math.copysign(math.inf, f)


def f2u(f):
    return struct.unpack('<I', struct.pack('<f', f32(f)))[0]


def u2f(u):
    return float(struct.unpack('<f', struct.pack('<I', u & 0xFFFFFFFF))[0])


def half(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


def to_half(f):
    try:
        return struct.unpack('<H', struct.pack('<e', f))[0]
    except OverflowError:
        return 0x7C00 | (0x8000 if f < 0 else 0)


class Memory:
    def __init__(self):
        self.ranges = []
        self.missing = set()
        self.last = None
        self.pc = -1

    def add(self, base, hexdata):
        if hexdata:
            self.ranges.append((int(base, 16), bytes.fromhex(hexdata)))

    def u32(self, addr):
        hit = self.last
        if hit and hit[0] <= addr and addr + 4 <= hit[0] + len(hit[1]):
            return struct.unpack_from('<I', hit[1], addr - hit[0])[0]
        for base, data in self.ranges:
            if base <= addr and addr + 4 <= base + len(data):
                self.last = (base, data)
                return struct.unpack_from('<I', data, addr - base)[0]
        self.missing.add((self.pc, addr & ~0xF))
        return 0


# Buffer data formats the fetch path decodes: dfmt -> (bits per component, n).
DFMT = {1: (8, 1), 2: (16, 1), 3: (8, 2), 4: (32, 1), 5: (16, 2), 7: (32, 2),
        10: (8, 4), 11: (32, 2), 12: (16, 4), 13: (32, 3), 14: (32, 4)}


def decode_attr(data, off, dfmt, nfmt):
    if dfmt in (8, 9):  # 10_10_10_2 / 2_10_10_10
        w = struct.unpack_from('<I', data, off)[0]
        widths = (10, 10, 10, 2)
        out, shift = [], 0
        for bits in widths:
            v = (w >> shift) & ((1 << bits) - 1)
            shift += bits
            if nfmt == 1:
                v = v - (1 << bits) if v >> (bits - 1) else v
                out.append(max(v / ((1 << (bits - 1)) - 1), -1.0))
            elif nfmt == 0:
                out.append(v / ((1 << bits) - 1))
            else:
                out.append(float(v))
        return out
    bits, n = DFMT[dfmt]
    out = []
    for c in range(n):
        if bits == 32:
            raw = struct.unpack_from('<I', data, off + 4 * c)[0]
            out.append(u2f(raw) if nfmt == 7 else float(raw if nfmt == 4 else
                       raw - (1 << 32) if raw >> 31 else raw))
        elif bits == 16:
            raw = struct.unpack_from('<H', data, off + 2 * c)[0]
            if nfmt == 7:
                out.append(half(raw))
            elif nfmt == 0:
                out.append(raw / 65535)
            elif nfmt == 1:
                s = raw - 65536 if raw >> 15 else raw
                out.append(max(s / 32767, -1.0))
            else:
                out.append(float(raw - 65536 if nfmt in (3, 5) and raw >> 15
                                 else raw))
        else:
            raw = data[off + c]
            if nfmt == 0:
                out.append(raw / 255)
            elif nfmt == 1:
                s = raw - 256 if raw >> 7 else raw
                out.append(max(s / 127, -1.0))
            else:
                out.append(float(raw - 256 if nfmt in (3, 5) and raw >> 7
                                 else raw))
    return out


def smrd_dwords(op):
    m = re.match(r's_(?:buffer_)?load_dword(?:x(\d+))?$', op)
    return int(m.group(1) or 1) if m else 0


def plan_bindings(prog):
    """pc -> cbuf binding and pc -> raw buffer binding, in the order the
    recompiler's PlanCbufs and PlanGfxBuffers hand them out."""
    loaded = {}  # sgpr -> (first, count, pc) of the SMRD that wrote it

    def version(sgpr, dwords):
        e = loaded.get(sgpr)
        return e[2] if e and sgpr + dwords <= e[0] + e[1] else -1

    cb_keys, cb_plan, buf_keys, buf_plan = {}, {}, {}, {}
    for pc, op, rest in prog:
        if op in ('s_setpc_b64', 's_endpgm'):
            break
        n = smrd_dwords(op)
        if n:
            dst = int(re.match(r's\[?(\d+)', rest).group(1))
            base = int(re.search(r', s\[(\d+):', rest).group(1))
            pointer = not op.startswith('s_buffer')
            key = (base, pointer, version(base, 2 if pointer else 4))
            cb_plan[pc] = cb_keys.setdefault(key, len(cb_keys))
            stale = {loaded[s][:2] for s in range(dst, dst + n) if s in loaded}
            for first, count in stale:
                for t in range(first, first + count):
                    loaded.pop(t, None)
            for s in range(dst, dst + n):
                loaded[s] = (dst, n, pc)
        elif op.startswith('buffer_load_dword') or op.startswith('tbuffer_load'):
            srsrc = int(re.search(r's\[(\d+):\d+\]', rest).group(1))
            key = (srsrc, version(srsrc, 4))
            buf_plan[pc] = buf_keys.setdefault(key, len(buf_keys))
    return cb_plan, buf_plan


INSN = re.compile(r'^([0-9a-f]{4}):\s+(?:[0-9a-f]{8}\s+)+(\S+)\s*(.*)$')


def parse_listing(path):
    prog = []
    for line in open(path):
        m = INSN.match(line)
        if m:
            prog.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return prog


SPECIAL = {'vcc_lo': 106, 'vcc_hi': 107, 'vcc': 106, 'm0': 124,
           'exec_lo': 126, 'exec_hi': 127, 'exec': 126}
INLINE_F = {'0.5': 0x3F000000, '-0.5': 0xBF000000, '1.0': 0x3F800000,
            '-1.0': 0xBF800000, '2.0': 0x40000000, '-2.0': 0xC0000000,
            '4.0': 0x40800000, '-4.0': 0xC0800000}


class Lane:
    def __init__(self, mem, tex):
        self.s = [0] * 128
        self.v = [0] * 256
        self.scc = 0
        self.mem = mem
        self.tex = tex
        self.pos = None
        self.notes = []
        self.cb_plan, self.buf_plan = {}, {}
        self.cbufs, self.bufs = [], []

    def reg(self, tok):
        m = re.match(r'([sv])\[(\d+):(\d+)\]$', tok)
        if m:
            return m.group(1), int(m.group(2)), int(m.group(3)) - int(m.group(2)) + 1
        m = re.match(r'([sv])(\d+)$', tok)
        if m:
            return m.group(1), int(m.group(2)), 1
        if tok in SPECIAL:
            return 's', SPECIAL[tok], 2 if tok in ('vcc', 'exec') else 1
        return None

    def read(self, tok):
        neg = absv = False
        if tok.startswith('-') and not re.match(r'^-\d', tok):
            neg, tok = True, tok[1:]
        if tok.startswith('|') and tok.endswith('|'):
            absv, tok = True, tok[1:-1]
        r = self.reg(tok)
        if r:
            u = (self.s if r[0] == 's' else self.v)[r[1]]
        elif tok == 'scc':
            u = self.scc
        elif tok in ('vccz', 'execz'):
            u = int(self.mask(106 if tok == 'vccz' else 126) == 0)
        elif tok in INLINE_F:
            u = INLINE_F[tok]
        elif tok.startswith('0x'):
            u = int(tok, 16) & 0xFFFFFFFF
        else:
            u = int(tok) & 0xFFFFFFFF
        if neg or absv:
            f = u2f(u)
            f = abs(f) if absv else f
            u = f2u(-f if neg else f)
        return u

    def f(self, tok):
        return u2f(self.read(tok))

    def write(self, tok, u):
        k, i, n = self.reg(tok)
        (self.s if k == 's' else self.v)[i] = u & 0xFFFFFFFF

    def mask(self, i):
        return (self.s[i] | self.s[i + 1] << 32) & 1

    def set_mask(self, tok, bit):
        k, i, n = self.reg(tok)
        self.s[i], self.s[i + 1] = bit & 1, 0

    def vdesc(self, base_sgpr):
        w0, w1, w2, w3 = self.s[base_sgpr:base_sgpr + 4]
        return (w0 | (w1 & 0xFFF) << 32, (w1 >> 16) & 0x3FFF, w2)


CMP = {'lt': lambda a, b: a < b, 'eq': lambda a, b: a == b,
       'le': lambda a, b: a <= b, 'gt': lambda a, b: a > b,
       'lg': lambda a, b: a != b, 'ne': lambda a, b: a != b,
       'ge': lambda a, b: a >= b, 'neq': lambda a, b: not (a == b),
       'nlt': lambda a, b: not (a < b), 'ngt': lambda a, b: not (a > b),
       'nle': lambda a, b: not (a <= b), 'nge': lambda a, b: not (a >= b),
       'o': lambda a, b: not (math.isnan(a) or math.isnan(b)),
       'u': lambda a, b: math.isnan(a) or math.isnan(b),
       'f': lambda a, b: False, 't': lambda a, b: True}


def as_int(u, signed):
    return u - (1 << 32) if signed and u >> 31 else u


def step(lane, op, rest, mem, pc=-1):
    mods = []
    for mod in (' clamp', ' mul:2', ' mul:4', ' div:2'):
        if mod in rest:
            mods.append(mod.strip())
            rest = rest.replace(mod, '')
    extra = {}
    for key in ('offset', 'dmask'):
        m = re.search(key + r':(0x[0-9a-f]+|\d+)', rest)
        if m:
            extra[key] = int(m.group(1), 0)
            rest = rest.replace(m.group(0), '')
    flags = set(re.findall(r'\b(idxen|offen|glc)\b', rest))
    rest = re.sub(r'\b(idxen|offen|glc)\b', '', rest)
    a = [x.strip() for x in rest.split(',')] if rest.strip() else []

    def out_f(v):
        if 'mul:2' in mods:
            v *= 2
        if 'mul:4' in mods:
            v *= 4
        if 'div:2' in mods:
            v /= 2
        if 'clamp' in mods:
            v = 0.0 if math.isnan(v) else min(max(v, 0.0), 1.0)
        return f2u(v)

    L = lane
    active = L.mask(126)
    vec = op.startswith('v_')
    if vec and not active and not op.startswith('v_cmp') and \
            op not in ('v_readlane_b32', 'v_readfirstlane_b32'):
        return
    fo = lambda: [L.f(x) for x in a[1:]]
    if op in ('s_waitcnt', 's_nop', 's_swappc_b64', 's_setprio',
              's_sendmsg', 's_dcache_inv'):
        return
    if op == 's_mov_b32':
        L.write(a[0], L.read(a[1]))
    elif op == 's_mov_b64':
        k, i, n = L.reg(a[0])
        r = L.reg(a[1])
        if r:
            L.s[i], L.s[i + 1] = L.s[r[1]], L.s[r[1] + 1]
        else:
            L.s[i], L.s[i + 1] = L.read(a[1]), 0
    elif op in ('s_and_b64', 's_or_b64', 's_andn2_b64', 's_xor_b64'):
        k, i, n = L.reg(a[0])
        x, y = (L.mask(L.reg(t)[1]) if L.reg(t) else L.read(t) & 1
                for t in a[1:3])
        r = {'s_and_b64': x & y, 's_or_b64': x | y, 's_andn2_b64': x & ~y & 1,
             's_xor_b64': x ^ y}[op]
        L.s[i], L.s[i + 1] = r, 0
        L.scc = int(r != 0)
    elif op in ('s_and_saveexec_b64', 's_or_saveexec_b64'):
        k, i, n = L.reg(a[0])
        src = L.mask(L.reg(a[1])[1])
        old = L.mask(126)
        L.s[i], L.s[i + 1] = old, 0
        new = src & old if op == 's_and_saveexec_b64' else src | old
        L.s[126], L.s[127] = new, 0
        L.scc = new
    elif op in ('s_add_u32', 's_add_i32', 's_sub_u32', 's_sub_i32',
                's_and_b32', 's_or_b32', 's_lshl_b32', 's_lshr_b32',
                's_mul_i32', 's_min_u32', 's_max_u32', 's_xor_b32'):
        x, y = L.read(a[1]), L.read(a[2])
        r = {'s_add_u32': x + y, 's_add_i32': x + y, 's_sub_u32': x - y,
             's_sub_i32': x - y, 's_and_b32': x & y, 's_or_b32': x | y,
             's_lshl_b32': x << (y & 31), 's_lshr_b32': x >> (y & 31),
             's_mul_i32': x * y, 's_min_u32': min(x, y),
             's_max_u32': max(x, y), 's_xor_b32': x ^ y}[op]
        L.write(a[0], r)
        L.scc = int((r & 0xFFFFFFFF) != 0) if 'add' not in op else int(r >> 32 != 0)
    elif op.startswith('s_load_dword'):
        k, i, n = L.reg(a[0])
        _, b, _ = L.reg(a[1])
        addr = L.s[b] | (L.s[b + 1] & 0xFFFF) << 32
        addr += int(a[2], 16) * 4 if a[2].startswith('0x') else L.read(a[2])
        for j in range(n):
            L.s[i + j] = mem.u32(addr + 4 * j)
    elif op.startswith('s_buffer_load_dword'):
        k, i, n = L.reg(a[0])
        _, b, _ = L.reg(a[1])
        base, _, _ = L.vdesc(b)
        if pc in L.cb_plan and L.cb_plan[pc] < len(L.cbufs) and L.cbufs[L.cb_plan[pc]]:
            base = L.cbufs[L.cb_plan[pc]][0]
        off = int(a[2], 16) * 4 if a[2].startswith('0x') else L.read(a[2])
        for j in range(n):
            L.s[i + j] = mem.u32(base + off + 4 * j)
    elif op.startswith('buffer_load_dword'):
        k, i, n = L.reg(a[0])
        _, vi, _ = L.reg(a[1])
        _, b, _ = L.reg(a[2])
        base, stride, _ = L.vdesc(b)
        if pc in L.buf_plan and L.buf_plan[pc] < len(L.bufs) and L.bufs[L.buf_plan[pc]]:
            base, stride = L.bufs[L.buf_plan[pc]]
        addr = base + extra.get('offset', 0)
        if 'idxen' in flags:
            addr += L.v[vi] * stride
        if 'offen' in flags:
            addr += L.v[vi + (1 if 'idxen' in flags else 0)]
        soff = L.read(a[3]) if len(a) > 3 else 0
        for j in range(n):
            L.v[i + j] = mem.u32(addr + soff + 4 * j)
    elif op.startswith('image_sample') or op.startswith('image_load'):
        k, i, n = L.reg(a[0])
        for j in range(bin(extra.get('dmask', 1)).count('1')):
            L.v[i + j] = f2u(L.tex)
        L.notes.append('texture sample returned --tex')
    elif op in ('v_mov_b32', 'v_readfirstlane_b32', 'v_readlane_b32'):
        L.write(a[0], L.read(a[1]))
    elif op in ('v_add_f32', 'v_sub_f32', 'v_subrev_f32', 'v_mul_f32',
                'v_max_f32', 'v_min_f32', 'v_mad_f32', 'v_madak_f32',
                'v_madmk_f32', 'v_mac_f32', 'v_fma_f32', 'v_mul_legacy_f32',
                'v_mac_legacy_f32', 'v_mad_legacy_f32', 'v_med3_f32',
                'v_min3_f32', 'v_max3_f32', 'v_min_legacy_f32',
                'v_max_legacy_f32'):
        x = fo()
        r = {'v_add_f32': lambda: x[0] + x[1], 'v_sub_f32': lambda: x[0] - x[1],
             'v_subrev_f32': lambda: x[1] - x[0],
             'v_mul_f32': lambda: x[0] * x[1],
             'v_mul_legacy_f32': lambda: 0.0 if 0.0 in x[:2] else x[0] * x[1],
             'v_max_f32': lambda: max(x[0], x[1]),
             'v_min_f32': lambda: min(x[0], x[1]),
             'v_mad_f32': lambda: x[0] * x[1] + x[2],
             'v_mad_legacy_f32': lambda: x[0] * x[1] + x[2],
             'v_fma_f32': lambda: x[0] * x[1] + x[2],
             'v_madak_f32': lambda: x[0] * x[1] + x[2],
             'v_madmk_f32': lambda: x[0] * x[2] + x[1],
             'v_mac_f32': lambda: x[0] * x[1] + L.f(a[0]),
             'v_mac_legacy_f32': lambda: x[0] * x[1] + L.f(a[0]),
             'v_med3_f32': lambda: sorted(x[:3])[1],
             'v_min3_f32': lambda: min(x[:3]), 'v_max3_f32': lambda: max(x[:3]),
             'v_min_legacy_f32': lambda: min(x[0], x[1]),
             'v_max_legacy_f32': lambda: max(x[0], x[1])}[op]()
        L.write(a[0], out_f(f32(r)))
    elif op in ('v_rcp_f32', 'v_rsq_f32', 'v_sqrt_f32', 'v_floor_f32',
                'v_fract_f32', 'v_ceil_f32', 'v_trunc_f32', 'v_sin_f32',
                'v_cos_f32', 'v_exp_f32', 'v_log_f32', 'v_rndne_f32'):
        x = L.f(a[1])
        def safe(fn):
            try:
                return fn()
            except (ValueError, OverflowError, ZeroDivisionError):
                return float('nan')
        r = safe({'v_rcp_f32': lambda: 1 / x if x else math.copysign(math.inf, x),
                  'v_rsq_f32': lambda: 1 / math.sqrt(x) if x else math.inf,
                  'v_sqrt_f32': lambda: math.sqrt(x),
                  'v_floor_f32': lambda: float(math.floor(x)),
                  'v_ceil_f32': lambda: float(math.ceil(x)),
                  'v_trunc_f32': lambda: float(math.trunc(x)),
                  'v_rndne_f32': lambda: float(round(x)),
                  'v_fract_f32': lambda: x - math.floor(x),
                  'v_sin_f32': lambda: math.sin(2 * math.pi * x),
                  'v_cos_f32': lambda: math.cos(2 * math.pi * x),
                  'v_exp_f32': lambda: 2.0 ** x,
                  'v_log_f32': lambda: math.log2(x) if x > 0 else -math.inf}[op])
        L.write(a[0], out_f(r))
    elif op in ('v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_lshlrev_b32',
                'v_lshrrev_b32', 'v_ashrrev_i32', 'v_add_i32', 'v_sub_i32',
                'v_subrev_i32', 'v_mul_lo_u32', 'v_mul_u32_u24', 'v_min_u32',
                'v_max_u32', 'v_min_i32', 'v_max_i32', 'v_bfe_u32',
                'v_bfi_b32', 'v_cvt_pkrtz_f16_f32', 'v_mad_u32_u24',
                'v_bfe_i32'):
        x = [L.read(t) for t in a[1:]]
        if op in ('v_add_i32', 'v_sub_i32', 'v_subrev_i32') and len(x) > 2:
            x = x[1:]  # vcc carry-out operand
        r = {'v_and_b32': lambda: x[0] & x[1], 'v_or_b32': lambda: x[0] | x[1],
             'v_xor_b32': lambda: x[0] ^ x[1],
             'v_lshlrev_b32': lambda: x[1] << (x[0] & 31),
             'v_lshrrev_b32': lambda: x[1] >> (x[0] & 31),
             'v_ashrrev_i32': lambda: as_int(x[1], True) >> (x[0] & 31),
             'v_add_i32': lambda: x[0] + x[1], 'v_sub_i32': lambda: x[0] - x[1],
             'v_subrev_i32': lambda: x[1] - x[0],
             'v_mul_lo_u32': lambda: x[0] * x[1],
             'v_mul_u32_u24': lambda: (x[0] & 0xFFFFFF) * (x[1] & 0xFFFFFF),
             'v_mad_u32_u24': lambda: (x[0] & 0xFFFFFF) * (x[1] & 0xFFFFFF) + x[2],
             'v_min_u32': lambda: min(x[0], x[1]), 'v_max_u32': lambda: max(x[0], x[1]),
             'v_min_i32': lambda: min(as_int(x[0], 1), as_int(x[1], 1)),
             'v_max_i32': lambda: max(as_int(x[0], 1), as_int(x[1], 1)),
             'v_bfe_u32': lambda: (x[0] >> (x[1] & 31)) & ((1 << (x[2] & 31)) - 1),
             'v_bfe_i32': lambda: as_int(((x[0] >> (x[1] & 31)) & ((1 << (x[2] & 31)) - 1)) << (32 - (x[2] & 31)) & 0xFFFFFFFF, True) >> (32 - (x[2] & 31)) if x[2] & 31 else 0,
             'v_bfi_b32': lambda: (x[0] & x[1]) | (~x[0] & x[2]),
             'v_cvt_pkrtz_f16_f32': lambda: to_half(u2f(x[0])) |
                 to_half(u2f(x[1])) << 16}[op]()
        L.write(a[0], r)
    elif op in ('v_cvt_f32_u32', 'v_cvt_f32_i32', 'v_cvt_u32_f32',
                'v_cvt_i32_f32', 'v_cvt_f32_ubyte0', 'v_cvt_f16_f32',
                'v_cvt_f32_f16'):
        x = L.read(a[1])
        if op == 'v_cvt_f32_u32':
            r = out_f(float(x))
        elif op == 'v_cvt_f32_i32':
            r = out_f(float(as_int(x, True)))
        elif op == 'v_cvt_f32_ubyte0':
            r = out_f(float(x & 0xFF))
        elif op == 'v_cvt_f32_f16':
            r = out_f(half(x & 0xFFFF))
        elif op == 'v_cvt_f16_f32':
            r = to_half(u2f(x))
        else:
            fx = u2f(x)
            fx = 0.0 if math.isnan(fx) else fx
            r = max(0, min(int(fx), 0xFFFFFFFF)) if op == 'v_cvt_u32_f32' else \
                max(-2**31, min(int(fx), 2**31 - 1))
        L.write(a[0], r)
    elif op == 'v_cndmask_b32':
        sel = L.mask(L.reg(a[3])[1]) if len(a) > 3 else L.mask(106)
        L.write(a[0], L.read(a[2]) if sel else L.read(a[1]))
    elif op.startswith('v_cmp'):
        m = re.match(r'v_cmpx?_(\w+?)_([fiu])32$', op)
        kind, ty = m.group(1), m.group(2)
        if ty == 'f':
            x, y = L.f(a[1]), L.f(a[2])
        else:
            x, y = as_int(L.read(a[1]), ty == 'i'), as_int(L.read(a[2]), ty == 'i')
        bit = int(bool(CMP[kind](x, y))) & active
        L.set_mask(a[0], bit)
        if op.startswith('v_cmpx'):
            L.s[126], L.s[127] = bit, 0
    elif op == 'exp':
        if a and a[0].startswith('pos0'):
            regs = [a[0].split()[1]] + [t.split()[0] for t in a[1:4]]
            L.pos = [L.f(r) for r in regs]
    else:
        raise NotImplementedError(op + ' ' + rest)


def run(prog, lane, mem, stop_pc, probe_regs):
    index = {pc: i for i, (pc, _, _) in enumerate(prog)}
    i, steps, probed = 0, 0, None
    while i < len(prog) and steps < 100000:
        pc, op, rest = prog[i]
        steps += 1
        if op == 's_endpgm':
            break
        target = re.search(r'-> ([0-9a-f]{4})', rest)
        if op.startswith('s_cbranch') or op == 's_branch':
            cond = {'s_branch': True,
                    's_cbranch_scc0': lane.scc == 0, 's_cbranch_scc1': lane.scc == 1,
                    's_cbranch_vccz': lane.mask(106) == 0,
                    's_cbranch_vccnz': lane.mask(106) != 0,
                    's_cbranch_execz': lane.mask(126) == 0,
                    's_cbranch_execnz': lane.mask(126) != 0}[op]
            i = index[int(target.group(1), 16)] if cond else i + 1
            continue
        mem.pc = pc
        step(lane, op, rest, mem, pc)
        if pc == stop_pc and probed is None:
            probed = probe_regs()
        i += 1
    return probed


def fetch_into(lane, fetch_prog, draw, mem):
    """Run the fetch shader's loads against the draw's resolved attributes."""
    vbufs = {v['i']: (int(v['base'], 16), v['stride'], bytes.fromhex(v.get('data', '')))
             for v in draw['vbufs']}
    attrs = sorted(draw['vattrs'], key=lambda x: x['location'])
    k = 0
    for pc, op, rest in fetch_prog:
        if op in ('s_setpc_b64', 's_endpgm'):
            break
        m = re.match(r'buffer_load_format_(x|xy|xyz|xyzw)$', op) or \
            re.match(r'tbuffer_load_format_(x|xy|xyz|xyzw)$', op)
        if not m:
            continue
        n = len(m.group(1))
        dst = int(re.match(r'v\[?(\d+)', rest).group(1))
        idx = int(re.search(r', v(\d+),', rest).group(1))
        a = attrs[k]
        k += 1
        base, stride, data = vbufs[a['binding']]
        off = lane.v[idx] * stride + a['offset']
        if off + 4 > len(data):
            lane.notes.append(f'attribute {a["location"]} outside captured data')
            vals = [0.0] * 4
        else:
            vals = decode_attr(data, off, a['dfmt'], a['nfmt'])
        vals = (vals + [0.0, 0.0, 0.0, 1.0])[:4] if len(vals) < 4 else vals
        if len(vals) < 4:
            vals += [0.0, 0.0, 0.0, 1.0][len(vals):]
        for c in range(n):
            v = vals[c] if c < len(vals) else (1.0 if c == 3 else 0.0)
            int_fmt = a['nfmt'] in (4, 5) and a['dfmt'] not in (8, 9)
            lane.v[dst + c] = (int(v) & 0xFFFFFFFF) if int_fmt else f2u(v)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cap', help='capture directory or frame jsonl')
    p.add_argument('--pc', type=lambda x: int(x, 0), required=True)
    p.add_argument('--reg', type=int, default=0,
                   help='first probed register: v<n>, or s<n-1000>; '
                        'unused with --pc 0xffff (the exported position)')
    p.add_argument('--vs', type=lambda x: int(x, 0), help='probed VS address')
    p.add_argument('--tex', type=float, default=0.0)
    p.add_argument('--limit', type=int, default=12, help='mismatches to print')
    p.add_argument('--tol', type=float, default=1e-3)
    p.add_argument('--outliers', type=float,
                   help='with --pc 0xffff: only slots whose GPU position is '
                        'beyond this NDC magnitude (or behind the camera)')
    p.add_argument('--rt-width', type=int,
                   help='only draws into a target this wide')
    p.add_argument('--max', type=int, default=400,
                   help='invocations to replay, sampled evenly')
    a = p.parse_args()

    jsonl = a.cap if a.cap.endswith('.jsonl') else \
        sorted(glob.glob(os.path.join(a.cap, 'frame_*.jsonl')))[-1]
    prefix = jsonl[:-len('.jsonl')]
    probe_path = prefix + '_vsprobe.txt'
    if not os.path.exists(probe_path):
        sys.exit(f'no probe dump {probe_path}: capture with DELTA_GPU_GCN_VSVGPR_VS')
    probe = {}
    for line in open(probe_path):
        inst, vtx, t0, t1, *regs = line.split()
        probe.setdefault((int(t0, 16), int(t1, 16)), {})[(int(inst), int(vtx))] = \
            [int(r, 16) for r in regs]
    draws = [json.loads(l) for l in open(jsonl)]
    draws = [d for d in draws if d.get('t') == 'draw' and
             (a.vs is None or int(d['vs']['addr'], 16) == a.vs) and
             any(v.get('data') for v in d['vbufs'])]
    if not draws:
        sys.exit('no draws of the probed shader with captured buffers')
    vs_addr = int(draws[0]['vs']['addr'], 16)
    vs_prog = parse_listing(f'{prefix}_sh_{vs_addr:x}.s')
    cb_plan, buf_plan = plan_bindings(vs_prog)
    probe_regs = (lambda L: [L.s[a.reg - 1000 + j] for j in range(4)]) \
        if a.reg >= 1000 else (lambda L: [L.v[a.reg + j] for j in range(4)])
    total = bad = 0
    ndc = []
    shown = 0
    def prepare(d):
        ud = [int(x, 16) for x in d['vs_user_data']]
        mem = Memory()
        for c in d['cbufs']:
            mem.add(c['base'], c.get('data', ''))
        for b in d['bufs']:
            mem.add(b['base'], b.get('data', ''))
        cbufs = [None] * 64
        for c in d['cbufs']:
            cbufs[c['i']] = (int(c['base'], 16),)
        bufs = [None] * 16
        for b in d['bufs']:
            bufs[b['i']] = (int(b['base'], 16), b.get('stride', 0))
        fetch_addr = ud[0] | (ud[1] & 0xFFFF) << 32
        fetch_prog = parse_listing(f'{prefix}_sh_{fetch_addr:x}.s')
        return d, ud, mem, cbufs, bufs, fetch_prog

    def replay(ctx, inst, vtx):
        d, ud, mem, cbufs, bufs, fetch_prog = ctx
        L = Lane(mem, a.tex)
        L.cb_plan, L.buf_plan = cb_plan, buf_plan
        L.cbufs, L.bufs = cbufs, bufs
        for i, x in enumerate(ud):
            L.s[i] = x
        L.s[126] = 1
        L.v[0], L.v[1], L.v[3] = vtx, inst, inst
        fetch_into(L, fetch_prog, d, mem)
        cpu = run(vs_prog, L, mem, a.pc, lambda: probe_regs(L))
        if a.pc == 0xFFFF and L.pos:
            cpu = [f2u(x) for x in L.pos]
        return cpu, L

    def differs(gpu, cpu):
        return any(abs(u2f(g) - u2f(c)) > a.tol * max(1.0, abs(u2f(c))) and
                   g != c for g, c in zip(gpu, cpu))

    # Draws that share tags overwrote each other's slots: a slot matches when
    # any of them reproduces it.
    groups = {}
    for d in draws:
        ud = [int(x, 16) for x in d['vs_user_data']]
        groups.setdefault((ud[2], ud[12]), []).append(d)
    # --rt-width picks which slots to check; every draw sharing their tags
    # stays a candidate, whatever pass it belongs to.
    work = [(tags, key, gpu) for tags, ds in groups.items()
            if a.rt_width is None or any(d['rt_w'] == a.rt_width for d in ds)
            for key, gpu in sorted(probe.get(tags, {}).items())]
    if a.outliers is not None:
        def far(gpu):
            x, y, z, w = (u2f(v) for v in gpu)
            return w <= 0 or abs(x / w) > a.outliers or abs(y / w) > a.outliers
        work = [item for item in work if far(item[2])]
        print(f'{len(work)} slots with the GPU position beyond {a.outliers} NDC')
    missing_mem = set()
    for tags, (inst, vtx), gpu in work[::max(1, len(work) // a.max)]:
        best = None
        for d in groups[tags]:
            cpu, L = replay(prepare(d), inst, vtx)
            missing_mem |= L.mem.missing
            if cpu is None:
                continue
            if L.pos and L.pos[3]:
                ndc.append([c / L.pos[3] for c in L.pos[:3]] + [L.pos[3]])
            if not differs(gpu, cpu):
                best = None
                break
            best = best or (d, cpu, L)
        else:
            if best is None:
                continue
        total += 1
        if best is None:
            continue
        bad += 1
        if shown < a.limit:
            shown += 1
            d, cpu, L = best
            fmt = lambda xs: ' '.join(f'{u2f(x):.5g}' for x in xs)
            widths = sorted({g['rt_w'] for g in groups[tags]})
            print(f'draw {d["draw"]} (of {len(groups[tags])} sharing its tags, '
                  f'targets {widths} wide) '
                  f'inst {inst} vtx {vtx}:\n  gpu {fmt(gpu)}\n  cpu {fmt(cpu)}'
                  + (f'\n  note: {L.notes[0]}' if L.notes else ''))
    mem = Memory()
    mem.missing = missing_mem
    if mem.missing:
        by_pc = {}
        for pc, addr in mem.missing:
            by_pc.setdefault(pc, set()).add(addr)
        print('reads outside captured memory: ' + ', '.join(
            f'pc {pc:#x} at {min(addrs):#x} ({len(addrs)} lines)'
            for pc, addrs in sorted(by_pc.items())[:8]))
    print(f'{total} invocations compared, {bad} differ')
    if ndc:
        inside = sum(1 for x, y, z, w in ndc if abs(x) <= 1.5 and abs(y) <= 1.5)
        behind = sum(1 for *_, w in ndc if w <= 0)
        print(f'cpu positions: {inside}/{len(ndc)} within 1.5 NDC, '
              f'{behind} behind the camera, |x| up to '
              f'{max(abs(x) for x, *_ in ndc):.3g}')


if __name__ == '__main__':
    main()
