#!/usr/bin/env python3
"""Replay one captured GCN compute dispatch on the CPU and diff it against
what our recompiled shader wrote on the GPU.

The interpreter runs a real 64-lane wave with 64-bit EXEC/VCC masks, so it is
the ground truth for shaders that lean on wave-wide behaviour. It is a
debugging oracle only: nothing in the emulator uses it.

Capture a dispatch first (the Nth dispatch of one shader):

  tools/drun.py uc2 -t 40 -o DELTA_GPU_CAPTURE_CS=0xe380089300 \\
      -o DELTA_GPU_CAPTURE_CS_SKIP=187 -o DELTA_GPU_CAPTURE_RAW=1 \\
      -o DELTA_GPU_CAPTURE_DIR=/tmp/cap -o DELTA_GPU_CAPTURE_EXIT=1 \\
      -o DELTA_GPU_CAPTURE_DUMP=none

then replay it with the launch parameters DELTA_GPU_CSDUMP prints:

  tools/gcn_interp.py /tmp/cap/frame_65.jsonl --tg 4,16,1 --user-sgpr 7 \\
      --tgid 1 --lds 4

Inputs are the "bound" snapshots (what the GPU shader actually read), so a
difference in the outputs is a translation bug, not stale data. Only the
opcodes the replayed shaders needed so far are implemented; anything else
stops with the instruction that is missing.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

LANES = 64
U32 = np.uint32
MASK64 = (1 << 64) - 1
LANE_BITS = np.array([1 << i for i in range(LANES)], dtype=object)


class Unsupported(Exception):
    pass


def f2u(x):
    return np.asarray(x, np.float32).view(U32)


def u2f(x):
    return np.asarray(x, U32).view(np.float32)


def sext(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def to_mask(bools):
    m = 0
    for i in np.nonzero(bools)[0]:
        m |= 1 << int(i)
    return m


def lanes_of(mask):
    return ((mask >> np.arange(LANES, dtype=object)) & 1).astype(bool)


# --- guest memory ------------------------------------------------------------
class Memory:
    def __init__(self):
        self.regions = []  # (base, bytearray)
        self.misses = {}
        self.watch = None
        self.on_watch = None

    def add(self, base, data):
        end = base + len(data)
        for i, (b, d) in enumerate(self.regions):
            e = b + len(d)
            if base <= e and b <= end:
                lo, hi = min(b, base), max(e, end)
                merged = bytearray(hi - lo)
                merged[b - lo:e - lo] = d
                merged[base - lo:end - lo] = data
                self.regions.pop(i)
                self.add(lo, merged)
                return
        self.regions.append((base, bytearray(data)))

    def find(self, addr, n):
        for b, d in self.regions:
            if b <= addr and addr + n <= b + len(d):
                return d, addr - b
        return None, 0

    def read(self, addr, n):
        d, o = self.find(addr, n)
        if d is None:
            self.misses[addr & ~0xfff] = self.misses.get(addr & ~0xfff, 0) + 1
            return bytes(n)
        return bytes(d[o:o + n])

    def write(self, addr, data):
        if self.watch is not None and addr <= self.watch < addr + len(data):
            self.on_watch(addr, data)
        d, o = self.find(addr, len(data))
        if d is None:
            self.misses[addr & ~0xfff] = self.misses.get(addr & ~0xfff, 0) + 1
            return
        d[o:o + len(data)] = data

    def u32(self, addr):
        return struct.unpack("<I", self.read(addr, 4))[0]


# --- buffer formats ------------------------------------------------------------
# dfmt -> (bits per component, components)
DFMT = {1: (8, 1), 2: (16, 1), 3: (8, 2), 4: (32, 1), 5: (16, 2), 10: (8, 4),
        11: (32, 2), 12: (16, 4), 13: (32, 3), 14: (32, 4)}


def decode_component(raw, bits, nfmt):
    if bits == 32:
        return raw & 0xffffffff
    if nfmt == 0:
        return int(f2u(np.float32(raw / float((1 << bits) - 1))))
    if nfmt == 1:
        v = sext(raw, bits) / float((1 << (bits - 1)) - 1)
        return int(f2u(np.float32(max(v, -1.0))))
    if nfmt == 2:
        return int(f2u(np.float32(raw)))
    if nfmt == 3:
        return int(f2u(np.float32(sext(raw, bits))))
    if nfmt == 4:
        return raw
    if nfmt == 5:
        return sext(raw, bits) & 0xffffffff
    if nfmt == 7 and bits == 16:
        return int(f2u(np.frombuffer(struct.pack("<H", raw), np.float16)[0]
                       .astype(np.float32)))
    raise Unsupported(f"buffer format bits={bits} nfmt={nfmt}")


def encode_component(value, bits, nfmt):
    if bits == 32:
        return value & 0xffffffff
    if nfmt in (4, 5):
        return value & ((1 << bits) - 1)
    f = float(u2f(np.uint32(value)))
    if nfmt == 0:
        return int(round(min(max(f, 0.0), 1.0) * ((1 << bits) - 1)))
    if nfmt == 7 and bits == 16:
        return struct.unpack("<H", np.float16(f).tobytes())[0]
    raise Unsupported(f"store format bits={bits} nfmt={nfmt}")


def apply_dst_sel(values, word3, n):
    one = 1
    out = []
    for c in range(n):
        sel = (word3 >> (3 * c)) & 7
        if sel >= 4:
            out.append(values[sel - 4] if sel - 4 < len(values) else 0)
        elif sel == 1:
            out.append(one)
        else:
            out.append(0)
    return out


# --- image layout ------------------------------------------------------------
def linear_pitch(pitch, width, height, tiling):
    pitch = max(pitch, width)
    if tiling == 31:
        return pitch
    if tiling != 8:
        raise Unsupported(f"tiled image (tiling {tiling})")
    pitch = (pitch + 15) & ~15
    while (pitch * height) % 64:
        pitch += 16
    return pitch


class TImage:
    def __init__(self, w):
        self.base = ((w[0] | ((w[1] & 0xff) << 32)) << 8)
        self.dfmt = (w[1] >> 20) & 0x3f
        self.nfmt = (w[1] >> 26) & 0xf
        self.width = (w[2] & 0x3fff) + 1
        self.height = ((w[2] >> 14) & 0x3fff) + 1
        self.dst_sel = w[3] & 0xfff
        self.base_mip = (w[3] >> 12) & 0xf
        self.last_mip = (w[3] >> 16) & 0xf
        self.tiling = (w[3] >> 20) & 0x1f
        self.pow2 = (w[3] >> 25) & 1
        self.type = (w[3] >> 28) & 0xf
        self.depth = (w[4] & 0x1fff) + 1
        self.pitch = ((w[4] >> 13) & 0x3fff) + 1
        self.base_array = w[5] & 0x1fff
        if self.base_mip or self.last_mip:
            raise Unsupported("mipmapped image")
        if self.pow2:
            raise Unsupported("pow2-padded image")
        if self.dfmt not in DFMT:
            raise Unsupported(f"image dfmt {self.dfmt}")
        self.bits, self.comps = DFMT[self.dfmt]
        self.bpp = self.bits * self.comps // 8
        self.row = linear_pitch(self.pitch, self.width, self.height,
                                self.tiling)

    def address(self, x, y, layer):
        return self.base + ((layer + self.base_array) * self.row * self.height +
                            y * self.row + x) * self.bpp


# --- the wave ------------------------------------------------------------------
class Wave:
    def __init__(self, mem, code, code_base, lds_bytes):
        self.mem = mem
        self.code = code
        self.code_base = code_base
        self.s = [0] * 128
        self.scc = 0
        self.v = np.zeros((256, LANES), U32)
        self.lds = bytearray(max(lds_bytes, 4))
        self.executed = 0
        self.trace = None
        self.probe_pc = None
        self.lds_watch = None
        self.lane = None
        self.probe_reg = 0
        self.probe = np.zeros((4, LANES), U32)

    # exec / vcc as 64-bit ints in s126/s127, s106/s107
    def get64(self, r):
        return self.s[r] | (self.s[r + 1] << 32)

    def set64(self, r, v):
        self.s[r] = v & 0xffffffff
        self.s[r + 1] = (v >> 32) & 0xffffffff

    @property
    def exec(self):
        return self.get64(126)

    def active(self):
        return lanes_of(self.exec)

    # scalar operands
    def ssrc(self, f, lit, wide=False):
        if f <= 103 or f in (106, 107, 124, 126, 127):
            return self.get64(f) if wide else self.s[f]
        if 128 <= f <= 192:
            return f - 128
        if 193 <= f <= 208:
            v = -(f - 192)
            return v & (MASK64 if wide else 0xffffffff)
        if 240 <= f <= 247:
            c = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][f - 240]
            return int(f2u(np.float32(c)))
        if f == 251:
            return int(self.get64(106) == 0)
        if f == 252:
            return int(self.exec == 0)
        if f == 253:
            return self.scc
        if f == 255:
            return lit
        raise Unsupported(f"scalar operand {f}")

    def sdst(self, d, v, wide=False):
        if wide:
            self.set64(d, v)
        else:
            self.s[d] = v & 0xffffffff

    # vector operands (9-bit)
    def vsrc(self, f, lit):
        if f >= 256:
            return self.v[f - 256].copy()
        return np.full(LANES, self.ssrc(f, lit) & 0xffffffff, U32)

    def vwrite(self, d, val, mask=None):
        m = self.active() if mask is None else mask
        self.v[d][m] = np.asarray(val, U32)[m] if np.ndim(val) else U32(val)

    # --- execution --------------------------------------------------------
    def run(self, pc=0, limit=200_000_000):
        while True:
            if self.executed > limit:
                raise Unsupported("instruction limit")
            w0 = self.code[pc]
            self.pc = pc
            active = self.active() if pc == self.probe_pc else None
            npc = self.step(pc, w0)
            if active is not None:
                for i in range(4):
                    r = self.probe_reg + i
                    val = np.full(LANES, self.s[r - 1000], U32) \
                        if r >= 1000 else self.v[r]
                    self.probe[i][active] = val[active]
            self.executed += 1
            if npc is None:
                return
            pc = npc

    def literal(self, pc, *fields):
        return self.code[pc + 1] if 255 in fields else 0

    def step(self, pc, w):
        if self.trace is not None:
            self.trace(self, pc)
        top2 = w >> 30
        if w >> 23 == 0x17d:
            return self.sop1(pc, w)
        if w >> 23 == 0x17e:
            return self.sopc(pc, w)
        if w >> 23 == 0x17f:
            return self.sopp(pc, w)
        if w >> 28 == 0xb:
            return self.sopk(pc, w)
        if top2 == 2:
            return self.sop2(pc, w)
        if w >> 27 == 0x18:
            return self.smrd(pc, w)
        if w >> 25 == 0x3f:
            return self.vop1(pc, w)
        if w >> 25 == 0x3e:
            return self.vopc(pc, w)
        if w >> 31 == 0:
            return self.vop2(pc, w)
        if w >> 26 == 0x34:
            return self.vop3(pc, w, self.code[pc + 1])
        if w >> 26 == 0x36:
            return self.ds(pc, w, self.code[pc + 1])
        if w >> 26 == 0x38:
            return self.mubuf(pc, w, self.code[pc + 1], None)
        if w >> 26 == 0x3a:
            return self.mubuf(pc, w, self.code[pc + 1], w)
        if w >> 26 == 0x3c:
            return self.mimg(pc, w, self.code[pc + 1])
        raise Unsupported(f"encoding {w:08x} at pc {pc:#x}")

    # --- scalar ALU -------------------------------------------------------
    def sop2(self, pc, w):
        op = (w >> 23) & 0x7f
        d, s1f, s0f = (w >> 16) & 0x7f, (w >> 8) & 0xff, w & 0xff
        lit = self.literal(pc, s0f, s1f)
        n = pc + (2 if 255 in (s0f, s1f) else 1)
        wide = op in (0x0b, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1b, 0x1d,
                      0x1f, 0x21, 0x23, 0x25, 0x29, 0x2a)
        a = self.ssrc(s0f, lit, wide and op not in (0x25,))
        b = self.ssrc(s1f, lit, wide and op not in (0x1f, 0x21, 0x23, 0x25,
                                                    0x29, 0x2a))
        M = MASK64 if wide else 0xffffffff
        si = lambda x: sext(x, 32)
        if op == 0x00:
            r = a + b; self.scc = int(r >> 32); r &= M
        elif op == 0x01:
            r = (a - b) & M; self.scc = int(b > a)
        elif op == 0x02:
            r = si(a) + si(b); self.scc = int(r != sext(r & M, 32)); r &= M
        elif op == 0x03:
            r = si(a) - si(b); self.scc = int(r != sext(r & M, 32)); r &= M
        elif op == 0x04:
            r = a + b + self.scc; self.scc = int(r >> 32); r &= M
        elif op == 0x05:
            r = (a - b - self.scc); self.scc = int(r < 0); r &= M
        elif op == 0x06:
            self.scc = int(si(a) < si(b)); r = a if self.scc else b
        elif op == 0x07:
            self.scc = int(a < b); r = a if self.scc else b
        elif op == 0x08:
            self.scc = int(si(a) > si(b)); r = a if self.scc else b
        elif op == 0x09:
            self.scc = int(a > b); r = a if self.scc else b
        elif op in (0x0a, 0x0b):
            r = a if self.scc else b
            self.sdst(d, r, wide)
            return n
        elif op in (0x0e, 0x0f):
            r = a & b; self.scc = int(r != 0)
        elif op in (0x10, 0x11):
            r = a | b; self.scc = int(r != 0)
        elif op in (0x12, 0x13):
            r = a ^ b; self.scc = int(r != 0)
        elif op in (0x14, 0x15):
            r = a & ~b & M; self.scc = int(r != 0)
        elif op in (0x16, 0x17):
            r = (a | ~b) & M; self.scc = int(r != 0)
        elif op in (0x18, 0x19):
            r = ~(a & b) & M; self.scc = int(r != 0)
        elif op in (0x1a, 0x1b):
            r = ~(a | b) & M; self.scc = int(r != 0)
        elif op in (0x1c, 0x1d):
            r = ~(a ^ b) & M; self.scc = int(r != 0)
        elif op == 0x1e:
            r = (a << (b & 31)) & M; self.scc = int(r != 0)
        elif op == 0x1f:
            r = (a << (b & 63)) & M; self.scc = int(r != 0)
        elif op == 0x20:
            r = a >> (b & 31); self.scc = int(r != 0)
        elif op == 0x21:
            r = a >> (b & 63); self.scc = int(r != 0)
        elif op == 0x22:
            r = (si(a) >> (b & 31)) & M; self.scc = int(r != 0)
        elif op == 0x24:
            r = (((1 << (a & 31)) - 1) << (b & 31)) & M
            self.sdst(d, r)
            return n
        elif op == 0x26:
            r = (si(a) * si(b)) & M
            self.sdst(d, r)
            return n
        elif op == 0x27:
            off, wid = b & 31, (b >> 16) & 0x7f
            r = (a >> off) & ((1 << wid) - 1) if wid else 0
            self.scc = int(r != 0)
        elif op == 0x28:
            off, wid = b & 31, (b >> 16) & 0x7f
            r = sext((a >> off) & ((1 << wid) - 1), wid) & M if wid else 0
            self.scc = int(r != 0)
        else:
            raise Unsupported(f"sop2 {op:#x} at pc {pc:#x}")
        self.sdst(d, r, wide)
        return n

    def sopk(self, pc, w):
        op = (w >> 23) & 0x1f
        d, imm = (w >> 16) & 0x7f, w & 0xffff
        simm = sext(imm, 16)
        if op == 0x00:
            self.s[d] = simm & 0xffffffff
        elif op == 0x02:
            if self.scc:
                self.s[d] = simm & 0xffffffff
        elif 0x03 <= op <= 0x08:
            a = sext(self.s[d], 32)
            self.scc = int([a == simm, a != simm, a > simm, a >= simm,
                            a < simm, a <= simm][op - 3])
        elif 0x09 <= op <= 0x0e:
            a = self.s[d]
            self.scc = int([a == imm, a != imm, a > imm, a >= imm,
                            a < imm, a <= imm][op - 9])
        elif op == 0x0f:
            r = sext(self.s[d], 32) + simm
            self.scc = int(r != sext(r & 0xffffffff, 32))
            self.s[d] = r & 0xffffffff
        elif op == 0x10:
            self.s[d] = (sext(self.s[d], 32) * simm) & 0xffffffff
        else:
            raise Unsupported(f"sopk {op:#x} at pc {pc:#x}")
        return pc + 1

    def sop1(self, pc, w):
        op = (w >> 8) & 0xff
        d, s0f = (w >> 16) & 0x7f, w & 0xff
        lit = self.literal(pc, s0f)
        n = pc + (2 if s0f == 255 else 1)
        if op == 0x03:
            self.s[d] = self.ssrc(s0f, lit)
        elif op == 0x04:
            self.set64(d, self.ssrc(s0f, lit, True))
        elif op == 0x07:
            r = ~self.ssrc(s0f, lit) & 0xffffffff
            self.scc = int(r != 0); self.s[d] = r
        elif op == 0x08:
            r = ~self.ssrc(s0f, lit, True) & MASK64
            self.scc = int(r != 0); self.set64(d, r)
        elif op == 0x0f:
            r = bin(self.ssrc(s0f, lit)).count("1")
            self.scc = int(r != 0); self.s[d] = r
        elif op == 0x10:
            r = bin(self.ssrc(s0f, lit, True)).count("1")
            self.scc = int(r != 0); self.s[d] = r
        elif op in (0x13, 0x14):
            v = self.ssrc(s0f, lit, op == 0x14)
            self.s[d] = ((v & -v).bit_length() - 1) & 0xffffffff if v else \
                0xffffffff
        elif op == 0x1f:
            self.set64(d, self.code_base + (pc + 1) * 4)
        elif 0x24 <= op <= 0x2b:
            src = self.ssrc(s0f, lit, True)
            ex = self.exec
            self.set64(d, ex)
            f = {0x24: lambda: src & ex, 0x25: lambda: src | ex,
                 0x26: lambda: src ^ ex, 0x27: lambda: src & ~ex,
                 0x28: lambda: src | ~ex, 0x29: lambda: ~(src & ex),
                 0x2a: lambda: ~(src | ex), 0x2b: lambda: ~(src ^ ex)}[op]
            ne = f() & MASK64
            self.set64(126, ne)
            self.scc = int(ne != 0)
        else:
            raise Unsupported(f"sop1 {op:#x} at pc {pc:#x}")
        return n

    def sopc(self, pc, w):
        op = (w >> 16) & 0x7f
        s1f, s0f = (w >> 8) & 0xff, w & 0xff
        lit = self.literal(pc, s0f, s1f)
        n = pc + (2 if 255 in (s0f, s1f) else 1)
        a, b = self.ssrc(s0f, lit), self.ssrc(s1f, lit)
        if op <= 5:
            a, b = sext(a, 32), sext(b, 32)
        k = op % 6 if op <= 0x0b else None
        if k is not None:
            self.scc = int([a == b, a != b, a > b, a >= b, a < b, a <= b][k])
        elif op == 0x0c:
            self.scc = int(((a >> (b & 31)) & 1) == 0)
        elif op == 0x0d:
            self.scc = int(((a >> (b & 31)) & 1) == 1)
        else:
            raise Unsupported(f"sopc {op:#x} at pc {pc:#x}")
        return n

    def sopp(self, pc, w):
        op = (w >> 16) & 0x7f
        simm = sext(w & 0xffff, 16)
        target = pc + 1 + simm
        if op == 0x01:
            return None
        if op in (0x00, 0x0a, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x13,
                  0x14):
            return pc + 1
        cond = {0x02: True, 0x04: self.scc == 0, 0x05: self.scc == 1,
                0x06: self.get64(106) == 0, 0x07: self.get64(106) != 0,
                0x08: self.exec == 0, 0x09: self.exec != 0}
        if op not in cond:
            raise Unsupported(f"sopp {op:#x} at pc {pc:#x}")
        return target if cond[op] else pc + 1

    def smrd(self, pc, w):
        op = (w >> 22) & 0x1f
        d, sbase = (w >> 15) & 0x7f, ((w >> 9) & 0x3f) * 2
        imm, off = (w >> 8) & 1, w & 0xff
        n = pc + 1
        if imm:
            byte_off = off * 4
        elif off == 255:
            byte_off = self.code[pc + 1]; n = pc + 2
        else:
            byte_off = self.s[off]
        counts = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 8: 1, 9: 2, 10: 4, 11: 8,
                  12: 16}
        if op not in counts:
            raise Unsupported(f"smrd {op:#x} at pc {pc:#x}")
        if op < 8:
            base = self.s[sbase] | ((self.s[sbase + 1] & 0xffff) << 32)
            limit = None
        else:
            base = self.s[sbase] | ((self.s[sbase + 1] & 0xffff) << 32)
            limit = self.s[sbase + 2]
        for i in range(counts[op]):
            o = byte_off + i * 4
            self.s[d + i] = 0 if limit is not None and o + 4 > limit else \
                self.mem.u32(base + o)
        return n

    # --- vector ALU ---------------------------------------------------------
    def vop1(self, pc, w):
        op = (w >> 9) & 0xff
        d, s0f = (w >> 17) & 0xff, w & 0x1ff
        lit = self.literal(pc, s0f)
        n = pc + (2 if s0f == 255 else 1)
        if op == 0x02:
            a = self.vsrc(s0f, lit)
            act = np.nonzero(self.active())[0]
            self.s[d] = int(a[act[0]] if act.size else a[0])
            return n
        self.valu(0x180 + op, d, [self.vsrc(s0f, lit)], pc)
        return n

    def vop2(self, pc, w):
        op = (w >> 25) & 0x3f
        d, v1, s0f = (w >> 17) & 0xff, (w >> 9) & 0xff, w & 0x1ff
        if op in (0x20, 0x21):
            k = self.code[pc + 1]
            a, b = self.vsrc(s0f, 0), self.v[v1].copy()
            fa, fb, fk = u2f(a), u2f(b), np.float32(u2f(np.uint32(k)))
            r = fa * fk + fb if op == 0x20 else fa * fb + fk
            self.vwrite(d, f2u(r))
            return pc + 2
        lit = self.literal(pc, s0f)
        n = pc + (2 if s0f == 255 else 1)
        a, b = self.vsrc(s0f, lit), self.v[v1].copy()
        if op == 0x00:
            self.vwrite(d, np.where(lanes_of(self.get64(106)), b, a))
            return n
        if op == 0x01:
            self.s[d] = int(a[self.ssrc(v1, 0) & 63])
            return n
        if op == 0x02:
            self.v[d][self.ssrc(v1, 0) & 63] = self.ssrc(s0f, lit)
            return n
        self.valu(0x100 + op, d, [a, b], pc)
        return n

    def vopc(self, pc, w):
        op = (w >> 17) & 0xff
        v1, s0f = (w >> 9) & 0xff, w & 0x1ff
        lit = self.literal(pc, s0f)
        n = pc + (2 if s0f == 255 else 1)
        self.compare(op, self.vsrc(s0f, lit), self.v[v1].copy(), 106, pc)
        return n

    def compare(self, op, a, b, dst, pc):
        kind, x = op >> 4, op & 0xf
        cmpx = bool(kind & 1)
        group = op & 0xf0
        if group in (0x80, 0x90):
            a, b, k = a.view(np.int32), b.view(np.int32), x & 7
        elif group in (0xc0, 0xd0):
            k = x & 7
        elif group in (0x00, 0x10):
            fa, fb = u2f(a), u2f(b)
            res = [np.zeros(LANES, bool), fa < fb, fa == fb, fa <= fb,
                   fa > fb, (fa < fb) | (fa > fb), fa >= fb,
                   ~(np.isnan(fa) | np.isnan(fb)), np.isnan(fa) | np.isnan(fb),
                   ~(fa >= fb), ~((fa < fb) | (fa > fb)), ~(fa > fb),
                   ~(fa <= fb), ~(fa == fb), ~(fa < fb),
                   np.ones(LANES, bool)][x]
            return self._cmp_write(res, dst, cmpx)
        else:
            raise Unsupported(f"vopc {op:#x} at pc {pc:#x}")
        if x > 7:
            raise Unsupported(f"vopc {op:#x} at pc {pc:#x}")
        res = [np.zeros(LANES, bool), a < b, a == b, a <= b, a > b, a != b,
               a >= b, np.ones(LANES, bool)][k]
        self._cmp_write(res, dst, cmpx)

    def _cmp_write(self, res, dst, cmpx):
        m = to_mask(res & self.active())
        self.set64(dst, m)
        if cmpx:
            self.set64(126, m)

    def vop3(self, pc, w, w1):
        op = (w >> 17) & 0x1ff
        d = w & 0xff
        s0f, s1f, s2f = w1 & 0x1ff, (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff
        neg = (w1 >> 29) & 7
        abs_ = (w >> 8) & 7
        clamp = (w >> 11) & 1
        if op < 0x100:
            self.compare(op, self.vsrc(s0f, 0), self.vsrc(s1f, 0), d, pc)
            return pc + 2
        srcs = [self.vsrc(s0f, 0), self.vsrc(s1f, 0), self.vsrc(s2f, 0)]
        if op in (0x125, 0x126, 0x127, 0x128, 0x129, 0x12a):
            sd = (w >> 8) & 0x7f
            self.valu(op, d, srcs, pc, sdst=sd, cin=s2f)
            return pc + 2
        if op == 0x100:
            sel = lanes_of(self.ssrc(s2f, 0, True))
            self.vwrite(d, np.where(sel, srcs[1], srcs[0]))
            return pc + 2
        if op == 0x101:
            self.s[d] = int(srcs[0][self.ssrc(s1f, 0) & 63])
            return pc + 2
        if op == 0x102:
            self.v[d][self.ssrc(s1f, 0) & 63] = self.ssrc(s0f, 0)
            return pc + 2
        if op == 0x176:
            out_lo, out_hi = self.v[d].copy(), self.v[d + 1].copy()
            carry = 0
            for l in np.nonzero(self.active())[0]:
                acc = (int(srcs[0][l]) * int(srcs[1][l]) +
                       (int(self.vsrc(s2f, 0)[l]) |
                        (int(self.vsrc(s2f + 1, 0)[l]) << 32)
                        if s2f >= 256 else self.ssrc(s2f, 0, True)))
                out_lo[l], out_hi[l] = acc & 0xffffffff, (acc >> 32) & \
                    0xffffffff
                carry |= (acc >> 64) << int(l)
            self.v[d], self.v[d + 1] = out_lo, out_hi
            self.set64((w >> 8) & 0x7f, carry)
            return pc + 2
        if neg or abs_:
            for i in range(3):
                if (abs_ >> i) & 1 or (neg >> i) & 1:
                    f = u2f(srcs[i]).copy()
                    if (abs_ >> i) & 1:
                        f = np.abs(f)
                    if (neg >> i) & 1:
                        f = -f
                    srcs[i] = f2u(f)
        self.valu(op, d, srcs, pc, clamp=clamp)
        return pc + 2

    def valu(self, op, d, s, pc, sdst=None, cin=None, clamp=0):
        a = s[0]
        b = s[1] if len(s) > 1 else None
        c = s[2] if len(s) > 2 else None
        i32 = lambda x: x.view(np.int32)
        act = self.active()
        u64 = lambda x: x.astype(np.uint64)
        i64 = lambda x: x.view(np.int32).astype(np.int64)
        fa = u2f(a)
        fb = u2f(b) if b is not None else None
        carry_out = None
        if op == 0x181:
            r = a
        elif op == 0x185:
            r = f2u(i32(a).astype(np.float32))
        elif op == 0x186:
            r = f2u(a.astype(np.float32))
        elif op == 0x187:
            f = np.nan_to_num(fa, nan=0.0)
            r = np.clip(np.trunc(f), 0, 4294967295.0).astype(np.uint64) \
                .astype(U32)
        elif op == 0x188:
            f = np.nan_to_num(fa, nan=0.0)
            r = np.clip(np.trunc(f), -2147483648.0, 2147483647.0) \
                .astype(np.int64).astype(np.int32).view(U32)
        elif op in (0x1aa, 0x1ab):
            with np.errstate(divide="ignore"):
                r = f2u(np.float32(1.0) / fa)
        elif op == 0x1a4:
            r = f2u(np.floor(fa))
        elif op == 0x1a1:
            r = f2u(np.trunc(fa))
        elif op == 0x1b7:
            r = ~a
        elif op == 0x103:
            r = f2u(fa + fb)
        elif op == 0x104:
            r = f2u(fa - fb)
        elif op == 0x105:
            r = f2u(fb - fa)
        elif op == 0x108:
            r = f2u(fa * fb)
        elif op == 0x109:
            r = ((sx24(a) * sx24(b)) & 0xffffffff).astype(U32)
        elif op == 0x10b:
            r = ((u64(a & 0xffffff) * u64(b & 0xffffff)) & 0xffffffff) \
                .astype(U32)
        elif op == 0x10c:
            r = ((u64(a & 0xffffff) * u64(b & 0xffffff)) >> 32).astype(U32)
        elif op == 0x10f:
            r = f2u(np.minimum(fa, fb))
        elif op == 0x110:
            r = f2u(np.maximum(fa, fb))
        elif op == 0x111:
            r = np.minimum(i32(a), i32(b)).view(U32)
        elif op == 0x112:
            r = np.maximum(i32(a), i32(b)).view(U32)
        elif op == 0x113:
            r = np.minimum(a, b)
        elif op == 0x114:
            r = np.maximum(a, b)
        elif op == 0x115:
            r = a >> (b & 31)
        elif op == 0x116:
            r = b >> (a & 31)
        elif op == 0x117:
            r = (i32(a) >> (b & 31).astype(np.int32)).view(U32)
        elif op == 0x118:
            r = (i32(b) >> (a & 31).astype(np.int32)).view(U32)
        elif op == 0x119:
            r = a << (b & 31)
        elif op == 0x11a:
            r = b << (a & 31)
        elif op == 0x11b:
            r = a & b
        elif op == 0x11c:
            r = a | b
        elif op == 0x11d:
            r = a ^ b
        elif op == 0x11e:
            r = ((((np.uint64(1) << u64(a & 31)) - np.uint64(1)) << u64(b & 31))
                 & np.uint64(0xffffffff)).astype(U32)
        elif op == 0x11f:
            old = u2f(self.v[d])
            r = f2u(fa * fb + old)
        elif op == 0x122:
            r = (popcount(a) + b).astype(U32)
        elif op in (0x123, 0x124):
            lane = np.arange(LANES)
            shift = lane if op == 0x123 else lane - 32
            below = np.array([(1 << int(min(max(k, 0), 32))) - 1 for k in shift],
                             dtype=np.uint64)
            r = (popcount((a.astype(np.uint64) & below).astype(U32)) + b) \
                .astype(U32)
        elif op in (0x125, 0x126, 0x127, 0x128, 0x129, 0x12a):
            if op == 0x127:
                a, b = b, a
                op = 0x126
            if op == 0x12a:
                a, b = b, a
                op = 0x129
            if op in (0x128, 0x129):
                cmask = self.get64(106) if cin is None or sdst is None else \
                    self.ssrc(cin, 0, True)
                cbit = lanes_of(cmask).astype(np.uint64)
            else:
                cbit = np.zeros(LANES, np.uint64)
            if op in (0x125, 0x128):
                full = u64(a) + u64(b) + cbit
                carry = full >> 32
            else:
                full = u64(a) - u64(b) - cbit
                carry = (u64(b) + cbit > u64(a)).astype(np.uint64)
            r = (full & 0xffffffff).astype(U32)
            carry_out = carry.astype(bool)
        elif op == 0x141:
            r = f2u(fa * fb + u2f(c))
        elif op == 0x143:
            r = ((u64(a & 0xffffff) * u64(b & 0xffffff) + u64(c)) & 0xffffffff) \
                .astype(U32)
        elif op == 0x142:
            r = ((sx24(a) * sx24(b) + i64(c)) & 0xffffffff).astype(U32)
        elif op == 0x10a:
            r = (((sx24(a) * sx24(b)) >> 32) & 0xffffffff).astype(U32)
        elif op == 0x148:
            off, wid = b & 31, c & 31
            r = np.where(wid == 0, U32(0),
                         (a >> off) & ((np.uint64(1) << u64(wid)) - 1).astype(U32))
        elif op == 0x149:
            off, wid = (b & 31).astype(np.int64), (c & 31).astype(np.int64)
            v = (a.astype(np.int64) >> off) & ((1 << wid) - 1)
            v = np.where((wid > 0) & ((v >> np.maximum(wid - 1, 0)) & 1 == 1),
                         v - (1 << wid), v)
            r = np.where(wid == 0, 0, v).astype(np.int64).astype(np.uint64) \
                .astype(U32)
        elif op == 0x14a:
            r = (a & b) | (~a & c)
        elif op == 0x14b:
            r = f2u(fa * fb + u2f(c))
        elif op == 0x151:
            r = np.minimum(np.minimum(i32(a), i32(b)), i32(c)).view(U32)
        elif op == 0x152:
            r = np.minimum(np.minimum(a, b), c)
        elif op == 0x154:
            r = np.maximum(np.maximum(i32(a), i32(b)), i32(c)).view(U32)
        elif op == 0x155:
            r = np.maximum(np.maximum(a, b), c)
        elif op == 0x158:
            x, y, z = i32(a), i32(b), i32(c)
            r = np.maximum(np.minimum(x, y), np.minimum(np.maximum(x, y), z)) \
                .view(U32)
        elif op == 0x159:
            r = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
        elif op == 0x15d:
            r = ((np.where(a > b, u64(a) - u64(b), u64(b) - u64(a)) + u64(c))
                 & 0xffffffff).astype(U32)
        elif op == 0x169:
            r = ((u64(a) * u64(b)) & 0xffffffff).astype(U32)
        elif op == 0x16a:
            r = ((u64(a) * u64(b)) >> 32).astype(U32)
        elif op == 0x16b:
            r = ((i64(a) * i64(b)) & 0xffffffff).astype(U32)
        elif op == 0x16c:
            r = ((i64(a) * i64(b)) >> 32).astype(np.int64).astype(np.uint64) \
                .astype(U32)
        else:
            raise Unsupported(f"valu {op:#x} at pc {pc:#x}")
        if clamp and op in (0x103, 0x104, 0x105, 0x108, 0x141, 0x14b, 0x11f):
            r = f2u(np.clip(u2f(r), 0.0, 1.0))
        self.vwrite(d, r)
        if carry_out is not None:
            m = to_mask(carry_out & act)
            self.set64(sdst if sdst is not None else 106, m)

    # --- LDS ------------------------------------------------------------------
    def ds(self, pc, w, w1):
        op = (w >> 18) & 0xff
        off0, off1 = w & 0xff, (w >> 8) & 0xff
        if (w >> 17) & 1:
            raise Unsupported(f"gds op {op:#x} at pc {pc:#x}")
        addr = self.v[w1 & 0xff]
        d0, d1, vd = (w1 >> 8) & 0xff, (w1 >> 16) & 0xff, (w1 >> 24) & 0xff
        lanes = np.nonzero(self.active())[0]
        lds = self.lds

        def rd(a, n):
            a = int(a)
            if a + n > len(lds):
                return 0
            return int.from_bytes(lds[a:a + n], "little")

        def wr(a, n, v):
            a = int(a)
            if self.lds_watch is not None and a <= self.lds_watch < a + n:
                print(f"lds watch: pc {pc:#x} lane {self.lane} wrote "
                      f"{int(v):#x} at {a:#x}")
            if a + n <= len(lds):
                lds[a:a + n] = int(v).to_bytes(n, "little")

        single = off0 | (off1 << 8)
        if op == 0x0d:
            for l in lanes:
                self.lane = int(l)
                wr(addr[l] + single, 4, self.v[d0][l])
        elif op in (0x0e, 0x0f):
            st = 4 * (64 if op == 0x0f else 1)
            for l in lanes:
                wr(addr[l] + off0 * st, 4, self.v[d0][l])
                wr(addr[l] + off1 * st, 4, self.v[d1][l])
        elif op == 0x4d:
            for l in lanes:
                wr(addr[l] + single, 4, self.v[d0][l])
                wr(addr[l] + single + 4, 4, self.v[d0 + 1][l])
        elif op in (0x4e, 0x4f):
            st = 8 * (64 if op == 0x4f else 1)
            for l in lanes:
                for k, (o, dd) in enumerate(((off0, d0), (off1, d1))):
                    wr(addr[l] + o * st, 4, self.v[dd][l])
                    wr(addr[l] + o * st + 4, 4, self.v[dd + 1][l])
        elif op in (0x1e, 0x1f):
            n = 1 if op == 0x1e else 2
            for l in lanes:
                wr(addr[l] + single, n, int(self.v[d0][l]) & ((1 << 8 * n) - 1))
        elif op == 0x36:
            out = self.v[vd].copy()
            for l in lanes:
                out[l] = rd(addr[l] + single, 4)
            self.v[vd] = out
        elif op in (0x37, 0x38):
            st = 4 * (64 if op == 0x38 else 1)
            o0, o1 = self.v[vd].copy(), self.v[vd + 1].copy()
            for l in lanes:
                o0[l] = rd(addr[l] + off0 * st, 4)
                o1[l] = rd(addr[l] + off1 * st, 4)
            self.v[vd], self.v[vd + 1] = o0, o1
        elif op == 0x76:
            o0, o1 = self.v[vd].copy(), self.v[vd + 1].copy()
            for l in lanes:
                o0[l] = rd(addr[l] + single, 4)
                o1[l] = rd(addr[l] + single + 4, 4)
            self.v[vd], self.v[vd + 1] = o0, o1
        elif op in (0x77, 0x78):
            st = 8 * (64 if op == 0x78 else 1)
            outs = [self.v[vd + i].copy() for i in range(4)]
            for l in lanes:
                for k, o in enumerate((off0, off1)):
                    outs[2 * k][l] = rd(addr[l] + o * st, 4)
                    outs[2 * k + 1][l] = rd(addr[l] + o * st + 4, 4)
            for i in range(4):
                self.v[vd + i] = outs[i]
        elif op in (0x39, 0x3a, 0x3b, 0x3c):
            n = 1 if op in (0x39, 0x3a) else 2
            signed = op in (0x39, 0x3b)
            out = self.v[vd].copy()
            for l in lanes:
                v = rd(addr[l] + single, n)
                out[l] = (sext(v, 8 * n) & 0xffffffff) if signed else v
            self.v[vd] = out
        else:
            raise Unsupported(f"ds {op:#x} at pc {pc:#x}")
        return pc + 2

    # --- buffers ----------------------------------------------------------------
    def vsharp(self, r):
        return [self.s[r + i] for i in range(4)]

    def mubuf(self, pc, w, w1, mtbuf):
        if mtbuf is not None:
            op = (w >> 16) & 7
            fmt = ((w >> 19) & 0xf, (w >> 23) & 7)
        else:
            op = (w >> 18) & 0x7f
            fmt = None
        offset, offen, idxen = w & 0xfff, (w >> 12) & 1, (w >> 13) & 1
        if (w >> 15) & 1:
            raise Unsupported(f"addr64 buffer at pc {pc:#x}")
        vaddr, vdata = w1 & 0xff, (w1 >> 8) & 0xff
        V = self.vsharp(((w1 >> 16) & 0x1f) * 4)
        soff = self.ssrc((w1 >> 24) & 0xff, 0)
        base = V[0] | ((V[1] & 0xffff) << 32)
        stride, nrec = (V[1] >> 16) & 0x3fff, V[2]
        if (V[1] >> 31) & 1:
            raise Unsupported(f"swizzled buffer at pc {pc:#x}")
        lanes = np.nonzero(self.active())[0]
        if mtbuf is not None:
            kind = "fmt_load" if op < 4 else "fmt_store"
            count = (op & 3) + 1
            dfmt, nfmt = fmt
        elif op <= 3:
            kind, count = "fmt_load", op + 1
            dfmt, nfmt = (V[3] >> 15) & 0xf, (V[3] >> 12) & 7
        elif op <= 7:
            kind, count = "fmt_store", op - 3
            dfmt, nfmt = (V[3] >> 15) & 0xf, (V[3] >> 12) & 7
        elif op in (0x08, 0x09, 0x0a, 0x0b):
            kind, count = "sub_load", 1
        elif op in (0x0c, 0x0d, 0x0e, 0x0f):
            kind, count = "load", {0x0c: 1, 0x0d: 2, 0x0e: 4, 0x0f: 3}[op]
        elif op in (0x18, 0x1a):
            kind, count = "sub_store", 1
        elif op in (0x1c, 0x1d, 0x1e, 0x1f):
            kind, count = "store", {0x1c: 1, 0x1d: 2, 0x1e: 4, 0x1f: 3}[op]
        else:
            raise Unsupported(f"mubuf {op:#x} at pc {pc:#x}")
        outs = [self.v[vdata + i].copy() for i in range(count)] \
            if "load" in kind else None
        for l in lanes:
            index = int(self.v[vaddr][l]) if idxen else 0
            voff = int(self.v[vaddr + idxen][l]) if offen else 0
            off = soff + offset + voff
            if stride == 0:
                oob = off >= nrec
            else:
                oob = index >= nrec
            addr = base + index * stride + off
            if kind == "load":
                for i in range(count):
                    outs[i][l] = 0 if oob else self.mem.u32(addr + 4 * i)
            elif kind == "sub_load":
                n = 1 if op in (0x08, 0x09) else 2
                v = 0 if oob else int.from_bytes(self.mem.read(addr, n),
                                                 "little")
                if op in (0x09, 0x0b):
                    v = sext(v, 8 * n) & 0xffffffff
                outs[0][l] = v
            elif kind == "fmt_load":
                if dfmt not in DFMT:
                    raise Unsupported(f"buffer dfmt {dfmt} at pc {pc:#x}")
                bits, comps = DFMT[dfmt]
                vals = []
                for cmp in range(comps):
                    nb = bits // 8
                    raw = 0 if oob else int.from_bytes(
                        self.mem.read(addr + cmp * nb, nb), "little")
                    vals.append(decode_component(raw, bits, nfmt))
                for cmp in range(comps, 4):
                    vals.append(1 if cmp == 3 and nfmt in (4, 5) else
                                (int(f2u(np.float32(1.0))) if cmp == 3 else 0))
                for i in range(count):
                    outs[i][l] = vals[i]
            elif oob:
                continue
            elif kind == "store":
                for i in range(count):
                    self.mem.write(addr + 4 * i,
                                   int(self.v[vdata + i][l]).to_bytes(4,
                                                                      "little"))
            elif kind == "sub_store":
                n = 1 if op == 0x18 else 2
                self.mem.write(addr, (int(self.v[vdata][l]) &
                                      ((1 << 8 * n) - 1)).to_bytes(n, "little"))
            elif kind == "fmt_store":
                bits, comps = DFMT[dfmt]
                nb = bits // 8
                for cmp in range(min(count, comps)):
                    self.mem.write(addr + cmp * nb, encode_component(
                        int(self.v[vdata + cmp][l]), bits, nfmt)
                        .to_bytes(nb, "little"))
        if outs is not None:
            for i in range(count):
                self.v[vdata + i] = outs[i]
        return pc + 2

    # --- images ----------------------------------------------------------------
    def mimg(self, pc, w, w1):
        op = (w >> 18) & 0x7f
        dmask, da = (w >> 8) & 0xf, (w >> 14) & 1
        if (w >> 15) & 1:
            raise Unsupported(f"r128 image at pc {pc:#x}")
        vaddr, vdata = w1 & 0xff, (w1 >> 8) & 0xff
        r = ((w1 >> 16) & 0x1f) * 4
        T = TImage([self.s[r + i] for i in range(8)])
        if op not in (0x00, 0x08):
            raise Unsupported(f"mimg {op:#x} at pc {pc:#x}")
        lanes = np.nonzero(self.active())[0]
        n = bin(dmask).count("1")
        nb = T.bits // 8
        outs = [self.v[vdata + i].copy() for i in range(n)] if op == 0 else None
        for l in lanes:
            self.lane = int(l)
            x = int(self.v[vaddr][l])
            y = int(self.v[vaddr + 1][l]) if T.type != 8 else 0
            layer = int(self.v[vaddr + 2][l]) if da or T.type == 10 else 0
            inb = x < T.width and y < T.height
            a = T.address(x, y, layer)
            if op == 0x00:
                vals = []
                for cmp in range(T.comps):
                    raw = int.from_bytes(self.mem.read(a + cmp * nb, nb),
                                         "little") if inb else 0
                    vals.append(decode_component(raw, T.bits, T.nfmt))
                for cmp in range(T.comps, 4):
                    vals.append(0)
                vals = apply_dst_sel(vals, T.dst_sel, 4)
                k = 0
                for cmp in range(4):
                    if (dmask >> cmp) & 1:
                        outs[k][l] = vals[cmp] if inb else 0
                        k += 1
            elif inb:
                k = 0
                for cmp in range(T.comps):
                    if (dmask >> cmp) & 1:
                        v = int(self.probe[k][l] if self.probe_pc is not None
                                else self.v[vdata + k][l])
                        k += 1
                        self.mem.write(a + cmp * nb, encode_component(
                            v, T.bits, T.nfmt).to_bytes(nb, "little"))
        if outs is not None:
            for i in range(n):
                self.v[vdata + i] = outs[i]
        return pc + 2


def popcount(x):
    x = x.astype(np.uint64)
    c = np.zeros_like(x)
    for _ in range(32):
        c += x & 1
        x >>= 1
    return c.astype(U32)


def sx24(x):
    v = (x & 0xffffff).astype(np.int64)
    return np.where(v >= 1 << 23, v - (1 << 24), v)


# --- capture loading ---------------------------------------------------------
def load_capture(jsonl):
    inputs, bound, info, shader = {}, {}, None, None
    for line in open(jsonl):
        d = json.loads(line)
        t = d.get("t")
        if t == "dispatch_inputs":
            info = d
        if t != "dispatch_dump" or not d.get("file"):
            continue
        if d["kind"] == "shader":
            shader = (int(d["base"], 16), d["file"])
        elif d["kind"] == "buffer":
            store = {"input": inputs, "bound": bound}.get(d["phase"])
            if store is not None:
                store.setdefault(d["binding"], (int(d["base"], 16),
                                                d["offset"], d["file"]))
    outputs = []
    for line in open(jsonl):
        d = json.loads(line)
        if d.get("t") == "dispatch_dump" and d["phase"] == "output" and \
                d.get("file"):
            outputs.append((int(d["base"], 16), d["file"]))
    return info, shader, inputs, bound, outputs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("capture", help="frame_N.jsonl of a DELTA_GPU_CAPTURE_CS run")
    p.add_argument("--tg", default="64,1,1", help="threads per group x,y,z")
    p.add_argument("--user-sgpr", type=int, required=True)
    p.add_argument("--tgid", type=int, default=1, help="TGID enable bits")
    p.add_argument("--lds", type=int, default=0, help="LDS size field (x128 dwords)")
    p.add_argument("--groups", default="", help="only run these group ids (a,b,..)")
    p.add_argument("--trace-pc", default="", help="print lane values at pc (hex)")
    p.add_argument("--report", type=int, default=12, help="differences to list")
    p.add_argument("--watch", default="", help="report every write to ADDR")
    p.add_argument("--lds-watch", default="", help="report LDS writes to ADDR")
    p.add_argument("--probe", default="",
                   help="PC:REG, mirror DELTA_GPU_GCN_CSVGPR_PC/_CSVGPR (REG "
                        ">= 1000 is s[REG-1000]); compare a probed capture")
    p.add_argument("--trace-lane", default="",
                   help="G:L, print every register lane L of group G changes")
    a = p.parse_args()

    info, shader, inputs, bound, outputs = load_capture(a.capture)
    if not info or not shader:
        sys.exit("capture has no dispatch_inputs record / shader dump")
    mem = Memory()
    for b, (base, off, f) in inputs.items():
        mem.add(base, open(f, "rb").read())
    # The bound snapshots are what the shader saw; they override the guest
    # copy, which can lag behind dispatches still queued in the batch.
    for b, (base, off, f) in bound.items():
        if off:
            mem.add(off, open(f, "rb").read())
    code_base, code_file = shader
    code = list(struct.unpack(f"<{os.path.getsize(code_file) // 4}I",
                              open(code_file, "rb").read())) + [0xbf810000] * 4

    tg = [int(x) for x in a.tg.split(",")]
    threads = tg[0] * tg[1] * tg[2]
    if threads > LANES:
        sys.exit("multi-wave groups are not supported")
    groups = [int(x) for x in info["groups"]]
    ud = [int(x, 16) for x in info["user_data"]]
    only = {int(x) for x in a.groups.split(",") if x}
    trace_pcs = {int(x, 16) for x in a.trace_pc.split(",") if x}

    ctx = {}
    if a.watch:
        mem.watch = int(a.watch, 16)
        mem.on_watch = lambda addr, data: print(
            f"watch: group {ctx['gid']} pc {ctx['wave'].pc:#x} "
            f"lane {getattr(ctx['wave'], 'lane', '?')} wrote "
            f"{data.hex()} at {addr:#x}")
    total = 0
    for gz in range(groups[2]):
        for gy in range(groups[1]):
            for gx in range(groups[0]):
                gid = gx + groups[0] * (gy + groups[1] * gz)
                if only and gid not in only:
                    continue
                wave = Wave(mem, code, code_base, a.lds * 128 * 4)
                ctx["gid"], ctx["wave"] = gid, wave
                if a.lds_watch:
                    wave.lds_watch = int(a.lds_watch, 16)
                if a.probe:
                    ppc, preg = a.probe.split(":")
                    wave.probe_pc, wave.probe_reg = int(ppc, 16), int(preg)
                for i in range(a.user_sgpr):
                    wave.s[i] = ud[i]
                sg = a.user_sgpr
                for bit, val in ((1, gx), (2, gy), (4, gz)):
                    if a.tgid & bit:
                        wave.s[sg] = val
                        sg += 1
                lane = np.arange(LANES)
                wave.v[0] = (lane % tg[0]).astype(U32)
                wave.v[1] = ((lane // tg[0]) % tg[1]).astype(U32)
                wave.v[2] = (lane // (tg[0] * tg[1])).astype(U32)
                wave.set64(126, (1 << threads) - 1)
                if a.trace_lane and int(a.trace_lane.split(":")[0]) == gid:
                    tl = int(a.trace_lane.split(":")[1])
                    snap = {}

                    def tr(w, pc, tl=tl, snap=snap):
                        if "pc" in snap:
                            ch = [f"v{i}={int(w.v[i][tl]):#x}" for i in range(256)
                                  if int(w.v[i][tl]) != snap["v"][i]]
                            ch += [f"s{i}={w.s[i]:#x}" for i in range(128)
                                   if w.s[i] != snap["s"][i]]
                            on = (snap["exec"] >> tl) & 1
                            print(f"{snap['pc']:04x} {'+' if on else '-'} "
                                  + " ".join(ch))
                        snap.update(pc=pc, v=[int(x) for x in w.v[:, tl]],
                                    s=list(w.s), exec=w.exec)
                    wave.trace = tr
                elif trace_pcs:
                    def tr(w, pc):
                        if pc in trace_pcs:
                            print(f"group {gid} pc {pc:#x} exec {w.exec:016x} "
                                  f"vcc {w.get64(106):016x} scc {w.scc}")
                    wave.trace = tr
                try:
                    wave.run()
                except Unsupported as e:
                    sys.exit(f"group {gid}: unsupported: {e}")
                total += wave.executed
    print(f"executed {total} wave instructions")
    if mem.misses:
        print("reads/writes outside the captured memory:",
              ", ".join(f"{k:#x}x{v}" for k, v in sorted(mem.misses.items())[:8]))

    for base, f in outputs:
        gpu = np.frombuffer(open(f, "rb").read(), np.uint8)
        ours = np.frombuffer(mem.read(base, gpu.size), np.uint8)
        diff = np.nonzero(gpu != ours)[0]
        print(f"output {base:#x} +{gpu.size:#x}: {diff.size} bytes differ")
        for i in diff[:a.report]:
            print(f"  {base + int(i):#x}: gpu {gpu[i]:3d} cpu {ours[i]:3d}")


if __name__ == "__main__":
    main()
