#!/usr/bin/env python3
# Extract a fake-signed PS4 .pkg: decrypt the PFS, inflate the inner PFSC image,
# walk the filesystem and rebuild each fSELF into a loadable ELF.
#
# Key-free: works on fake/repacked pkgs using the public fake-pkg RSA keyset.
# Retail pkgs (Sony-signed image_key) are not supported. Needs pycryptodome.
#
# Usage: pkg_extract.py <game.pkg> <out_dir>
import struct, hmac, hashlib, zlib, os, sys
from Crypto.Cipher import AES

# Public fake-pkg keysets (from LibOrbisPkg): RSA-2048 modulus + private exponent.
N_FAKE = 0xc6cf71e7e59af0d12a2c458bf92a0ec143058bc37117801dcd497dde359d259ba0d7a0f27d6c087eaa5502682b23c644b84418eb56cf16a24803c9e74f87eb3d30c31588bf20e79dff770cde1d241e63a94f8abf5bbe601968333bfced9f474e5ff8eacb3d00bd6701f92c6dc6ac1364e76714f3dc52696ab9832c4230131bb2d8a5020d79ed96b10df8cc0cdf81954f035809570e80692efeff5277ea7528a8fbc9bebf9fbbb7798e1805e180bd50349481d353c269a2d24ccf6cf4572c104a3ffb22fd8b97e2c95ba62bcdd61b6bdb687f4bc2a05034c005e58def2467ff9340cf2d62a2a050b1f13aa83dfd80d1f9b80522afc8354590588ee33a7cbd3e27
D_FAKE = 0x7f76cd0ee2d4de051cc6d9a80e8dfa7bca1eaa271a40f8f1228735dddbfdeef8c2bcbd01fb8be23e63b2b1225c56496e11be07440b9a2666d1492c8fd31bcfa4a1b8d1fba49ed2212883098af6a00ba3d60f9b6368ccbc0c4e145b27a4a9f42bb9b87bc0e651ad1d77d46bb9ce20d126667e5e9ea2e96b90f373b8528f4411030c1397393d132258d5438249da6e7ca1c58ca5b009e0ce3ddff49d3c9715e26ac72b3c509323dbba4a226644ac78bb0e1a2743b57167aff4ab48469373d042ab9363e56c9ade5024c0237d99793f2207e0c148561bdf830912b42d456bc9c06885999079961ad7f54d1f3783404aec3937a680927dc580c7d66ffe8a7989c6b1
N_DK3  = 0xd212fc335f6ddb831609628b0356273782d477853529392d526b8c4c8cfb06c1845be7d4f7bcd24e6245cd2abbd77776453655273fb3f5f98eda4befaa59aeb39bea5498d206326a58312ae0d44f90b50a7decf43a9c52672d99318e0c43e682fe0746e12e50d41f2d2f7ed908ba06b3bf2e203f4e3ffe44ffaa50435791699449158282e40f4c8d9d2cc95b1d64bf888bd4c594e76547841ee57910fb989347b97d8512a640982cf792bc951932ede890560d65c1aa78c62e54fd5f54a1f67ee5e05f61c120b4b9b4330870e4df8956ed012946775f8cb8a9f51e2eb3b9bfe009b78d28d4a6c3b81e1f07ebb4120b95b88530fddc3913d07cdc8fedf9c9a3c1
D_DK3  = 0x32d903908fbdb08f572b285e0b8db3ea5cd17ea890888cdd6a80bbb1dfc1f70daa32f0b77ccb88800e8b64b0be4cd60e9b8c1e2a64e1f35cd77601415e935c94fedd4662c31b5ae2a0bc2debc3980aa7b7856970682b644ab31fcc7ddc7c26f477f65cf2ae5a442dd3ab16620419bafb90ffe23050896ecb56b2ebc09116925e308eaec7945dfd35e120f8ad3ebc08bfc036749fd5bb5208fd0666f37ab304f475295de95faa1030b20f5a1ac12ab3fecb21ad80ec8f20091cdbc55894c29cc6ce82653e5790bca98b06b4f072f677df9864f1ecfe372dbcae8c08811fc3c9891ac742824b2edc8e8d73ceb1cc01d90870873c4408ec498f815ae240ff77fc0d

def _rsa(ct, N, D):
    m = pow(int.from_bytes(ct, "big"), D, N).to_bytes(0x100, "big")
    return m[m.index(0, 2) + 1:] if m[0] == 0 and m[1] == 2 else None

le = lambda b, o, n: int.from_bytes(b[o:o + n], "little")

class Raw:
    def __init__(s, f, base): s.f = f; s.base = base
    def read(s, o, n): s.f.seek(s.base + o); return s.f.read(n)
class Sub:
    def __init__(s, i, o): s.i = i; s.o = o
    def read(s, o, n): return s.i.read(s.o + o, n)
class Xts:
    def __init__(s, i, tk, dk, ss):
        s.i = i; s.et = AES.new(tk, AES.MODE_ECB); s.ed = AES.new(dk, AES.MODE_ECB); s.ss = ss
    def _dec(s, si, data):
        t = bytearray(s.et.encrypt(si.to_bytes(16, "little"))); out = bytearray()
        for i in range(0, len(data), 16):
            x = bytes(a ^ b for a, b in zip(data[i:i + 16], t)); p = s.ed.decrypt(x)
            out += bytes(a ^ b for a, b in zip(p, t))
            c = 0; nt = bytearray(16)
            for j in range(16): nt[j] = ((t[j] << 1) | c) & 0xff; c = (t[j] >> 7) & 1
            if c: nt[0] ^= 0x87
            t = nt
        return bytes(out)
    def read(s, o, n):
        out = bytearray(); p = o
        while p < o + n:
            si = p // 0x1000; so = p % 0x1000; take = min(0x1000 - so, o + n - p)
            raw = s.i.read(si * 0x1000, 0x1000)
            out += (raw if si < s.ss else s._dec(si, raw))[so:so + take]; p += take
        return bytes(out)
class Pfsc:  # zlib-compressed inner image
    def __init__(s, p):
        h = p.read(0, 0x30); _, _, _, _, s.bs, s.bo, _, s.dl = struct.unpack_from("<iiiiQQQQ", h, 0)
        s.n = s.dl // s.bs; s.map = struct.unpack("<%dQ" % (s.n + 1), p.read(s.bo, (s.n + 1) * 8)); s.p = p
    def read(s, o, n):
        out = bytearray()
        while n > 0:
            bi = o // s.bs; bo = o % s.bs; a, b = s.map[bi], s.map[bi + 1]; c = s.p.read(a, b - a)
            blk = c if (b - a) == s.bs else zlib.decompress(c)
            take = min(s.bs - bo, n); out += blk[bo:bo + take]; o += take; n -= take
        return bytes(out)

class Pfs:
    def __init__(s, r, ekpfs):
        h = r.read(0, 0x400); s.bs = le(h, 0x20, 4); mode = le(h, 0x1c, 2)
        s.nd = le(h, 0x30, 8); s.ndb = le(h, 0x40, 8); seed = h[0x370:0x380]
        signed = mode & 1
        if mode & 4:
            d = hmac.new(ekpfs, struct.pack("<I", 1) + seed, hashlib.sha256).digest()
            r = Xts(r, d[:16], d[16:32], s.bs // 0x1000)
        s.r = r; s.dsz = 0x2C8 if signed else 0xA8; s.signed = signed
        per = s.bs // s.dsz; s.ino = []; tot = 0
        for bi in range(max(1, s.ndb)):
            blk = r.read((1 + bi) * s.bs, s.bs)
            for j in range(per):
                if tot >= s.nd: break
                d = blk[j * s.dsz:(j + 1) * s.dsz]
                st = le(d, 0x84, 4) if signed else le(d, 0x64, 4)
                s.ino.append((le(d, 0, 2), le(d, 8, 8), le(d, 0x60, 4), st)); tot += 1
    def data(s, i):
        _, size, _, st = s.ino[i]; out = bytearray(); need = size; b = st
        while need > 0:
            c = s.r.read(b * s.bs, s.bs); out += c[:need]; need -= min(need, len(c)); b += 1
        return bytes(out[:size])
    def walk(s, i, pre=""):
        _, _, blocks, st = s.ino[i]; res = []
        for bb in range(blocks):
            d = s.r.read((st + bb) * s.bs, s.bs); o = 0
            while o < s.bs - 17:
                ch = le(d, o, 4); ty = le(d, o + 4, 4); nl = le(d, o + 8, 4); es = le(d, o + 12, 4)
                if es == 0: break
                if 0 < nl < 256:
                    nm = d[o + 16:o + 16 + nl].decode("latin1", "replace")
                    if ty == 2: res.append((pre + "/" + nm, ch))
                    elif ty == 3 and nm not in (".", ".."): res += s.walk(ch, pre + "/" + nm)
                o += es
        return res

def get_ekpfs(f):
    R = lambda o, n: (f.seek(o), f.read(n))[1]
    ec = int.from_bytes(R(0x10, 4), "big"); to = int.from_bytes(R(0x18, 4), "big")
    tb = R(to, ec * 0x20); rows = {}
    for i in range(ec):
        e = tb[i * 0x20:(i + 1) * 0x20]
        rows[int.from_bytes(e[:4], "big")] = (e, int.from_bytes(e[16:20], "big"), int.from_bytes(e[20:24], "big"))
    ik_row, ik_off, ik_sz = rows[0x20]; _, ek_off, ek_sz = rows[0x10]
    dk3 = _rsa(R(ek_off, ek_sz)[0x400:0x500], N_DK3, D_DK3)
    ivk = hashlib.sha256(ik_row + dk3).digest()
    imdec = AES.new(ivk[16:32], AES.MODE_CBC, ivk[:16]).decrypt(R(ik_off, ik_sz))
    return _rsa(imdec, N_FAKE, D_FAKE)

def self2elf(data):
    if data[:4] != b'\x4f\x15\x3d\x1d': return None
    nseg = le(data, 0x18, 2)
    segs = [struct.unpack_from("<QQQQ", data, 0x20 + i * 0x20) for i in range(nseg)]
    e = 0x20 + nseg * 0x20
    if data[e:e + 4] != b'\x7fELF': return None
    phoff = le(data, e + 0x20, 8); ehsize = le(data, e + 0x34, 2)
    phentsize = le(data, e + 0x36, 2); phnum = le(data, e + 0x38, 2)
    phdrs = [(le(data, e + phoff + i * phentsize + 0x08, 8), le(data, e + phoff + i * phentsize + 0x20, 8))
             for i in range(phnum)]
    hdr_end = phoff + phnum * phentsize
    out = bytearray(max([hdr_end] + [o + s for o, s in phdrs]))
    out[0:hdr_end] = data[e:e + hdr_end]
    for flags, off, filesz, _ in segs:
        if not (flags & 0x800): continue
        idx = (flags >> 20) & 0xFFF
        if idx < len(phdrs):
            po, _ = phdrs[idx]; out[po:po + filesz] = data[off:off + filesz]
    return bytes(out)

def main():
    pkg, out = sys.argv[1], sys.argv[2]
    f = open(pkg, "rb")
    ekpfs = get_ekpfs(f)
    if not ekpfs:
        sys.exit("Could not recover EKPFS -- not a fake pkg?")
    print("EKPFS", ekpfs.hex())
    outer = Pfs(Raw(f, 0x700000), ekpfs)
    # outer uroot holds the contiguous, PFSC-compressed inner image at block 11
    inner = Pfs(Pfsc(Sub(outer.r, 11 * outer.bs)), ekpfs)
    files = inner.walk(0)
    print("%d files in image" % len(files))
    os.makedirs(out, exist_ok=True)
    for path, ino in files:
        data = inner.data(ino)
        elf = self2elf(data)
        dst = os.path.join(out, path.lstrip("/"))
        if elf is not None:
            dst = dst.rsplit(".", 1)[0] + ".elf" if "." in os.path.basename(dst) else dst + ".elf"
            data = elf
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(data)
    print("extracted to", out)

if __name__ == "__main__":
    main()
