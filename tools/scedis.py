#!/usr/bin/env python3
# Disassemble / search a decrypted SCE ELF (type 0xfe18) that objdump can't parse.
# Maps a vaddr to its file offset via the PT_LOAD program headers, carves the
# bytes, and shells out to `objdump -b binary` (verified to work on raw x86-64).
#
#   scedis.py <module> dis  <vaddr_hex> <len>     # disassemble a range
#   scedis.py <module> find <hexbytes>            # find byte pattern -> vaddrs
#   scedis.py <module> seg                         # list LOAD segments
import struct, subprocess, sys, tempfile, os

def loads(path):
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"\x7fELF"
    e_phoff = struct.unpack_from("<Q", d, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", d, 0x36)[0]
    e_phnum = struct.unpack_from("<H", d, 0x38)[0]
    segs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", d, off)[0]
        p_offset = struct.unpack_from("<Q", d, off + 0x08)[0]
        p_vaddr = struct.unpack_from("<Q", d, off + 0x10)[0]
        p_filesz = struct.unpack_from("<Q", d, off + 0x20)[0]
        p_memsz = struct.unpack_from("<Q", d, off + 0x28)[0]
        p_flags = struct.unpack_from("<I", d, off + 0x04)[0]
        if p_type == 1:  # PT_LOAD
            segs.append((p_vaddr, p_offset, p_filesz, p_memsz, p_flags))
    return d, segs

def v2o(segs, vaddr):
    for v, o, fz, mz, fl in segs:
        if v <= vaddr < v + fz:
            return o + (vaddr - v)
    return None

def main():
    path, cmd = sys.argv[1], sys.argv[2]
    d, segs = loads(path)
    if cmd == "seg":
        for v, o, fz, mz, fl in segs:
            rwx = "".join(c if fl & b else "-" for c, b in (("r",4),("w",2),("x",1)))
            print(f"vaddr={v:#x} foff={o:#x} filesz={fz:#x} memsz={mz:#x} {rwx}")
    elif cmd == "ripref":
        # find rip-relative refs to target: any disp32 D at vaddr V where the
        # next-insn anchor (V+4)+D == target. Catches lea/mov reg,[rip+disp]
        # (disp32 is the instruction's last 4 bytes). Some false positives.
        target = int(sys.argv[3], 16)
        for v, o, fz, mz, fl in segs:
            if not (fl & 1):
                continue
            for i in range(o, o + fz - 4):
                disp = struct.unpack_from("<i", d, i)[0]
                va = i - o + v
                if (va + 4) + disp == target:
                    print(f"disp32@vaddr={va:#x} -> {target:#x} (insn starts a few bytes before)")
    elif cmd == "xref":
        # find E8 (call) / E9 (jmp) rel32 whose target == given vaddr
        target = int(sys.argv[3], 16)
        for v, o, fz, mz, fl in segs:
            if not (fl & 1):  # exec only
                continue
            for i in range(o, o + fz - 5):
                if d[i] in (0xE8, 0xE9):
                    rel = struct.unpack_from("<i", d, i + 1)[0]
                    site = (i - o + v)
                    if site + 5 + rel == target:
                        kind = "call" if d[i] == 0xE8 else "jmp"
                        print(f"{kind} site vaddr={site:#x} -> {target:#x}")
    elif cmd == "find":
        pat = bytes.fromhex(sys.argv[3])
        i = 0
        while True:
            j = d.find(pat, i)
            if j < 0:
                break
            v = next((j - o + vv for vv, o, fz, mz, fl in segs if o <= j < o + fz), None)
            print(f"foff={j:#x} vaddr={v:#x}" if v is not None else f"foff={j:#x} (no seg)")
            i = j + 1
    elif cmd == "dis":
        vaddr = int(sys.argv[3], 16)
        length = int(sys.argv[4], 0)
        o = v2o(segs, vaddr)
        if o is None:
            print("vaddr not in any LOAD segment", file=sys.stderr); sys.exit(1)
        chunk = d[o:o + length]
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tf.write(chunk)
            tmp = tf.name
        try:
            p = subprocess.run(
                ["objdump", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                 "-D", f"--adjust-vma={vaddr:#x}", tmp], capture_output=True)
        finally:
            os.unlink(tmp)
        out = p.stdout.decode(errors="replace")
        for line in out.splitlines():
            s = line.strip()
            if s and s[0] in "0123456789abcdef" and ":" in s:
                print(line)

if __name__ == "__main__":
    main()
