#!/usr/bin/env python3
"""Sony symbol NIDs: name -> id, and a brute-force search id -> name.

A NID is the first 8 bytes of SHA-1(name + salt), read little-endian, printed
in Sony's own base64 alphabet. Nothing in the tree could compute one, so an
unknown export could only be identified by reading its code.

  nid.py hash <name>...            print the id and encoded form of each name
  nid.py find <encoded|0xid> [...] brute-force names against the built-in
                                   generator (add --words extra,words)
"""
import hashlib
import itertools
import struct
import sys

SALT = bytes([0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
              0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30])
ALPHABET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+-")


def nid(name: str) -> int:
    digest = hashlib.sha1(name.encode() + SALT).digest()
    return struct.unpack("<Q", digest[:8])[0]


def encode(value: int) -> str:
    """Sony's 11-character form: six bits per character taken from the top,
    with the last character carrying the low nibble shifted up by two."""
    out = [ALPHABET[(value >> (58 - i * 6)) & 0x3F] for i in range(10)]
    out.append(ALPHABET[(value & 0xF) << 2])
    return "".join(out)


def decode(text: str) -> int:
    v = 0
    for ch in text[:10]:
        v = (v << 6) | ALPHABET.index(ch)
    return (v << 4) | (ALPHABET.index(text[10]) >> 2)


PREFIXES = ["sceKernel", "sce", "_sce", "scePthread"]
# Words seen across Sony export names, in the CamelCase they appear in.
WORDS = """Get Set Is Has Check Query Read Write Load Store Open Close Create
Destroy Init Initialize Finalize Term Terminate Enable Disable Start Stop
Config Setting Settings Registry Reg Mgr Value Param Parameter Flag Flags
Mode State Status Info Information Type Kind Id Index Key Name Path
System Kernel Process Proc Thread Fiber Memory Mem Heap Pool Alloc Free
Debug Dev Devkit Retail Prospero Orbis Neo Base Pro Model Hardware Hw
Feature Features Capability Cap Support Supported Available Avail Budget
Sdk Version Ver Revision Rev Build Firmware Fw Update Boot Sandbox
Profil Profiling Profiler Perf Performance Trace Tracing Razor Sanitizer
User Account Console Machine Unit Product Region Locale Language
Time Clock Tsc Freq Frequency Rate Count Counter Size Length Limit Max Min
Enabled Disabled Allow Allowed Deny Permission Priv Privilege Auth
Internal Private Public Common Shared Global Local Current Default
Cpu Gpu Core Cores Cluster Affinity Priority Sched Scheduler
File Dir Directory Mount Fs Vfs Blob Bin Str String Int Uint Bool
Random Rng Entropy Uuid Guid Hash Crc Checksum
App Application Title Game Program Module Library Lib Prx Elf Self
Save Data Savedata Trophy Np Psn Net Network Socket
Video Audio Display Screen Output Input Pad Controller
Psm Ml Upscaler Resolution Refresh""".split()


def candidates(extra):
    words = WORDS + list(extra)
    for prefix in PREFIXES:
        for n in (1, 2, 3):
            for combo in itertools.permutations(words, n):
                yield prefix + "".join(combo)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "hash":
        for name in sys.argv[2:]:
            v = nid(name)
            print(f"{name:<52} {v:#018x}  {encode(v)}")
        return 0
    if cmd == "find":
        extra = []
        args = []
        for a in sys.argv[2:]:
            if a.startswith("--words"):
                extra = a.split("=", 1)[1].split(",")
            else:
                args.append(a)
        wanted = {}
        for a in args:
            wanted[int(a, 16) if a.startswith("0x") else decode(a)] = a
        print(f"searching for {len(wanted)} id(s)...", file=sys.stderr)
        tried = 0
        for name in candidates(extra):
            tried += 1
            v = nid(name)
            if v in wanted:
                print(f"FOUND {wanted[v]} = {name}  ({v:#018x})")
                del wanted[v]
                if not wanted:
                    break
            if tried % 2000000 == 0:
                print(f"  {tried} tried...", file=sys.stderr)
        print(f"done, {tried} candidates tried", file=sys.stderr)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
