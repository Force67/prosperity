#!/usr/bin/env python3
# Recover C++ class names from a stripped PS4/PS5 eboot dumped with
# DELTA_DUMP_MODULE. Walks the R_X86_64_RELATIVE relocations to find which
# vtable slot points at a given function, then reads vtable[-1] -> typeinfo ->
# +8 name string. Turns a probe backtrace of bare offsets into class + slot,
# which is the only way to tell what a 254 MB stripped binary is waiting on.
#
#   tools/eboot_rtti.py <fn-vaddr> [<fn-vaddr>...]
import os, struct, sys
d=open(os.environ.get('ELF','/tmp/dumped_module.elf'),'rb').read()
phoff=struct.unpack_from('<Q',d,0x20)[0]
phentsize,phnum=struct.unpack_from('<HH',d,0x36)
segs=[]
for i in range(phnum):
    o=phoff+i*phentsize
    t,_=struct.unpack_from('<II',d,o)
    off,va,pa,fsz,msz,al=struct.unpack_from('<QQQQQQ',d,o+8)
    segs.append((t,off,va,fsz))
dyn=[s for s in segs if s[0]==2][0]
tags={}
o=dyn[1]
while True:
    tag,val=struct.unpack_from('<QQ',d,o)
    if tag==0: break
    tags.setdefault(tag,[]).append(val); o+=16
def v2f(va):
    for t,off,v,fsz in segs:
        if t in (1,0x61000002) and v<=va<v+fsz: return off+(va-v)
RELA=v2f(tags[7][0]); n=tags[8][0]//24
rel={}
byadd={}
for i in range(n):
    off,info,add=struct.unpack_from('<QQq',d,RELA+i*24)
    if (info&0xffffffff)==8:
        rel[off]=add
        byadd.setdefault(add,[]).append(off)
def cstr(va):
    f=v2f(va)
    if f is None: return None
    e=d.find(b'\0',f)
    if e<0 or e-f>200: return None
    s=d[f:e].decode('utf8','replace')
    return s if s and s.isprintable() else None
def rtti_for_slot(slot):
    for k in range(0,80):
        vt=slot-k*8
        ti=rel.get(vt-8)
        if ti is None: continue
        nm=cstr(rel.get(ti+8,0)) if (ti+8) in rel else None
        if nm and (nm[0].isalpha() or nm[0] in '_N1'):
            return vt, k, nm
    return None
for fn in (int(x,0) for x in sys.argv[1:]):
    for slot in byadd.get(fn,[]):
        r=rtti_for_slot(slot)
        print(f"fn {fn:#x} slot@{slot:#x} -> " + (f"class '{r[2]}' (vtable {r[0]:#x}, slot #{r[1]})" if r else "no RTTI found"))
