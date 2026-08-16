#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * How a register write reaches the register file.
 *
 * The PS4 has one way: the value is in the packet. AGC has four, and three of
 * them put it in guest memory instead -- a LOAD_*_REG restores a register
 * shadow image, a SET_*_REG_INDIRECT names a block of (offset, value) entries,
 * a type-0 packet writes an absolute run. Which of those a title uses is a
 * property of the title and not of the state it is setting: Skyrim never issues
 * SET_CONTEXT_REG at all, and builds its context (render target included) into
 * a shadow with CP DMA.
 *
 * The state blocks are the interesting part, and are read from evidence rather
 * than from a documented encoding, because nothing in the packet says where in
 * the register file a type-1 block's entries land. A colour-target bind is
 * recognised by a base that points at a mapped surface whose CB_COLOR0_INFO and
 * ATTRIB2 agree with it; the blend object by the tag array that always ends
 * one. Both checks are conservative: a wrong reading binds nothing rather than
 * something wrong, and a draw that renders into nothing is a black target, not
 * a crash.
 */

#include "base/arch.h"

#include "gpu/ps5/agc_regs.h"

namespace gpu::ps5 {

// SET_*_REG: body[0] is the register offset (carrying the gfx10 selector bits),
// body[1..] the values.
void SetRegs(Regs& regs, u32 base, const u32* body, u32 count);

// A type-0 packet: `count` consecutive registers from an absolute offset. The
// AGC driver programs shader state this way, which is why no SET_SH_REG carries
// it.
void SetRegRun(Regs& regs, u32 first_reg, const u32* values, u32 count);

// op 0x93, an inline SH-register set: body[0]'s low 16 bits are the offset (the
// high bits are flags/count), body[1..] the values.
void SetShRegsInline(Regs& regs, const u32* body, u32 count);

// LOAD_*_REG: the values live in a register shadow image in GPU memory at
// body[0..1], and body[2..] are (offset, dword count) ranges into it.
void LoadRegImage(Regs& regs, u32 base, const u32* body, u32 count);

// SET_*_REG_INDIRECT: body = [addrLo, addrHi, mode, numPairs], and the GPU
// buffer at that address holds numPairs (offset, value) entries.
void LoadRegBlock(Regs& regs, u32 base, const u32* body, u32 count);

// IT_WRITE_DATA with DST_SEL=0, which targets the memory-mapped register file
// rather than memory: body[1] is a register offset, not an address.
void WriteDataRegs(Regs& regs, const u32* body, u32 count);

}  // namespace gpu::ps5
