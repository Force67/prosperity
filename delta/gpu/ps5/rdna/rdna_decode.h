#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "base/memory/shared_pointer.h"
#include "gpu/gcn/gcn_decode.h"

namespace gpu::rdna {

using gpu::gcn::Enc;
using gpu::gcn::Inst;
using gpu::gcn::Program;

u32 Vop3SourceCount(Enc enc, u32 op);

Program Decode(const u32* code, u32 max_dwords, bool stop_at_endpgm = true);

u32 CodeLength(const u32* code, u32 max_dwords);

Program DecodeShader(const u32* code, u32 max_dwords);

Program ReachableProgram(const Program& program);

base::SharedPointer<const Program> CachedReachableProgram(const u32* code,
                                                          u32 max_dwords);

u64 CachedCodeHash(const u32* code, u32 max_dwords);

void NextProgramGeneration();

}  // namespace gpu::rdna
