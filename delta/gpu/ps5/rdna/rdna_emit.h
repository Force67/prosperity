#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/spirv/translator.h"

namespace gpu::rdna {

void EmitBvh(gpu::gcn::Translator& t,
             const gpu::gcn::Inst& inst,
             gpu::gcn::StageContext& sc);

void EmitCfg(gpu::gcn::Translator& t,
             const gpu::gcn::Program& program,
             gpu::gcn::StageContext& sc);

u64 RdnaTexDimMask(const gpu::gcn::Program& program,
                   const gpu::gcn::MimgBindingPlan& plan,
                   u32 dimensions);

void SetComputeAddress(u64 address);

u32 LdsAccess(const gpu::gcn::Inst& inst);

bool EmitCsMemory(gpu::gcn::Translator& t,
                  const gpu::gcn::Inst& inst,
                  gpu::gcn::StageContext& sc);

}  // namespace gpu::rdna
