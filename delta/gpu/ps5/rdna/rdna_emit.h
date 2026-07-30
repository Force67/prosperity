#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Internal seam between the two RDNA2 translator TUs: rdna_translate.cc owns
 * the per-instruction dispatch and the control-flow lowering, rdna_compute.cc
 * the compute stage. Both directions are needed (compute drives the CFG, the
 * CFG dispatches compute's memory instructions), so the declarations live here
 * rather than in either file.
 *
 * Exposes the gpu/gcn SPIR-V backend's internal types, so only those two TUs
 * include it, and only inside their DELTA_HAVE_SPIRV_BACKEND branch.
 */

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/spirv/translator.h"

namespace gpu::rdna {

// rdna_translate.cc: lower the program's branches into the while/switch state
// machine, emitting each instruction through the RDNA2 dispatch.
void EmitCfg(gpu::gcn::Translator& t,
             const gpu::gcn::Program& program,
             gpu::gcn::StageContext& sc);

// rdna_compute.cc: emit a memory instruction against the compute resource
// model (set-0 storage buffers). Returns false for encodings that have no
// compute-specific form, which the caller then emits normally.
bool EmitCsMemory(gpu::gcn::Translator& t,
                  const gpu::gcn::Inst& inst,
                  gpu::gcn::StageContext& sc);

}  // namespace gpu::rdna
