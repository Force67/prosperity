#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/strings/xstring.h"

namespace gpu::gcn::spirv {

// The recompiler's own SPIR-V optimizer, shaped around what the emitter
// produces: a Private/Function register file accessed through constant access
// chains. Level 1 promotes it to SSA and drops dead code; level 2 also folds,
// value-numbers and hoists expressions repeated in sibling branches. Input it
// does not model comes back unchanged. With `err`, also runs CheckModule's
// checks: false means the module failed them.
bool OptimizeModule(const base::Vector<u32>& in,
                    base::Vector<u32>* out,
                    u32 level,
                    base::String* err = nullptr);

// Structural checks only: layout, unique result ids, every id operand defined,
// block shape. Catches emitter slips, not type errors.
bool CheckModule(const base::Vector<u32>& in, base::String* err);

}  // namespace gpu::gcn::spirv
