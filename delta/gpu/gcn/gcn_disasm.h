#pragma once

#include "base/arch.h"

#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

base::String Mnemonic(const Inst& inst);

base::String DisasmInst(const Inst& inst);

base::String DisasmLine(const Inst& inst);

void Disassemble(const u32* code, u32 max_dwords, const char* tag);

}  // namespace gpu::gcn
