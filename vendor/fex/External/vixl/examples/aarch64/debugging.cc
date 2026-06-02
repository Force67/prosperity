// Copyright 2023, VIXL authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//   * Neither the name of ARM Limited nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "examples.h"

#include "aarch64/disasm-aarch64.h"
#include "aarch64/macro-assembler-aarch64.h"
#include "aarch64/simulator-aarch64.h"

using namespace vixl;
using namespace vixl::aarch64;

#define __ masm->

void GenerateDebugExample(MacroAssembler* masm) {
  // Create a breakpoint here to break into the debugger.
  __ Brk(0);

  // Do some arithmetic.
  __ Add(x1, x0, 5);
  __ Mov(x2, 2);
  __ Sub(x3, x1, x2);

  __ Ret();
}

#ifndef TEST_EXAMPLES
#ifdef VIXL_INCLUDE_SIMULATOR_AARCH64

int main(void) {
  MacroAssembler masm;

  // Generate the code for the example function.
  Label debug_example;
  masm.Bind(&debug_example);
  GenerateDebugExample(&masm);
  masm.FinalizeCode();

  Instruction* start = masm.GetLabelAddress<Instruction*>(&debug_example);

  // Disassemble the generated code.
  PrintDisassembler disassembler(stdout);
  disassembler.DisassembleBuffer(start, masm.GetSizeOfCodeGenerated());

  Decoder decoder;
  Simulator simulator(&decoder);

  simulator.SetColouredTrace(true);
  simulator.SetDebuggerEnabled(true);

  int32_t input_a = 1;
  int32_t input_b = 2;
  simulator.WriteWRegister(0, input_a);
  simulator.WriteWRegister(1, input_b);
  simulator.RunFrom(start);
  printf("The final result is %ld\n", simulator.ReadXRegister(3));

  return 0;
}

#else
int main(void) { return 0; }
#endif  // VIXL_INCLUDE_SIMULATOR_AARCH64
#endif  // TEST_EXAMPLES
