// Copyright 2014, VIXL authors
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

#include "executable-memory.h"

#include "aarch64/macro-assembler-aarch64.h"
#include "aarch64/simulator-aarch64.h"

using namespace vixl;
using namespace vixl::aarch64;

#define __ masm->

using namespace vixl::aarch64;

void GenerateDemoFunction(MacroAssembler *masm) {
  // uint64_t demo_function(uint64_t x)
  __ Ldr(x1, 0x1122334455667788);
  __ And(x0, x0, x1);
  __ Ret();
}


#ifndef TEST_EXAMPLES
int main() {
  MacroAssembler masm;

  Label demo;
  masm.Bind(&demo);
  GenerateDemoFunction(&masm);
  masm.FinalizeCode();

#ifdef VIXL_INCLUDE_SIMULATOR_AARCH64
  Decoder decoder;
  Simulator simulator(&decoder);

  simulator.WriteXRegister(0, 0x8899aabbccddeeff);
  simulator.RunFrom(masm.GetLabelAddress<Instruction *>(&demo));
  printf("x0 = 0x%" PRIx64 "\n", simulator.ReadXRegister(0));

#else
  byte* code = masm.GetBuffer()->GetStartAddress<byte*>();
  size_t code_size = masm.GetSizeOfCodeGenerated();
  ExecutableMemory memory(code, code_size);
  // Run the example function.
  uint64_t (*demo_function)(uint64_t) =
      memory.GetEntryPoint<uint64_t (*)(uint64_t)>(demo);
  uint64_t input_value = 0x8899aabbccddeeff;
  uint64_t output_value = (*demo_function)(input_value);
  printf("native: demo(0x%" PRIx64 ") = 0x%" PRIx64 "\n",
         input_value,
         output_value);
#endif  // VIXL_INCLUDE_SIMULATOR_AARCH64

  return 0;
}
#endif  // TEST_EXAMPLES
