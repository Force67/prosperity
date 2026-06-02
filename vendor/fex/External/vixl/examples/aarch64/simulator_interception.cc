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

enum Result { FAILURE, SUCCESS };

// This will be called via a runtime call.
extern "C" int example_1() { return SUCCESS; }

// This will never be called, instead it will be intercepted and 'callback'
// will be called.
uint32_t example_2() { return FAILURE; }

uint32_t example_3(uint32_t num, float f) {
  USE(f);
  return num;
}

// This will be called instead of example_2.
uint32_t callback(uint64_t original_target) {
  USE(original_target);
  return SUCCESS;
}

void GenerateInterceptionExamples(MacroAssembler* masm) {
  // Preserve lr, since the calls will overwrite it.
  __ Push(xzr, lr);

  // example_1 will be intercepted and called through a runtime call.
  __ Mov(x16, reinterpret_cast<uint64_t>(example_1));
  __ Blr(x16);
  __ Mov(w1, w0);

  // example_2 will be intercepted and callback will be called instead.
  __ Mov(x16, reinterpret_cast<uint64_t>(example_2));
  __ Blr(x16);
  __ Mov(w2, w0);

  // Pass FAILURE as a parameter.
  __ Mov(x0, FAILURE);
  __ Fmov(s0, 3.5);
  // example_3 will be intercepted and lambda callback will be called instead.
  __ Mov(x16, reinterpret_cast<uint64_t>(example_3));
  __ Blr(x16);
  __ Mov(w3, w0);

  // Restore lr and return.
  __ Pop(lr, xzr);
  __ Ret();
}

#ifndef TEST_EXAMPLES
#ifdef VIXL_INCLUDE_SIMULATOR_AARCH64

int main(void) {
  MacroAssembler masm;

  // Generate the code for the example function.
  Label call_simulator_interception;
  masm.Bind(&call_simulator_interception);
  GenerateInterceptionExamples(&masm);
  masm.FinalizeCode();

  Instruction* start =
      masm.GetLabelAddress<Instruction*>(&call_simulator_interception);

  // Disassemble the generated code.
  PrintDisassembler disassembler(stdout);
  disassembler.DisassembleBuffer(start, masm.GetSizeOfCodeGenerated());

  Decoder decoder;
  Simulator simulator(&decoder);

  // Register interceptions to the branches, example_1 will be called via a
  // runtime call and callback will be called instead of example_2.
  simulator.RegisterBranchInterception(example_1);
  simulator.RegisterBranchInterception(example_2, callback);

  // Lambda callbacks can be used to arbitrarily modify the simulator.
  simulator.RegisterBranchInterception(
      example_3, [&simulator](uint64_t original_target) {
        USE(original_target);
        ABI abi;

        uint32_t param1 = simulator.ReadGenericOperand<uint32_t>(
            abi.GetNextParameterGenericOperand<uint32_t>());
        float param2 = simulator.ReadGenericOperand<float>(
            abi.GetNextParameterGenericOperand<float>());

        if (param1 == FAILURE && param2 == 3.5) {
          simulator.WriteWRegister(0, SUCCESS);
        } else {
          simulator.WriteWRegister(0, FAILURE);
        }
      });

  simulator.RunFrom(start);

  uint32_t result_1 = simulator.ReadWRegister(1);
  if (result_1 == SUCCESS) {
    printf("SUCCESS: example_1 was called via a runtime call.\n");
  } else {
    printf("ERROR: example_1 was not called.\n");
  }

  uint32_t result_2 = simulator.ReadWRegister(2);
  if (result_2 == SUCCESS) {
    printf("SUCCESS: callback was called instead of example_2.\n");
  } else {
    printf("ERROR: example_2 was called incorrectly.\n");
  }

  uint32_t result_3 = simulator.ReadWRegister(0);
  if (result_3 == SUCCESS) {
    printf("SUCCESS: Lambda callback called instead of example_3.\n");
  } else {
    printf("ERROR: example_3 was called instead of the lambda.\n");
  }

  return 0;
}
#else
// TODO: Support running natively.
int main(void) { return 0; }
#endif  // VIXL_INCLUDE_SIMULATOR_AARCH64
#endif  // TEST_EXAMPLES
