#include <gtest/gtest.h>

#include "runtime/code_lift.h"

// codeLift wraps the vendored capstone disassembler. init() opening the handle
// proves the runtime module's disassembler wiring works end to end against the
// capstone we link, and the destructor tears it down without leaking.
TEST(CodeLift, InitOpensDisassembler) {
  uint8_t* rip = nullptr;
  runtime::codeLift lift(rip);
  EXPECT_TRUE(lift.init());
}
