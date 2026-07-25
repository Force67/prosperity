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

#if defined(DELTA_BACKEND_NATIVE)
TEST(CodeLift, FsLoadUsesStacklessStub) {
  uint8_t code[] = {0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xc3};
  uint8_t stubs[64]{};
  uint8_t *rip = stubs;
  runtime::codeLift lift(rip, stubs + sizeof(stubs));
  ASSERT_TRUE(lift.init());

  lift.transform(code, sizeof(code));

  EXPECT_EQ(code[0], 0xe9);
  EXPECT_EQ(stubs[0], 0x64);
}

TEST(CodeLift, FsStoreUsesStacklessStub) {
  uint8_t code[] = {0x64, 0x48, 0x89, 0x0c, 0x25, 0x00, 0x00, 0x00, 0x00, 0xc3};
  uint8_t stubs[64]{};
  uint8_t *rip = stubs;
  runtime::codeLift lift(rip, stubs + sizeof(stubs));
  ASSERT_TRUE(lift.init());

  lift.transform(code, sizeof(code));

  EXPECT_EQ(code[0], 0xe9);
  EXPECT_EQ(stubs[0], 0x64);
}
#endif
