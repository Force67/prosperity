#include <gtest/gtest.h>
#include "base/arch.h"

#include "runtime/vprx/vprx.h"

// The seven sanitizer fiber hooks libSceFiber imports from the TSan/ASan debug
// modules retail firmware doesn't ship (see vprx/ps5/libSceFiber_ps5.cpp).
namespace {
constexpr u64 kSanitizerHookNids[] = {
    0x52428EF1EB1400CE,  // __tsan_create_fiber
    0xA8F98DBB1D62524A,  // __tsan_destroy_fiber
    0xB5D9DC2DEBB6D28C,  // __tsan_get_current_fiber
    0x3394726352F54432,  // __tsan_switch_to_fiber
    0x00CE5D6D7A77A9B2,  // __sanitizer_start_switch_fiber
    0x718958B03418E74D,  // __sanitizer_finish_switch_fiber
    0x8DA721ADB8E91E73,  // __asan_destroy_fake_stack
};
}

TEST(FiberShim, SanitizerHooksRegistered) {
  runtime::vprx_init();
  for (u64 nid : kSanitizerHookNids)
    EXPECT_NE(runtime::vprx_get_forced("libSceFiber", nid), 0u) << std::hex
                                                                << nid;
}

TEST(FiberShim, StartSwitchNullsFakeStackSave) {
  runtime::vprx_init();
  auto fn = reinterpret_cast<void(PS4ABI *)(void **, const void *, size_t)>(
      runtime::vprx_get_forced("libSceFiber", 0x00CE5D6D7A77A9B2));
  ASSERT_NE(fn, nullptr);
  void *save = reinterpret_cast<void *>(0xdeadbeef);
  fn(&save, nullptr, 0);
  EXPECT_EQ(save, nullptr);
  fn(nullptr, nullptr, 0);  // must tolerate a null out-pointer
}

TEST(FiberShim, FinishSwitchZeroesOutputs) {
  runtime::vprx_init();
  auto fn = reinterpret_cast<void(PS4ABI *)(void *, const void **, size_t *)>(
      runtime::vprx_get_forced("libSceFiber", 0x718958B03418E74D));
  ASSERT_NE(fn, nullptr);
  const void *bottom = &bottom;
  size_t size = 42;
  fn(nullptr, &bottom, &size);
  EXPECT_EQ(bottom, nullptr);
  EXPECT_EQ(size, 0u);
  fn(nullptr, nullptr, nullptr);
}
