#include <array>

#include <gtest/gtest.h>

#include "runtime/nid.h"

TEST(Nid, EncodingPreservesLeadingZeroDigitsAndOutputBounds) {
  struct Vector {
    const char* name;
    const char* encoded;
  };
  for (const auto& value : {
           Vector{"sceNgs2VoiceRunCommands", "AbYvTOZ8Pts"},
           Vector{"nid-leading-zero-2318", "AAEzPdNykwk"},
           Vector{"sceNgs2SystemCreate", "koBbCMvOKWw"},
       }) {
    SCOPED_TRACE(value.name);
    std::array<u8, 14> output;
    output.fill(0xa5);
    runtime::nid::Encode(value.name, output.data() + 1);
    EXPECT_STREQ(reinterpret_cast<const char*>(output.data() + 1),
                 value.encoded);
    EXPECT_EQ(output[0], 0xa5);
    EXPECT_EQ(output[12], 0);
    EXPECT_EQ(output[13], 0xa5);
  }
}

TEST(Nid, NativeVoiceCommandsDecodeToExportId) {
  u64 id = 0;
  ASSERT_TRUE(runtime::nid::Decode("AbYvTOZ8Pts", 11, id));
  EXPECT_EQ(id, 0x01b62f4ce67c3edbull);
}
