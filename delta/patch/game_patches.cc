/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "patch/game_patches.h"

#include "base/containers/vector.h"
#include "base/strings/format.h"
#include "base/strings/string_ref.h"
#include "crypto/sha1.h"
#include "io/file.h"
#include "logger/logger.h"
#include "options/options.h"
#include "patch/patch_file.h"

namespace patch {
namespace {

DELTA_OPTION(const char*,
             kPatchFile,
             "DELTA_PATCH_FILE",
             nullptr,
             "game patch file (YAML)");
DELTA_OPTION(const char*,
             kPatches,
             "DELTA_PATCHES",
             nullptr,
             "';'-separated names of the patches to apply from "
             "DELTA_PATCH_FILE");

using Ref = base::StringRef;

bool Selected(const base::String& name) {
  if (!kPatches)
    return false;
  Ref list(kPatches);
  while (true) {
    const mem_size sep = list.find(';');
    if (list.substr(0, sep) == Ref(name.c_str()))
      return true;
    if (sep == Ref::npos)
      return false;
    list = list.substr(sep + 1);
  }
}

}  // namespace

base::String ImageKey(const Segment* segments, mem_size count) {
  sha1_context ctx;
  Sha1Starts(&ctx);
  for (mem_size i = 0; i < count; i++)
    Sha1Update(&ctx, segments[i].data, segments[i].size);
  u8 digest[20];
  Sha1Finish(&ctx, digest);
  base::String key("EBOOT-");
  for (u8 b : digest)
    base::FormatTo(key, "{:02x}", b);
  return key;
}

void PatchMainImage(uintptr_t base,
                    u64 end,
                    const Segment* segments,
                    mem_size count) {
  if (!kPatchFile)
    return;
  const base::String key = ImageKey(segments, count);
  LOG_INFO("patches: main image is {}", key.c_str());

  io::File file(base::String(kPatchFile), io::FileMode::kRead);
  if (!file.IsOpen()) {
    LOG_ERROR("patches: cannot open {}", kPatchFile);
    return;
  }
  base::String text;
  text.resize(static_cast<mem_size>(file.GetSize()));
  file.Read(text.data(), text.size());

  base::Vector<Patch> patches;
  base::String error;
  if (!ParsePatchFile(Ref(text.data(), text.size()), &patches, &error)) {
    LOG_ERROR("patches: {}: {}", kPatchFile, error.c_str());
    return;
  }
  for (const Patch& p : patches) {
    if (p.image != key || !Selected(p.name))
      continue;
    if (ApplyPatch(p, base, end))
      LOG_INFO("patches: applied \"{}\" ({} writes)", p.name.c_str(),
               p.writes.size());
    else
      LOG_ERROR("patches: \"{}\" writes outside the image, skipped",
                p.name.c_str());
  }
}

}  // namespace patch
