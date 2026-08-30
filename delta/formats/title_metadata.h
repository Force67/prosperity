/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The two files a title uses to describe itself, and nothing else: PS4's
// binary param.sfo and PS5's param.json. Pure readers over a buffer -- they
// open nothing and know nothing about the container the buffer came from, so
// the pkg, ffpkg and archive paths can all ask the same questions.

#include "base/arch.h"
#include <cstddef>
#include <string>

namespace formats {

// String value of `key` (e.g. "TITLE_ID") from a param.sfo image, or "" if
// absent. The SFO is a small flat table; every offset is bounds-checked.
std::string sfoGet(const u8 *data, size_t size, const char *key);

// u32 value of `key` from a param.sfo image, or 0 if absent.
u32 sfoGetU32(const u8 *data, size_t size, const char *key);

// One top-level string value out of a param.json (flat file, no nesting on the
// keys a title uses).
std::string jsonGetString(const std::string &json, const char *key);

// The display name, taken from localizedParameters.<defaultLanguage>.titleName
// so a title shipping several languages does not get whichever comes first.
std::string jsonGetTitleName(const std::string &json);

// param.json stores sdkVersion as "0xMMmmpppp00000000"; libkernel wants the top
// half (0x03000000 for a 3.00 title). Empty or unparsable -> 0.
u32 parseSdkVersion(const std::string &s);

}  // namespace formats
