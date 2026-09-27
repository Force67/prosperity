/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "formats/title_metadata.h"

#include <cstdlib>
#include <cstring>
#include "base/strings/xstring.h"

namespace formats {

// Minimal param.sfo reader: return the string value of `key` (e.g. "TITLE_ID"),
// or "" if absent. The SFO is a small flat table; all offsets are
// bounds-checked.
base::String SfoGet(const u8* d, size_t n, const char* key) {
  if (n < 20)
    return {};
  auto rd16 = [&](size_t o) -> u16 {
    return o + 2 <= n ? u16(d[o] | (d[o + 1] << 8)) : 0;
  };
  auto rd32 = [&](size_t o) -> u32 {
    return o + 4 <= n ? u32(d[o]) | (u32(d[o + 1]) << 8) |
                            (u32(d[o + 2]) << 16) | (u32(d[o + 3]) << 24)
                      : 0;
  };
  if (rd32(0) != 0x46535000u)  // "\0PSF"
    return {};
  u32 key_start = rd32(8), data_start = rd32(12), count = rd32(16);
  size_t klen = std::strlen(key);
  for (u32 i = 0, idx = 20; i < count; i++, idx += 16) {
    if (idx + 16 > n)
      break;
    size_t kpos = size_t(key_start) + rd16(idx);
    if (kpos + klen + 1 > n)
      continue;
    if (std::memcmp(d + kpos, key, klen) != 0 || d[kpos + klen] != '\0')
      continue;
    size_t dpos = size_t(data_start) + rd32(idx + 12);
    if (dpos >= n)
      return {};
    size_t avail = n - dpos, len = rd32(idx + 4);
    base::String s(reinterpret_cast<const char*>(d + dpos),
                   len < avail ? len : avail);
    while (!s.empty() && s.back() == '\0')
      s.pop_back();
    return s;
  }
  return {};
}

u32 SfoGetU32(const u8* d, size_t n, const char* key) {
  if (n < 20)
    return 0;
  auto rd16 = [&](size_t o) -> u16 {
    return o + 2 <= n ? u16(d[o] | (d[o + 1] << 8)) : 0;
  };
  auto rd32 = [&](size_t o) -> u32 {
    return o + 4 <= n ? u32(d[o]) | (u32(d[o + 1]) << 8) |
                            (u32(d[o + 2]) << 16) | (u32(d[o + 3]) << 24)
                      : 0;
  };
  if (rd32(0) != 0x46535000u)
    return 0;
  const u32 key_start = rd32(8), data_start = rd32(12), count = rd32(16);
  const size_t key_length = std::strlen(key);
  for (u32 i = 0, index = 20; i < count; i++, index += 16) {
    if (index + 16 > n)
      break;
    const size_t key_position = size_t(key_start) + rd16(index);
    if (key_position + key_length + 1 > n ||
        std::memcmp(d + key_position, key, key_length) != 0 ||
        d[key_position + key_length] != '\0')
      continue;
    const size_t data_position = size_t(data_start) + rd32(index + 12);
    return data_position + sizeof(u32) <= n ? rd32(data_position) : 0;
  }
  return 0;
}

// PS5 titles carry sce_sys/param.json instead of the PS4 param.sfo. Pull one
// top-level string value out of it (flat file, no nesting on the keys we want).
base::String JsonGetString(const base::String& js, const char* key) {
  base::String pat = base::String("\"") + key + "\"";
  size_t k = js.find(pat);
  if (k == base::String::npos)
    return {};
  size_t colon = js.find(':', k + pat.size());
  if (colon == base::String::npos)
    return {};
  size_t open = js.find('"', colon);
  size_t close = open == base::String::npos ? open : js.find('"', open + 1);
  if (close == base::String::npos)
    return {};
  return js.substr(open + 1, close - open - 1);
}

// param.json keeps the display name under localizedParameters.<defaultLanguage>
// .titleName. Search from the default language's block so a title shipping
// several languages doesn't pick whichever one happens to come first.
base::String JsonGetTitleName(const base::String& js) {
  const base::String lang = JsonGetString(js, "defaultLanguage");
  if (!lang.empty()) {
    const size_t block = js.find("\"" + lang + "\"");
    if (block != base::String::npos) {
      base::String name = JsonGetString(js.substr(block), "titleName");
      if (!name.empty())
        return name;
    }
  }
  return JsonGetString(js, "titleName");
}
// param.json stores sdkVersion as "0xMMmmpppp00000000"; libkernel wants the top
// half (0x03000000 for a 3.00 title). Empty/unparsable -> 0.
u32 ParseSdkVersion(const base::String& s) {
  if (s.empty())
    return 0;
  return static_cast<u32>(std::strtoull(s.c_str(), nullptr, 0) >> 32);
}

}  // namespace formats
