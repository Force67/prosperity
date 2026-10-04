#include "kern/system_settings.h"

#include <strings.h>
#include <cstring>

#include "base/containers/hash_map.h"
#include "base/strings/number_parse.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "options/options.h"

namespace kern {
namespace {
DELTA_OPTION(const char*,
             kSysLang,
             "DELTA_SYS_LANG",
             nullptr,
             "system language a title sees, as a code or a raw id");

constexpr u64 kLanguage = 0x12356328ecf5617b;
constexpr u64 kEnterButton = 0x22666251fe7becff;
struct LanguageName {
  const char* name;
  i32 id;
};
constexpr LanguageName kLanguages[] = {
    {"ja", 0},  {"en", 1},       {"en-us", 1},    {"fr", 2},     {"es", 3},
    {"de", 4},  {"it", 5},       {"nl", 6},       {"pt", 7},     {"ru", 8},
    {"ko", 9},  {"zh-hant", 10}, {"zh-hans", 11}, {"fi", 12},    {"sv", 13},
    {"da", 14}, {"no", 15},      {"pl", 16},      {"pt-br", 17}, {"en-gb", 18},
    {"tr", 19}, {"es-419", 20},  {"ar", 21},      {"fr-ca", 22}, {"cs", 23},
    {"hu", 24}, {"el", 25},      {"ro", 26},      {"th", 27},    {"vi", 28},
    {"id", 29},
};
struct Setting {
  u64 key;
  i32 value;
};
// Encoded keys used by the PS4 11.00 and PS5 01.14.00 libraries.
constexpr Setting kDefaults[] = {
    {kLanguage, 1},          {0x596b60e8fda47e4c, 0},  // date format
    {0x1bd16a3e9b03ed9a, 1},                           // time format
    {0x123d6328eff5667b, 0},                           // time zone
    {0x3a73688d92700623, 0},  // time zone offset, seconds
    {0x03d261027dfff0a4, 0},  // daylight saving
    {kEnterButton, 1},       {0x0cae671adf3aeb34, 0}, {0x338660835bde7cb1, 0},
};
base::Mutex g_mutex;
base::HashMap<u64, i32> g_values;

i32 Language() {
  if (const auto it = g_values.find(kLanguage); it != g_values.end())
    return it->second;
  const char* name = kSysLang;
  if (!name || !*name)
    return 1;
  u64 id;
  const char* end = nullptr;
  if (base::ParseUnsigned(name, id, 10, &end) && !*end && id <= 29)
    return id;
  for (const auto& language : kLanguages)
    if (strcasecmp(name, language.name) == 0)
      return language.id;
  return 1;
}

const guest::SessionReset g_reset([] {
  base::LockGuard lock(g_mutex);
  g_values.clear();
});
}  // namespace

bool GetRegistryInt(u64 key, i32& value) {
  base::LockGuard lock(g_mutex);
  for (const auto& setting : kDefaults) {
    if (setting.key != key)
      continue;
    if (const auto it = g_values.find(key); it != g_values.end())
      value = it->second;
    else if (key == kLanguage)
      value = Language();
    else if (key == kEnterButton)
      value = Language() != 0;
    else
      value = setting.value;
    return true;
  }
  return false;
}

bool SetRegistryInt(u64 key, i32 value) {
  base::LockGuard lock(g_mutex);
  for (const auto& setting : kDefaults) {
    if (setting.key == key) {
      g_values[key] = value;
      return true;
    }
  }
  return false;
}

void InitializeShellCoreSharedMemory(u8* memory, size_t size) {
  if (!memory || size < 0x18)
    return;
  const u64 magic = 0x6ed81ede6df17259;
  const i32 parental_level = 1;
  std::memcpy(memory, &magic, sizeof(magic));
  std::memcpy(memory + 0x14, &parental_level, sizeof(parental_level));
}
}  // namespace kern
