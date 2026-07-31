#pragma once

/*
 * The console's system settings, as sceSystemServiceParamGetInt reports them.
 *
 * There is no console to ask. Answering matters most for the system LANGUAGE:
 * SCE_SYSTEM_PARAM_LANG_JAPANESE is 0, which is also what a title reads when the
 * call fails, so a title comes up in Japanese unless we answer.
 *
 * PS4 and PS5 keep separate HLE registries (a PS5 import never resolves to a PS4
 * stub), so both copies of the module include this rather than sharing a symbol.
 */

#include <cstdint>
#include <cstdlib>
#include <strings.h>

namespace runtime::sysparam {

// SCE_SYSTEM_SERVICE_PARAM_ID_*.
constexpr int32_t kIdLang = 1;
constexpr int32_t kIdDateFormat = 2;
constexpr int32_t kIdTimeFormat = 3;
constexpr int32_t kIdTimeZone = 4;
constexpr int32_t kIdSummerTime = 5;
constexpr int32_t kIdParentalLevel = 7;
constexpr int32_t kIdEnterButtonAssign = 1000;

// SCE_SYSTEM_PARAM_LANG_*, in the order the console numbers them.
struct LangName {
  const char *name;
  int32_t id;
};
inline constexpr LangName kLangs[] = {
    {"ja", 0},       {"en", 1},      {"en-us", 1},  {"fr", 2},
    {"es", 3},       {"de", 4},      {"it", 5},     {"nl", 6},
    {"pt", 7},       {"ru", 8},      {"ko", 9},     {"zh-hant", 10},
    {"zh-hans", 11}, {"fi", 12},     {"sv", 13},    {"da", 14},
    {"no", 15},      {"pl", 16},     {"pt-br", 17}, {"en-gb", 18},
    {"tr", 19},      {"es-419", 20}, {"ar", 21},    {"fr-ca", 22},
    {"cs", 23},      {"hu", 24},     {"el", 25},    {"ro", 26},
    {"th", 27},      {"vi", 28},     {"id", 29},
};

// DELTA_SYS_LANG: the system language a title sees, as a code from the table
// above ("en", "ja", "de", "pt-br", ...) or a raw SCE_SYSTEM_PARAM_LANG_ number.
// Defaults to English (US), not the console's 0 (Japanese).
inline int32_t Language() {
  static const int32_t lang = [] {
    const char *e = std::getenv("DELTA_SYS_LANG");
    if (!e || !*e)
      return 1;
    if (e[0] >= '0' && e[0] <= '9')
      return static_cast<int32_t>(std::atoi(e));
    for (const auto &l : kLangs)
      if (strcasecmp(e, l.name) == 0)
        return l.id;
    return 1;
  }();
  return lang;
}

inline int ParamGetInt(int32_t paramId, int32_t *value) {
  if (!value)
    return -1;
  switch (paramId) {
  case kIdLang:
    *value = Language();
    break;
  case kIdTimeFormat:
    *value = 1; // 24-hour
    break;
  case kIdParentalLevel:
    *value = 1; // least restrictive
    break;
  case kIdEnterButtonAssign:
    // 0 = circle confirms (the Japanese default), 1 = cross. Follow the
    // language so the prompt glyphs match what the pad actually does.
    *value = Language() == 0 ? 0 : 1;
    break;
  case kIdDateFormat:
  case kIdTimeZone:
  case kIdSummerTime:
  default:
    *value = 0;
    break;
  }
  return 0;
}

} // namespace runtime::sysparam
