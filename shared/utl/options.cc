/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "options.h"

#include <cstring>

#include <base/option_file.h>
#include <base/strings/xstring.h>

#include <logger/logger.h>
#include <utl/path.h>

namespace utl {
namespace {

DELTA_OPTION(const char *, kOptionFiles, "DELTA_OPTIONS", nullptr,
             "options files to apply at startup, comma separated");
DELTA_OPTION(const char *, kOptionDump, "DELTA_OPT_DUMP", nullptr,
             "write every option and its value to this file at startup");
DELTA_OPTION(const char *, kProfile, "DELTA_PROFILE", nullptr,
             "game profile to apply: a file, or off for none");

void report(const char *path, const base::OptionFileResult &result) {
  LOG_INFO("options: {} set {} option(s)", path, result.applied);
  if (result.skipped)
    LOG_INFO("options: {} left {} option(s) to what was already set", path,
             result.skipped);
  if (result.unknown)
    LOG_WARNING("options: {} names {} unknown option(s)", path, result.unknown);
  if (result.invalid)
    LOG_WARNING("options: {} has {} unusable entries", path, result.invalid);
}

void loadOptionFileList(const char *list) {
  base::String path;
  for (const char *p = list;; ++p) {
    if (*p && *p != ',') {
      path.push_back(*p);
      continue;
    }
    if (!path.empty())
      loadOptionFile(path.c_str());
    path.clear();
    if (!*p)
      return;
  }
}

// Matches "--flag" or "--flag=value", handing back the value ("" when the
// argument carries none).
bool matchFlag(const char *arg, const char *flag, const char **value) {
  const char *a = arg;
  for (const char *f = flag; *f; ++f, ++a)
    if (*a != *f)
      return false;
  if (*a == '\0') {
    *value = "";
    return true;
  }
  if (*a != '=')
    return false;
  *value = a + 1;
  return true;
}

} // namespace

bool loadOptionFile(const char *path, bool optional) {
  const auto result = base::ApplyOptionFile(base::Path(path));
  if (!result.read) {
    if (!optional)
      LOG_WARNING("options: cannot read {}", path);
    return false;
  }
  report(path, result);
  return true;
}

void loadGameProfile(const char *titleId) {
  const char *want = kProfile;
  if (want && (!std::strcmp(want, "off") || !std::strcmp(want, "0"))) {
    LOG_INFO("options: profile disabled");
    return;
  }

  base::String path;
  if (want && *want) {
    path = base::String(want);
  } else {
    if (!titleId || !*titleId)
      return;
    base::String rel("game_profiles/");
    rel += titleId;
    rel += ".txt";
    path = make_abs_path(rel);
  }

  const auto result =
      base::ApplyOptionFile(base::Path(path.c_str()),
                            base::OptionApply::kFillUnset);
  if (!result.read) {
    LOG_INFO("options: no profile at {}", path.c_str());
    return;
  }
  report(path.c_str(), result);
}

void initOptions() {
  base::InitOptionsFromEnv();

  if (const char *list = kOptionFiles)
    loadOptionFileList(list);
}

void initOptions(int &argc, char **argv) {
  initOptions();

  int kept = 1;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    const char *value = nullptr;

    if (matchFlag(arg, "--options", &value)) {
      if (*value)
        loadOptionFileList(value);
      else
        LOG_WARNING("options: --options needs a path (--options=delta.txt)");
      continue;
    }
    if (matchFlag(arg, "--dump-options", &value)) {
      kOptionDump.set(*value ? value : "-");
      continue;
    }
    // '+Name=Value', the same entry an options file holds.
    if (arg[0] == '+') {
      report("command line", base::ApplyOptionText(arg));
      continue;
    }

    argv[kept++] = argv[i];
  }
  argc = kept;

  if (const char *dump = kOptionDump) {
    base::String text;
    base::AppendOptionText(text);
    if (dump[0] == '-' && dump[1] == '\0')
      LOG_INFO("options:\n{}", text.c_str());
    else if (base::WriteOptionFile(base::Path(dump)))
      LOG_INFO("options: wrote {}", dump);
    else
      LOG_WARNING("options: cannot write {}", dump);
  }
}

} // namespace utl
