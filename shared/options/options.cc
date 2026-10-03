/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "options/options.h"
#include "options/settings.h"

#include <cstring>
#include "base/containers/vector.h"

#include "base/option_file.h"
#include "base/strings/xstring.h"

#include "io/path.h"
#include "logger/logger.h"

namespace options {
namespace {

DELTA_OPTION(const char*,
             kOptionFiles,
             "DELTA_OPTIONS",
             nullptr,
             "options files to apply at startup, comma separated");
DELTA_OPTION(const char*,
             kOptionDump,
             "DELTA_OPT_DUMP",
             nullptr,
             "write every option and its value to this file at startup");
DELTA_OPTION(const char*,
             kProfile,
             "DELTA_PROFILE",
             nullptr,
             "game profile to apply: a file, or off for none");

void Report(const char* path, const base::OptionFileResult& result) {
  LOG_INFO("options: {} set {} option(s)", path, result.applied);
  if (result.skipped)
    LOG_INFO("options: {} left {} option(s) to what was already set", path,
             result.skipped);
  if (result.unknown)
    LOG_WARNING("options: {} names {} unknown option(s)", path, result.unknown);
  if (result.invalid)
    LOG_WARNING("options: {} has {} unusable entries", path, result.invalid);
}

void LoadFileList(const char* list) {
  base::String path;
  for (const char* p = list;; ++p) {
    if (*p && *p != ',') {
      path.push_back(*p);
      continue;
    }
    if (!path.empty())
      LoadFile(path.c_str());
    path.clear();
    if (!*p)
      return;
  }
}

// Matches "--flag" or "--flag=value", handing back the value ("" when the
// argument carries none).
bool MatchFlag(const char* arg, const char* flag, const char** value) {
  const char* a = arg;
  for (const char* f = flag; *f; ++f, ++a)
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

}  // namespace

bool LoadFile(const char* path, bool optional) {
  const auto result = base::ApplyOptionFile(base::Path(path));
  if (!result.read) {
    if (!optional)
      LOG_WARNING("options: cannot read {}", path);
    return false;
  }
  Report(path, result);
  return true;
}

void LoadGameProfile(const char* title_id) {
  const char* want = kProfile;
  if (want && (!std::strcmp(want, "off") || !std::strcmp(want, "0"))) {
    LOG_INFO("options: profile disabled");
    return;
  }

  base::String path;
  if (want && *want) {
    path = base::String(want);
  } else {
    if (!title_id || !*title_id)
      return;
    base::String rel("game_profiles/");
    rel += title_id;
    rel += ".txt";
    path = io::MakeAbsPath(rel);
  }

  const auto result = base::ApplyOptionFile(base::Path(path.c_str()),
                                            base::OptionApply::kFillUnset);
  if (!result.read) {
    LOG_INFO("options: no profile at {}", path.c_str());
    return;
  }
  Report(path.c_str(), result);
}

void Init() {
  const auto settings = SettingsPath();
  if (!settings.empty())
    LoadFile(settings.c_str(), true);
  base::InitOptionsFromEnv();

  if (const char* list = kOptionFiles)
    LoadFileList(list);
}

void Init(int& argc, char** argv) {
  Init();

  int kept = 1;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    const char* value = nullptr;

    if (MatchFlag(arg, "--options", &value)) {
      if (*value)
        LoadFileList(value);
      else
        LOG_WARNING("options: --options needs a path (--options=delta.txt)");
      continue;
    }
    if (MatchFlag(arg, "--dump-options", &value)) {
      kOptionDump.set(*value ? value : "-");
      continue;
    }
    // '+Name=Value', the same entry an options file holds.
    if (arg[0] == '+') {
      Report("command line", base::ApplyOptionText(arg));
      continue;
    }

    argv[kept++] = argv[i];
  }
  argc = kept;

  if (const char* dump = kOptionDump) {
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

namespace {
struct SavedOption {
  base::OptionBase* option;
  base::String value;
};
base::Vector<SavedOption> g_saved_options;
}  // namespace

void BeginGameSession() {
  g_saved_options.clear();
  base::OptionBase::VisitAll([](const base::OptionBase* option) {
    if (!option->overridden())
      return;
    base::Vector<char> buffer(256);
    auto length = option->FormatValue(buffer.data(), buffer.size());
    if (length >= buffer.size()) {
      buffer.resize(length + 1);
      length = option->FormatValue(buffer.data(), buffer.size());
    }
    g_saved_options.push_back({const_cast<base::OptionBase*>(option),
                               base::String(buffer.data(), length)});
  });
}

void EndGameSession() {
  base::OptionBase::VisitAll([](const base::OptionBase* option) {
    const_cast<base::OptionBase*>(option)->Reset();
  });
  for (const auto& saved : g_saved_options)
    saved.option->SetFromString(saved.value.c_str());
  g_saved_options = {};
}

}  // namespace options
