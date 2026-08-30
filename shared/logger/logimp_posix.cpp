/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>

#include "logger.h"

namespace utl {

class fileOut final : public logBase {
  FILE* handle{nullptr};
  size_t bytes_written{0};

 public:
  explicit fileOut(const base::String& filename) {
    handle = std::fopen(filename.c_str(), "w");
  }

  void close() {
    if (handle) {
      std::fclose(handle);
      handle = nullptr;
    }
  }

  const char* getName() override { return "fileOut"; }

  void write(const logEntry& entry) override {
    constexpr std::size_t MAX_BYTES_WRITTEN = 50 * 1024L * 1024L;

    if (!handle || bytes_written > MAX_BYTES_WRITTEN)
      return;

    auto msg = formatLogEntry(entry);
    msg.push_back('\n');
    bytes_written += std::fwrite(static_cast<const void*>(msg.c_str()),
                                 msg.length(), 1, handle);

    if (entry.log_level >= logLevel::Error) {
      std::fflush(handle);
    }
  }
};

class conOut_Posix final : public logBase {
 public:
  const char* getName() override { return "conOut"; }

  void write(const logEntry& entry) override {
    const char* color = "";
    const char* reset = "\x1b[0m";
    switch (entry.log_level) {
      case logLevel::Trace:    color = "\x1b[90m"; break;
      case logLevel::Debug:    color = "\x1b[36m"; break;
      case logLevel::Info:     color = "\x1b[37m"; break;
      case logLevel::Warning:  color = "\x1b[93m"; break;
      case logLevel::Error:    color = "\x1b[91m"; break;
      case logLevel::Critical: color = "\x1b[95m"; break;
      default: break;
    }
    auto str = formatLogEntry(entry);
    std::fprintf(stderr, "%s%s%s\n", color, str.c_str(), reset);
  }
};

void createLogger(bool createConsole) {
  if (createConsole) {
    addLogSink(base::MakeUnique<conOut_Posix>());
  }

  base::String log_path(FXNAME);
  log_path.append(".log");
  addLogSink(base::MakeUnique<fileOut>(log_path));

  std::atexit([]() {
    auto* sink = static_cast<fileOut*>(getLogSink(base::StringRef("fileOut")));
    if (sink)
      sink->close();
  });
}

}  // namespace utl
