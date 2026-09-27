/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>

#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"

#include "base/strings/string_ref.h"
#include "logger/logger.h"

namespace logger {

class FileSink final : public LogSink {
  FILE* handle_{nullptr};
  size_t bytes_written_{0};

 public:
  explicit FileSink(const base::String& filename) {
    handle_ = std::fopen(filename.c_str(), "w");
  }

  void Close() {
    if (handle_) {
      std::fclose(handle_);
      handle_ = nullptr;
    }
  }

  const char* GetName() override { return "fileOut"; }

  void Write(const LogEntry& entry) override {
    constexpr mem_size kMaxBytesWritten = 50 * 1024L * 1024L;

    if (!handle_ || bytes_written_ > kMaxBytesWritten)
      return;

    auto msg = FormatLogEntry(entry);
    msg.push_back('\n');
    bytes_written_ += std::fwrite(static_cast<const void*>(msg.c_str()),
                                  msg.length(), 1, handle_);

    if (entry.log_level >= LogLevel::kError) {
      std::fflush(handle_);
    }
  }
};

class ConsoleSink final : public LogSink {
 public:
  const char* GetName() override { return "conOut"; }

  void Write(const LogEntry& entry) override {
    const char* color = "";
    const char* reset = "\x1b[0m";
    switch (entry.log_level) {
      case LogLevel::kTrace:
        color = "\x1b[90m";
        break;
      case LogLevel::kDebug:
        color = "\x1b[36m";
        break;
      case LogLevel::kInfo:
        color = "\x1b[37m";
        break;
      case LogLevel::kWarning:
        color = "\x1b[93m";
        break;
      case LogLevel::kError:
        color = "\x1b[91m";
        break;
      case LogLevel::kCritical:
        color = "\x1b[95m";
        break;
      default:
        break;
    }
    auto str = FormatLogEntry(entry);
    std::fprintf(stderr, "%s%s%s\n", color, str.c_str(), reset);
  }
};

void CreateLogger(bool create_console) {
  if (create_console) {
    AddLogSink(base::MakeUnique<ConsoleSink>());
  }

  base::String log_path(FXNAME);
  log_path.append(".log");
  AddLogSink(base::MakeUnique<FileSink>(log_path));

  std::atexit([]() {
    auto* sink = static_cast<FileSink*>(GetLogSink(base::StringRef("fileOut")));
    if (sink)
      sink->Close();
  });
}

}  // namespace logger
