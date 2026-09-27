#pragma once

#include "base/arch.h"

#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "base/time/time.h"

// log impl is heavily influenced & based on the yuzu logger

namespace logger {
enum class LogLevel : u8 {
  kTrace,
  kDebug,
  kInfo,
  kWarning,
  kError,
  kCritical,
  kCount
};

struct LogEntry {
  base::TimeDelta timestamp;  // since the logger started
  LogLevel log_level;
  unsigned int line_num;
  base::String function;
  base::String message;
  bool final_entry = false;

  LogEntry() = default;
  LogEntry(LogEntry&& o) = default;

  LogEntry& operator=(LogEntry&& o) = default;
  LogEntry& operator=(const LogEntry& o) = default;
};

class LogSink {
 public:
  virtual ~LogSink() = default;

  virtual const char* GetName() { return "Unknown"; }

  virtual void Write(const LogEntry&) = 0;
};

base::String FormatLogEntry(const LogEntry& entry);
LogSink* AddLogSink(base::UniquePointer<LogSink> sink);
LogSink* GetLogSink(base::StringRef name);
void AddLogMsg(LogLevel lvl, u32 line, const char* func, base::String msg);

void CreateLogger(bool with_console = false);

// Route base::PrintLogMessage (BASE_LOGI and friends) into this logger, so a
// module that logs on a base channel reaches the same sinks asynchronously
// instead of writing to stderr on the calling thread. Call once, after
// CreateLogger.
void RouteBaseLogging();

// Drop all further log entries. Called from the crash handler so the async log
// backend thread stops writing to stderr and corrupting the fault dump.
void SilenceLogging();

template <typename... Args>
inline void FmtLogMsg(LogLevel lvl,
                      u32 line,
                      const char* func,
                      const char* fmt,
                      const Args&... args) {
  AddLogMsg(lvl, line, func, base::Format(fmt, args...));
}

inline void FmtLogMsg(LogLevel lvl,
                      u32 line,
                      const char* func,
                      const base::String& text) {
  AddLogMsg(lvl, line, func, text);
}
}  // namespace logger

#ifdef _DEBUG
#define LOG_TRACE(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kTrace, __LINE__, __func__, \
                      __VA_ARGS__)
#else
#define LOG_TRACE(fmt, ...) (void(0))
#endif

#define LOG_DEBUG(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kDebug, __LINE__, __func__, \
                      __VA_ARGS__)
#define LOG_INFO(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kInfo, __LINE__, __func__, \
                      __VA_ARGS__)
#define LOG_WARNING(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kWarning, __LINE__, __func__, \
                      __VA_ARGS__)
#define LOG_ERROR(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kError, __LINE__, __func__, \
                      __VA_ARGS__)
#define LOG_CRITICAL(...)                                                \
  ::logger::FmtLogMsg(::logger::LogLevel::kCritical, __LINE__, __func__, \
                      __VA_ARGS__)
#define LOG_ASSERT(expression)                                            \
  do {                                                                    \
    if (!(expression)) {                                                  \
      ::logger::FmtLogMsg(::logger::LogLevel::kError, __LINE__, __func__, \
                          "assertion failed at " #expression);            \
      __builtin_trap();                                                   \
    }                                                                     \
                                                                          \
  } while (0)

#define LOG_UNIMPLEMENTED                                             \
  ::logger::FmtLogMsg(::logger::LogLevel::kError, __LINE__, __func__, \
                      "Unimplemented function")
