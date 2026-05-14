#pragma once

#include <chrono>
#include <cstdint>

#include <base.h>

#include <base/strings/xstring.h>
#include <base/strings/string_ref.h>
#include <base/memory/unique_pointer.h>

// hard dependency
#include <fmt/format.h>

// log impl is heavily influenced & based on the yuzu logger

namespace utl {
enum class logLevel : uint8_t {
  Trace,
  Debug,
  Info,
  Warning,
  Error,
  Critical,
  Count
};

struct logEntry {
  std::chrono::microseconds timestamp;
  logLevel log_level;
  unsigned int line_num;
  base::String function;
  base::String message;
  bool final_entry = false;

  logEntry() = default;
  logEntry(logEntry &&o) = default;

  logEntry &operator=(logEntry &&o) = default;
  logEntry &operator=(const logEntry &o) = default;
};

class logBase {
public:
  virtual ~logBase() = default;

  virtual const char *getName() { return "Unknown"; }

  virtual void write(const logEntry &) = 0;
};

base::String formatLogEntry(const logEntry &entry);
logBase *addLogSink(base::UniquePointer<logBase> sink);
logBase *getLogSink(base::StringRef name);
void formatLogMsg(logLevel lvl, uint32_t line, const char *func,
                  const char *fmt, const fmt::format_args &args);

void createLogger(bool withConsole = false);

template <typename... Args>
inline void fmtLogMsg(logLevel lvl, uint32_t line, const char *func,
                      const char *fmt, const Args &... args) {
  formatLogMsg(lvl, line, func, fmt, fmt::make_format_args(args...));
}

template <typename... Args>
inline void fmtLogMsg(logLevel lvl, uint32_t line, const char *func,
                      const base::String &text) {
  formatLogMsg(lvl, line, func, text.c_str(), {});
}
}

#ifdef _DEBUG
#define LOG_TRACE(...)                                                         \
  ::utl::fmtLogMsg(::utl::logLevel::Trace, __LINE__, __func__, __VA_ARGS__)
#else
#define LOG_TRACE(fmt, ...) (void(0))
#endif

#define LOG_DEBUG(...)                                                         \
  ::utl::fmtLogMsg(::utl::logLevel::Debug, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...)                                                          \
  ::utl::fmtLogMsg(::utl::logLevel::Info, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARNING(...)                                                       \
  ::utl::fmtLogMsg(::utl::logLevel::Warning, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...)                                                         \
  ::utl::fmtLogMsg(::utl::logLevel::Error, __LINE__, __func__, __VA_ARGS__)
#define LOG_CRITICAL(...)                                                      \
  ::utl::fmtLogMsg(::utl::logLevel::Critical, __LINE__, __func__, __VA_ARGS__)
#define LOG_ASSERT(expression)                                                 \
  do {                                                                         \
    if (!(expression)) {                                                       \
      ::utl::fmtLogMsg(::utl::logLevel::Error, __LINE__, __func__,             \
                       "assertion failed at " #expression);                    \
      __debugbreak();                                                          \
    }                                                                          \
  \
} while (0)

#define LOG_UNIMPLEMENTED                                                      \
  ::utl::fmtLogMsg(::utl::logLevel::Error, __LINE__, __func__,                 \
                   "Unimplemented function")
