#include <unistd.h>
#include "base/arch.h"

#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"

#include "logger/logger.h"
#include "logger/threadsafe_queue.h"

namespace utl {

static base::Atomic<bool> g_log_silenced{false};
// Any address unique to the calling thread names it.
static mem_size ThisThread() {
  static thread_local char tag;
  return reinterpret_cast<mem_size>(&tag);
}
static base::Atomic<mem_size> g_dumping_thread{0};
void SilenceLogging() {
  g_dumping_thread.store(ThisThread(), base::memory_order_relaxed);
  g_log_silenced.store(true, base::memory_order_relaxed);
}

class LogRegistry {
  base::Mutex writing_lock_;
  base::UniquePointer<base::Thread> backend_thread_;
  base::Vector<base::UniquePointer<LogBase>> sinks_;
  common::MPSCQueue<LogEntry> pending_;
  base::TimeTicks time_origin_;

 public:
  LogRegistry(LogRegistry const&) = delete;
  const LogRegistry& operator=(LogRegistry const&) = delete;

  static LogRegistry& Instance() {
    static LogRegistry backend;
    return backend;
  }

  LogRegistry() {
    time_origin_ = base::TimeTicks::Now();

    backend_thread_ = base::MakeUnique<base::Thread>(
        "log",
        [this] {
          LogEntry entry;
          auto write_logs = [&](LogEntry& e) {
            base::LockGuard<base::Mutex> lock{writing_lock_};
            for (auto& sink : sinks_) {
              sink->Write(e);
            }
          };

          while (true) {
            entry = pending_.PopWait();

            if (entry.final_entry)
              break;

            write_logs(entry);
          }

          // drain (cap to avoid spinning forever during teardown)
          constexpr int kMaxLogsToWrite = 100;
          int logs_written = 0;
          while (logs_written++ < kMaxLogsToWrite && pending_.Pop(entry)) {
            write_logs(entry);
          }
        },
        /*start_now=*/true);
  }

  ~LogRegistry() {
    LogEntry entry;
    entry.final_entry = true;
    pending_.Push(entry);
    backend_thread_->Join();
  }

  void AddEntry(LogLevel lvl, u32 line, const char* func, base::String msg) {
    LogEntry entry{};
    entry.timestamp = base::TimeTicks::Now() - time_origin_;
    entry.log_level = lvl;
    entry.line_num = line;
    entry.function = base::String(func);
    entry.message = base::move(msg);

    if (g_log_silenced.load(base::memory_order_relaxed)) {
      // The crash handler stopped the backend thread so nothing races its
      // report on stderr, but the report itself comes through here, so the
      // dumping thread has to write its own lines, synchronously.
      if (g_dumping_thread.load(base::memory_order_relaxed) != ThisThread())
        return;
      base::String out = FormatLogEntry(entry);
      ssize_t w = ::write(2, out.c_str(), out.size());
      w = ::write(2, "\n", 1);
      (void)w;
      return;
    }

    pending_.Push(entry);
  }

  LogBase* AddSink(base::UniquePointer<LogBase> sink) {
    base::LockGuard<base::Mutex> lock{writing_lock_};
    auto* raw = sink.Get_UseOnlyIfYouKnowWhatYouareDoing();
    sinks_.push_back(base::move(sink));
    return raw;
  }

  void RemoveSink(base::StringRef name) {
    base::LockGuard<base::Mutex> lock{writing_lock_};
    // base::Vector lacks base::RemoveIf; do it inline.
    auto* it = sinks_.begin();
    auto* dst = sinks_.begin();
    for (; it != sinks_.end(); ++it) {
      if (name != base::StringRef((*it)->GetName())) {
        if (dst != it)
          *dst = base::move(*it);
        ++dst;
      }
    }
    while (sinks_.end() != dst)
      sinks_.pop_back();
  }

  LogBase* GetSink(base::StringRef name) {
    for (auto& sink : sinks_) {
      if (name == base::StringRef(sink->GetName()))
        return sink.Get_UseOnlyIfYouKnowWhatYouareDoing();
    }
    return nullptr;
  }
};

const char* GetLevelName(LogLevel log_level) {
  switch (log_level) {
    case LogLevel::kTrace:
      return "Trace";
    case LogLevel::kDebug:
      return "Debug";
    case LogLevel::kInfo:
      return "Info";
    case LogLevel::kWarning:
      return "Warning";
    case LogLevel::kError:
      return "Error";
    case LogLevel::kCritical:
      return "Critical";
    default:
      break;
  }
  return nullptr;
}

base::String FormatLogEntry(const LogEntry& entry) {
  const i64 us = entry.timestamp.InMicroseconds();
  u32 time_seconds = static_cast<u32>(us / 1000000);
  u32 time_fractional = static_cast<u32>(us % 1000000);

  const char* level_name = GetLevelName(entry.log_level);

  return base::Format("[{:4d}.{:06d}] <{}> {}:{}: {}", time_seconds,
                      time_fractional, level_name, entry.function,
                      entry.line_num, entry.message);
}

LogBase* AddLogSink(base::UniquePointer<LogBase> sink) {
  return LogRegistry::Instance().AddSink(base::move(sink));
}

void AddLogMsg(LogLevel lvl, u32 line, const char* func, base::String msg) {
  LogRegistry::Instance().AddEntry(lvl, line, func, base::move(msg));
}

LogBase* GetLogSink(base::StringRef name) {
  return LogRegistry::Instance().GetSink(name);
}

void RouteBaseLogging() {
  base::SetLogHandler(
      [](void*, const char* channel, base::LogLevel level, const char* msg) {
        static constexpr LogLevel kLevels[] = {
            LogLevel::kTrace,   LogLevel::kDebug, LogLevel::kInfo,
            LogLevel::kWarning, LogLevel::kError, LogLevel::kCritical};
        const auto i = static_cast<mem_size>(level);
        // The channel goes in the message as [channel], which is the form
        // these lines are grepped by; the function column names the bridge.
        FmtLogMsg(i < _countof(kLevels) ? kLevels[i] : LogLevel::kInfo, 0,
                  "base", "[{}] {}", channel ? channel : "?", msg ? msg : "");
      },
      nullptr);
}

}  // namespace utl
