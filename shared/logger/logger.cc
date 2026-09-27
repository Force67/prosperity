#include "base/arch.h"
#include <unistd.h>

#include <base/atomic.h>
#include <base/containers/vector.h>
#include <base/logging.h>
#include <base/memory/move.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/format.h>
#include <base/strings/string_ref.h>
#include <base/strings/xstring.h>
#include <base/threading/lock_guard.h>
#include <base/threading/mutex.h>
#include <base/threading/thread.h>
#include <base/time/time.h>

#include "logger.h"
#include "threadsafe_queue.h"

namespace utl {

static base::Atomic<bool> g_logSilenced{false};
// Any address unique to the calling thread names it.
static mem_size ThisThread() {
  static thread_local char tag;
  return reinterpret_cast<mem_size>(&tag);
}
static base::Atomic<mem_size> g_dumpingThread{0};
void silenceLogging() {
  g_dumpingThread.store(ThisThread(), base::memory_order_relaxed);
  g_logSilenced.store(true, base::memory_order_relaxed);
}

class LogRegistry {
  base::Mutex writing_lock;
  base::UniquePointer<base::Thread> backend_thread;
  base::Vector<base::UniquePointer<logBase>> sinks;
  Common::MPSCQueue<logEntry> pending;
  base::TimeTicks time_origin;

public:
  LogRegistry(LogRegistry const &) = delete;
  const LogRegistry &operator=(LogRegistry const &) = delete;

  static LogRegistry &Instance() {
    static LogRegistry backend;
    return backend;
  }

  LogRegistry() {
    time_origin = base::TimeTicks::Now();

    backend_thread = base::MakeUnique<base::Thread>("log", [this] {
      logEntry entry;
      auto write_logs = [&](logEntry &e) {
        base::LockGuard<base::Mutex> lock{writing_lock};
        for (auto &sink : sinks) {
          sink->write(e);
        }
      };

      while (true) {
        entry = pending.PopWait();

        if (entry.final_entry)
          break;

        write_logs(entry);
      }

      // drain (cap to avoid spinning forever during teardown)
      constexpr int MAX_LOGS_TO_WRITE = 100;
      int logs_written = 0;
      while (logs_written++ < MAX_LOGS_TO_WRITE && pending.Pop(entry)) {
        write_logs(entry);
      }
    }, /*start_now=*/true);
  }

  ~LogRegistry() {
    logEntry entry;
    entry.final_entry = true;
    pending.Push(entry);
    backend_thread->Join();
  }

  void AddEntry(logLevel lvl, u32 line, const char *func,
                base::String msg) {
    logEntry entry{};
    entry.timestamp = base::TimeTicks::Now() - time_origin;
    entry.log_level = lvl;
    entry.line_num = line;
    entry.function = base::String(func);
    entry.message = base::move(msg);

    if (g_logSilenced.load(base::memory_order_relaxed)) {
      // The crash handler stopped the backend thread so nothing races its
      // report on stderr, but the report itself comes through here, so the
      // dumping thread has to write its own lines, synchronously.
      if (g_dumpingThread.load(base::memory_order_relaxed) != ThisThread())
        return;
      base::String out = formatLogEntry(entry);
      ssize_t w = ::write(2, out.c_str(), out.size());
      w = ::write(2, "\n", 1);
      (void)w;
      return;
    }

    pending.Push(entry);
  }

  logBase *AddSink(base::UniquePointer<logBase> sink) {
    base::LockGuard<base::Mutex> lock{writing_lock};
    auto *raw = sink.Get_UseOnlyIfYouKnowWhatYouareDoing();
    sinks.push_back(base::move(sink));
    return raw;
  }

  void RemoveSink(base::StringRef name) {
    base::LockGuard<base::Mutex> lock{writing_lock};
    // base::Vector lacks base::RemoveIf; do it inline.
    auto* it = sinks.begin();
    auto* dst = sinks.begin();
    for (; it != sinks.end(); ++it) {
      if (name != base::StringRef((*it)->getName())) {
        if (dst != it) *dst = base::move(*it);
        ++dst;
      }
    }
    while (sinks.end() != dst) sinks.pop_back();
  }

  logBase *GetSink(base::StringRef name) {
    for (auto &sink : sinks) {
      if (name == base::StringRef(sink->getName()))
        return sink.Get_UseOnlyIfYouKnowWhatYouareDoing();
    }
    return nullptr;
  }
};

const char *GetLevelName(logLevel log_level) {
#define LVL(x)                                                                 \
  case logLevel::x:                                                            \
    return #x
  switch (log_level) {
    LVL(Trace);
    LVL(Debug);
    LVL(Info);
    LVL(Warning);
    LVL(Error);
    LVL(Critical);
    default: break;
  }
#undef LVL
  return nullptr;
}

base::String formatLogEntry(const logEntry &entry) {
  const i64 us = entry.timestamp.InMicroseconds();
  u32 time_seconds = static_cast<u32>(us / 1000000);
  u32 time_fractional = static_cast<u32>(us % 1000000);

  const char *level_name = GetLevelName(entry.log_level);

  return base::Format("[{:4d}.{:06d}] <{}> {}:{}: {}", time_seconds,
                      time_fractional, level_name, entry.function,
                      entry.line_num, entry.message);
}

logBase *addLogSink(base::UniquePointer<logBase> sink) {
  return LogRegistry::Instance().AddSink(base::move(sink));
}

void addLogMsg(logLevel lvl, u32 line, const char *func, base::String msg) {
  LogRegistry::Instance().AddEntry(lvl, line, func, base::move(msg));
}

logBase *getLogSink(base::StringRef name) {
  return LogRegistry::Instance().GetSink(name);
}

void routeBaseLogging() {
  base::SetLogHandler(
      [](void *, const char *channel, base::LogLevel level, const char *msg) {
        static constexpr logLevel kLevels[] = {
            logLevel::Trace, logLevel::Debug, logLevel::Info,
            logLevel::Warning, logLevel::Error, logLevel::Critical};
        const auto i = static_cast<mem_size>(level);
        // The channel goes in the message as [channel], which is the form
        // these lines are grepped by; the function column names the bridge.
        fmtLogMsg(i < _countof(kLevels) ? kLevels[i] : logLevel::Info, 0,
                  "base", "[{}] {}", channel ? channel : "?", msg ? msg : "");
      },
      nullptr);
}

}  // namespace utl
