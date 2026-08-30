#include <algorithm>
#include "base/arch.h"
#include <atomic>
#include <iterator>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <base/containers/vector.h>
#include <base/logging.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>

#include "logger.h"
#include "threadsafe_queue.h"

namespace utl {

static std::atomic<bool> g_logSilenced{false};
static std::atomic<std::thread::id> g_dumpingThread{};
void silenceLogging() {
  g_dumpingThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
  g_logSilenced.store(true, std::memory_order_relaxed);
}

class LogRegistry {
  std::mutex writing_lock;
  std::thread backend_thread;
  base::Vector<base::UniquePointer<logBase>> sinks;
  Common::MPSCQueue<logEntry> pending;
  std::chrono::steady_clock::time_point time_origin;

public:
  LogRegistry(LogRegistry const &) = delete;
  const LogRegistry &operator=(LogRegistry const &) = delete;

  static LogRegistry &Instance() {
    static LogRegistry backend;
    return backend;
  }

  LogRegistry() {
    time_origin = std::chrono::steady_clock::now();

    backend_thread = std::thread([&] {
      logEntry entry;
      auto write_logs = [&](logEntry &e) {
        std::lock_guard lock{writing_lock};
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
    });
  }

  ~LogRegistry() {
    logEntry entry;
    entry.final_entry = true;
    pending.Push(entry);
    backend_thread.join();
  }

  void AddEntry(logLevel lvl, u32 line, const char *func,
                base::String msg) {
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    logEntry entry{};
    entry.timestamp = duration_cast<std::chrono::microseconds>(
        steady_clock::now() - time_origin);
    entry.log_level = lvl;
    entry.line_num = line;
    entry.function = base::String(func);
    entry.message = std::move(msg);

    if (g_logSilenced.load(std::memory_order_relaxed)) {
      // The crash handler stopped the backend thread so nothing races its
      // report on stderr -- but the report itself comes through here, so the
      // dumping thread has to write its own lines, synchronously.
      if (g_dumpingThread.load(std::memory_order_relaxed) !=
          std::this_thread::get_id())
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
    std::lock_guard lock{writing_lock};
    auto *raw = sink.Get_UseOnlyIfYouKnowWhatYouareDoing();
    sinks.push_back(std::move(sink));
    return raw;
  }

  void RemoveSink(base::StringRef name) {
    std::lock_guard lock{writing_lock};
    // base::Vector lacks std::remove_if; do it inline.
    auto* it = sinks.begin();
    auto* dst = sinks.begin();
    for (; it != sinks.end(); ++it) {
      if (name != base::StringRef((*it)->getName())) {
        if (dst != it) *dst = std::move(*it);
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
  u32 time_seconds =
      static_cast<unsigned int>(entry.timestamp.count() / 1000000);
  u32 time_fractional =
      static_cast<unsigned int>(entry.timestamp.count() % 1000000);

  const char *level_name = GetLevelName(entry.log_level);

  // fmt::format produces std::string; copy into base::String once.
  std::string s = fmt::format("[{:4d}.{:06d}] <{}> {}:{}: {}", time_seconds,
                              time_fractional, level_name,
                              entry.function.c_str(), entry.line_num,
                              entry.message.c_str());
  return base::String(s.c_str(), static_cast<base::String::size_type>(s.size()));
}

logBase *addLogSink(base::UniquePointer<logBase> sink) {
  return LogRegistry::Instance().AddSink(std::move(sink));
}

void formatLogMsg(logLevel lvl, u32 line, const char *func,
                  const char *fmt, const fmt::format_args &args) {
  std::string s = fmt::vformat(fmt, args);
  auto &reg = LogRegistry::Instance();
  reg.AddEntry(lvl, line, func,
               base::String(s.c_str(),
                            static_cast<base::String::size_type>(s.size())));
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
        const auto i = static_cast<size_t>(level);
        // The channel goes in the message as [channel], which is the form
        // these lines are grepped by; the function column names the bridge.
        fmtLogMsg(i < std::size(kLevels) ? kLevels[i] : logLevel::Info, 0,
                  "base", "[{}] {}", channel ? channel : "?", msg ? msg : "");
      },
      nullptr);
}

}  // namespace utl
