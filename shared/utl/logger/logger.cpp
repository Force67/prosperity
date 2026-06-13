#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>

#include <base/containers/vector.h>
#include <base/memory/unique_pointer.h>
#include <base/strings/xstring.h>

#include "logger.h"
#include "threadsafe_queue.h"

namespace utl {

static std::atomic<bool> g_logSilenced{false};
void silenceLogging() { g_logSilenced.store(true, std::memory_order_relaxed); }

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

  void AddEntry(logLevel lvl, uint32_t line, const char *func,
                base::String msg) {
    if (g_logSilenced.load(std::memory_order_relaxed))
      return;  // crash handler is dumping; don't race it on stderr
    using std::chrono::duration_cast;
    using std::chrono::steady_clock;

    logEntry entry{};
    entry.timestamp = duration_cast<std::chrono::microseconds>(
        steady_clock::now() - time_origin);
    entry.log_level = lvl;
    entry.line_num = line;
    entry.function = base::String(func);
    entry.message = std::move(msg);

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
  uint32_t time_seconds =
      static_cast<unsigned int>(entry.timestamp.count() / 1000000);
  uint32_t time_fractional =
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

void formatLogMsg(logLevel lvl, uint32_t line, const char *func,
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

}  // namespace utl
