/*
 * PS4Delta : PS4 emulation and research project
 *
 * Shader translation audit. See gcn_audit.h.
 */

#include "gpu/gcn/gcn_audit.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <unordered_map>

#include "gpu/gcn/gcn_disasm.h"

namespace gpu::gcn {
namespace {

const char* DumpDir() {
  static const char* dir = std::getenv("DELTA_GPU_SHDUMP");
  return dir && *dir ? dir : nullptr;
}

const char* AuditEnv() {
  static const char* v = std::getenv("DELTA_GPU_SHAUDIT");
  return v;
}

uint64_t Fnv1a(const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; i++)
    h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

// An instruction the translator is expected to emit nothing for; not a
// silent drop.
bool ExpectedSilent(const Inst& inst) {
  switch (inst.enc) {
    case Enc::kSopp:
      return true;  // waitcnt/nop/hints; branches are CFG terminators
    case Enc::kVintrp:
      return inst.opcode == 0;  // P1 is a no-op in the interpolation model
    case Enc::kExp:
      return (inst.raw[0] & 0xF) == 0;  // null export
    default:
      return false;
  }
}

struct InstRecord {
  uint32_t words = 0;
  uint16_t notes = 0;
  bool visited = false;
  const char* tag = nullptr;
};

struct EventStat {
  uint32_t sites = 0;
  uint32_t first_pc = ~0u;
};

struct Active {
  bool on = false;
  const char* stage = "";
  const uint32_t* code = nullptr;
  const Program* program = nullptr;
  std::vector<InstRecord> insts;
  std::vector<uint8_t> reachable;
  // note key ("mubuf.ps op=0x0 buffer_load_format_x") -> stats
  std::map<std::string, EventStat> events;
  std::vector<std::string> plan;
  std::string decline;
  uint32_t cur = ~0u;  // program index currently being emitted
};

struct ShaderRecord {
  std::string stage;
  uint64_t hash = 0;
  uint64_t guest = 0;
  uint32_t dwords = 0;
  uint32_t insts = 0, dead = 0, silent_count = 0;
  uint32_t recompiles = 1;
  bool declined = false;
  std::string decline_reason;
  std::map<std::string, EventStat> events;
  std::map<std::string, EventStat> silent;  // mnemonic -> stats
};

Active g_active;
std::vector<ShaderRecord> g_records;
std::unordered_map<uint64_t, size_t> g_by_key;  // hash^stage -> record index
bool g_atexit_registered = false;

// Rewrite the report file after every new shader. The emulator is routinely
// stopped with SIGKILL (it ignores TERM), which skips atexit -- an
// exit-only report would usually be lost.
void MaybeWriteReportFile() {
  const char* v = AuditEnv();
  // Any value other than a plain enable flag names a report file.
  if (!v || !*v || std::strcmp(v, "1") == 0)
    return;
  if (std::FILE* f = std::fopen(v, "w")) {
    WriteAuditReport(f);
    std::fclose(f);
  } else {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr, "[shaudit] cannot write report to %s\n", v);
    }
  }
}

void ReportAtExit() {
  if (!AuditEnv())
    return;
  WriteAuditReport(stderr);
}

void WriteDumpFiles(const ShaderRecord& rec,
                    const Active& a,
                    const std::vector<uint32_t>* spirv) {
  const char* dir = DumpDir();
  if (!dir)
    return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);  // best effort

  char stem[256];
  std::snprintf(stem, sizeof(stem), "%s/%s_%016llx", dir, rec.stage.c_str(),
                static_cast<unsigned long long>(rec.hash));

  // Raw bytecode.
  if (std::FILE* f = std::fopen((std::string(stem) + ".gcn").c_str(), "wb")) {
    std::fwrite(a.code, 4, rec.dwords, f);
    std::fclose(f);
  }
  // Unoptimized SPIR-V (with OpLine pc markers when the translator saw the
  // dump env).
  if (spirv && !spirv->empty()) {
    if (std::FILE* f = std::fopen((std::string(stem) + ".spv").c_str(), "wb")) {
      std::fwrite(spirv->data(), 4, spirv->size(), f);
      std::fclose(f);
    }
  }

  std::FILE* f = std::fopen((std::string(stem) + ".txt").c_str(), "w");
  if (!f) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr, "[shdump] cannot write to %s\n", dir);
    }
    return;
  }
  std::fprintf(f, "; stage=%s hash=%016llx guest=%p dwords=%u verdict=%s\n",
               rec.stage.c_str(), static_cast<unsigned long long>(rec.hash),
               static_cast<const void*>(a.code), rec.dwords,
               rec.declined ? rec.decline_reason.c_str() : "ok");
  for (const std::string& line : a.plan)
    std::fprintf(f, "; %s\n", line.c_str());
  std::fprintf(
      f, "; fates: %u insts, %u dead, %u unsupported-sites, %u silent\n",
      rec.insts, rec.dead,
      [&] {
        uint32_t n = 0;
        for (const auto& e : rec.events)
          n += e.second.sites;
        return n;
      }(),
      rec.silent_count);
  // Shader-level events (no pc attribution: binding-plan overflows etc.).
  for (const auto& e : a.events)
    if (e.second.first_pc == ~0u)
      std::fprintf(f, "; NOTE %s (x%u)\n", e.first.c_str(), e.second.sites);
  std::fprintf(f, ";\n");

  const Program& program = *a.program;
  for (uint32_t i = 0; i < program.size(); i++) {
    const Inst& inst = program[i];
    const InstRecord& ir = i < a.insts.size() ? a.insts[i] : InstRecord{};
    std::string line = DisasmLine(inst);
    std::string mark;
    if (i < a.reachable.size() && !a.reachable[i]) {
      mark = " ; dead";
    } else {
      char w[32];
      std::snprintf(w, sizeof(w), " ; %uw", ir.words);
      mark = w;
      if (ir.tag) {
        mark += " =";
        mark += ir.tag;
      }
      if (ir.notes)
        mark += " !UNSUPPORTED";
      else if (ir.visited && ir.words == 0 && !ir.tag && !ExpectedSilent(inst))
        mark += " !SILENT";
    }
    std::fprintf(f, "%s%s\n", line.c_str(), mark.c_str());
  }
  std::fclose(f);
}

}  // namespace

bool ShaderDebugEnabled() {
  static const bool enabled = DumpDir() != nullptr || AuditEnv() != nullptr;
  return enabled;
}

bool ShaderDumpEnabled() {
  static const bool enabled = DumpDir() != nullptr;
  return enabled;
}

void AuditBegin(const char* stage,
                const uint32_t* code,
                const Program& program) {
  if (!ShaderDebugEnabled())
    return;
  if (!g_atexit_registered) {
    g_atexit_registered = true;
    std::atexit(ReportAtExit);
  }
  g_active = Active{};
  g_active.on = true;
  g_active.stage = stage;
  g_active.code = code;
  g_active.program = &program;
  g_active.insts.assign(program.size(), InstRecord{});
  g_active.reachable = ComputeReachability(program);
}

void AuditInstBegin(uint32_t index, uint32_t pc) {
  (void)pc;
  if (!g_active.on)
    return;
  g_active.cur = index;
}

void AuditInstEnd(uint32_t index, uint32_t spirv_words) {
  if (!g_active.on || index >= g_active.insts.size())
    return;
  InstRecord& ir = g_active.insts[index];
  ir.visited = true;
  ir.words += spirv_words;
  g_active.cur = ~0u;
}

void AuditInstTag(const char* tag) {
  if (!g_active.on || g_active.cur >= g_active.insts.size())
    return;
  g_active.insts[g_active.cur].tag = tag;
}

void AuditNote(const char* what, uint32_t op) {
  if (!g_active.on)
    return;
  char key[128];
  const uint32_t cur = g_active.cur;
  if (cur < g_active.program->size()) {
    const Inst& inst = (*g_active.program)[cur];
    std::snprintf(key, sizeof(key), "%s op=0x%x (%s)", what, op,
                  Mnemonic(inst).c_str());
    g_active.insts[cur].notes++;
    EventStat& st = g_active.events[key];
    st.sites++;
    if (st.first_pc == ~0u)
      st.first_pc = inst.pc;
  } else {
    std::snprintf(key, sizeof(key), "%s op=0x%x", what, op);
    g_active.events[key].sites++;
  }
}

void AuditPlan(const std::string& line) {
  if (g_active.on)
    g_active.plan.push_back(line);
}

void AuditDecline(const char* reason) {
  if (g_active.on)
    g_active.decline = reason;
}

void AuditEnd(const std::vector<uint32_t>* spirv) {
  if (!g_active.on)
    return;
  Active a = std::move(g_active);
  g_active = Active{};

  const Program& program = *a.program;
  const uint32_t dwords =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const uint64_t hash = Fnv1a(a.code, dwords * 4ull);
  const uint64_t key = hash ^ Fnv1a(a.stage, std::strlen(a.stage));
  auto it = g_by_key.find(key);
  if (it != g_by_key.end()) {
    g_records[it->second].recompiles++;
    return;
  }

  ShaderRecord rec;
  rec.stage = a.stage;
  rec.hash = hash;
  rec.guest = reinterpret_cast<uint64_t>(a.code);
  rec.dwords = dwords;
  rec.insts = static_cast<uint32_t>(program.size());
  rec.declined = !a.decline.empty();
  rec.decline_reason = a.decline;
  rec.events = a.events;
  for (uint32_t i = 0; i < program.size(); i++) {
    if (i < a.reachable.size() && !a.reachable[i]) {
      rec.dead++;
      continue;
    }
    const InstRecord& ir = a.insts[i];
    if (ir.visited && ir.words == 0 && !ir.notes && !ir.tag &&
        !ExpectedSilent(program[i])) {
      rec.silent_count++;
      EventStat& st = rec.silent[Mnemonic(program[i])];
      st.sites++;
      if (st.first_pc == ~0u)
        st.first_pc = program[i].pc;
    }
  }
  g_by_key[key] = g_records.size();
  g_records.push_back(rec);
  WriteDumpFiles(rec, a, spirv);
  MaybeWriteReportFile();
}

void WriteAuditReport(std::FILE* f) {
  uint32_t total = 0, declined = 0;
  std::map<std::string, uint32_t> per_stage;
  for (const ShaderRecord& r : g_records) {
    total++;
    per_stage[r.stage]++;
    if (r.declined)
      declined++;
  }
  std::fprintf(f, "[shaudit] ============ GCN shader audit ============\n");
  std::fprintf(f, "[shaudit] unique shaders: %u (", total);
  bool first = true;
  for (const auto& s : per_stage) {
    std::fprintf(f, "%s%s=%u", first ? "" : " ", s.first.c_str(), s.second);
    first = false;
  }
  std::fprintf(f, "), declined: %u\n", declined);

  // Aggregate events / silents across shaders: key -> (#shaders, #sites,
  // example).
  struct Agg {
    uint32_t shaders = 0, sites = 0;
    std::string example;
  };
  const auto example_of = [](const ShaderRecord& r, const EventStat& st) {
    char buf[96];
    if (st.first_pc != ~0u)
      std::snprintf(buf, sizeof(buf), "%s_%016llx pc=0x%x", r.stage.c_str(),
                    static_cast<unsigned long long>(r.hash), st.first_pc);
    else
      std::snprintf(buf, sizeof(buf), "%s_%016llx", r.stage.c_str(),
                    static_cast<unsigned long long>(r.hash));
    return std::string(buf);
  };
  std::map<std::string, Agg> events, silents;
  for (const ShaderRecord& r : g_records) {
    for (const auto& e : r.events) {
      Agg& agg = events[e.first];
      agg.shaders++;
      agg.sites += e.second.sites;
      if (agg.example.empty())
        agg.example = example_of(r, e.second);
    }
    for (const auto& e : r.silent) {
      Agg& agg = silents[e.first];
      agg.shaders++;
      agg.sites += e.second.sites;
      if (agg.example.empty())
        agg.example = example_of(r, e.second);
    }
  }
  const auto print_ranked = [f](const char* title,
                                const std::map<std::string, Agg>& m) {
    if (m.empty())
      return;
    std::fprintf(f, "[shaudit] -- %s --\n", title);
    std::vector<const std::pair<const std::string, Agg>*> rows;
    for (const auto& e : m)
      rows.push_back(&e);
    std::sort(rows.begin(), rows.end(), [](const auto* x, const auto* y) {
      if (x->second.shaders != y->second.shaders)
        return x->second.shaders > y->second.shaders;
      return x->second.sites > y->second.sites;
    });
    for (const auto* row : rows)
      std::fprintf(f, "[shaudit]   %3u shaders / %4u sites  %-40s e.g. %s\n",
                   row->second.shaders, row->second.sites, row->first.c_str(),
                   row->second.example.c_str());
  };
  print_ranked("unsupported / approximated ops (fix these first)", events);
  print_ranked("silently dropped (0 SPIR-V ops, no warning)", silents);

  if (declined) {
    std::fprintf(f, "[shaudit] -- declined shaders --\n");
    for (const ShaderRecord& r : g_records)
      if (r.declined)
        std::fprintf(f, "[shaudit]   %s_%016llx guest=0x%llx x%u: %s\n",
                     r.stage.c_str(), static_cast<unsigned long long>(r.hash),
                     static_cast<unsigned long long>(r.guest), r.recompiles,
                     r.decline_reason.c_str());
  }
  if (ShaderDumpEnabled())
    std::fprintf(f, "[shaudit] per-shader dumps in %s\n", DumpDir());
  std::fprintf(f, "[shaudit] ==========================================\n");
}

}  // namespace gpu::gcn
