// Developer tool, not a test: lowers every SPIR-V module in a directory (by
// default the recompiler's cache, ~/.cache/ps4delta/spirv) to GLSL the way
// the OpenGL backend does, compiles and links it with the GL driver, and
// reports per-stage success counts and the most common failure reasons.
//
//   rhi_lowering_corpus [dir] [--threads N] [--limit N] [--dump DIR]
//
// --dump writes the GLSL and the error of each failure to DIR.

#include <epoxy/gl.h>

#include <dirent.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/opengl/gl_context.h"
#include "gpu/opengl/gl_shader_lowering.h"

namespace {

using namespace gpu;
using namespace gpu::opengl;

enum class Outcome { kOk, kReflect, kLower, kCompile, kLink, kSkipped };

struct Result {
  u32 stage = 0;
  Outcome outcome = Outcome::kOk;
  base::String reason;
};

// The *.spv files directly inside `dir`.
void ListSpirv(const base::String& dir, base::Vector<base::String>* files) {
  DIR* d = ::opendir(dir.c_str());
  if (!d)
    return;
  while (const dirent* e = ::readdir(d)) {
    const mem_size n = std::strlen(e->d_name);
    if (n > 4 && !std::strcmp(e->d_name + n - 4, ".spv"))
      files->push_back(dir + "/" + e->d_name);
  }
  ::closedir(d);
}

base::Vector<u32> ReadWords(const base::String& path) {
  base::Vector<u32> words;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    return words;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0 && n % 4 == 0) {
    words.resize(static_cast<size_t>(n) / 4);
    if (std::fread(words.data(), 1, static_cast<size_t>(n), f) !=
        static_cast<size_t>(n))
      words.clear();
  }
  std::fclose(f);
  return words;
}

// Modules that need a capability the backend reports absent: Vulkan never
// gets them either on such a device, so they are skipped, not failed.
const char* Unsupported(const base::Vector<u32>& words) {
  constexpr u32 kOpCapability = 17;
  constexpr u32 kPhysicalStorageBufferAddresses = 5347;
  for (size_t i = 5; i < words.size();) {
    const u32 op = words[i] & 0xffff, count = words[i] >> 16;
    if (!count)
      break;
    if (op == kOpCapability && i + 1 < words.size() &&
        words[i + 1] == kPhysicalStorageBufferAddresses)
      return "buffer addresses (Caps::buffer_address is false)";
    i += count;
  }
  return nullptr;
}

GLenum GlStage(u32 stage) {
  switch (stage) {
    case rhi::kStageVertex:
      return GL_VERTEX_SHADER;
    case rhi::kStageGeometry:
      return GL_GEOMETRY_SHADER;
    case rhi::kStageFragment:
      return GL_FRAGMENT_SHADER;
    case rhi::kStageCompute:
      return GL_COMPUTE_SHADER;
    default:
      return 0;
  }
}

const char* StageName(u32 stage) {
  switch (stage) {
    case rhi::kStageVertex:
      return "vertex";
    case rhi::kStageGeometry:
      return "geometry";
    case rhi::kStageFragment:
      return "fragment";
    case rhi::kStageCompute:
      return "compute";
    case rhi::kStageMesh:
      return "mesh";
    default:
      return "unknown";
  }
}

// The first line of a log, with line/column numbers and identifiers in
// quotes blanked so that equal causes group together.
base::String Reason(const base::String& log) {
  base::String line = log.substr(0, log.find('\n'));
  base::String out;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); i++) {
    const char c = line[i];
    if (c == '"' || c == '\'') {
      quoted = !quoted;
      out += c;
      if (quoted)
        out += "...";
      continue;
    }
    if (quoted)
      continue;
    if (c >= '0' && c <= '9') {
      if (out.empty() || out.back() != '#')
        out += '#';
      continue;
    }
    out += c;
  }
  return out.size() > 160 ? out.substr(0, 160) : out;
}

base::String InfoLog(GLuint object, bool program) {
  GLint n = 0;
  if (program)
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &n);
  else
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &n);
  base::String log(n > 0 ? n : 1, '\0');
  if (program)
    glGetProgramInfoLog(object, n, nullptr, log.data());
  else
    glGetShaderInfoLog(object, n, nullptr, log.data());
  return log;
}

Result Process(const base::String& path,
               const GlslFeatures& features,
               const base::String& dump_dir) {
  Result r;
  const base::Vector<u32> words = ReadWords(path);
  base::Vector<rhi::BindGroupLayoutDesc> groups;
  u32 push_bytes = 0;
  base::String error;
  if (words.empty() || !ReflectLayout({words.data(), words.size()}, &r.stage,
                                      &groups, &push_bytes, &error)) {
    r.outcome = Outcome::kReflect;
    r.reason = error.empty() ? "unreadable" : Reason(error);
    return r;
  }
  if (!GlStage(r.stage)) {
    r.outcome = Outcome::kSkipped;
    r.reason = "mesh shader (Caps::mesh_shader is false)";
    return r;
  }
  if (const char* why = Unsupported(words)) {
    r.outcome = Outcome::kSkipped;
    r.reason = why;
    return r;
  }
  base::Vector<const rhi::BindGroupLayoutDesc*> layout;
  for (const auto& g : groups)
    layout.push_back(&g);
  StageCode stage{r.stage, {words.data(), words.size()}};
  base::Vector<base::String> glsl;
  ProgramInterface program;
  if (!LowerProgram(&stage, 1, layout, features, &glsl, &program, &error)) {
    r.outcome = Outcome::kLower;
    r.reason = Reason(error);
    return r;
  }
  const GLuint shader = glCreateShader(GlStage(r.stage));
  const char* src = glsl[0].c_str();
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  base::String log;
  if (!ok) {
    r.outcome = Outcome::kCompile;
    log = InfoLog(shader, false);
  } else if (r.stage != rhi::kStageGeometry) {
    // Linked alone, as the backend links whole programs: a geometry shader
    // cannot link without a vertex shader, so it is only compiled.
    const GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
      r.outcome = Outcome::kLink;
      log = InfoLog(prog, true);
    }
    glDeleteProgram(prog);
  }
  glDeleteShader(shader);
  if (r.outcome != Outcome::kOk) {
    r.reason = Reason(log);
    if (!dump_dir.empty()) {
      base::String stem = path;
      const mem_size slash = stem.find_last_of('/');
      if (slash != base::String::npos)
        stem = stem.substr(slash + 1);
      const mem_size dot = stem.find_last_of('.');
      if (dot != base::String::npos)
        stem = stem.substr(0, dot);
      const base::String base = dump_dir + "/" + stem;
      if (FILE* f = std::fopen((base + ".glsl").c_str(), "w")) {
        std::fputs(glsl[0].c_str(), f);
        std::fclose(f);
      }
      if (FILE* f = std::fopen((base + ".log").c_str(), "w")) {
        std::fputs(log.c_str(), f);
        std::fclose(f);
      }
    }
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  base::String dir;
  base::String dump_dir;
  u32 threads = 1;
  size_t limit = ~size_t(0);
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--threads") && i + 1 < argc)
      threads = static_cast<u32>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--limit") && i + 1 < argc)
      limit = static_cast<size_t>(std::atoll(argv[++i]));
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc)
      dump_dir = argv[++i];
    else
      dir = argv[i];
  }
  if (dir.empty()) {
    const char* home = std::getenv("HOME");
    dir = base::String(home ? home : ".") + "/.cache/ps4delta/spirv";
  }
  if (!dump_dir.empty())
    ::mkdir(dump_dir.c_str(), 0755);

  base::Vector<base::String> files;
  ListSpirv(dir, &files);
  base::Sort(files.begin(), files.end());
  if (files.size() > limit)
    files.resize(limit);
  std::printf("%zu modules in %s\n", files.size(), dir.c_str());

  EglDevice device;
  if (!OpenEglDevice(nullptr, &device)) {
    std::printf("no GL 4.6 device\n");
    return 1;
  }
  std::printf("renderer: %s\n", device.renderer.c_str());

  GlslFeatures features;
  base::Vector<Result> results(files.size());
  base::Atomic<size_t> next{0};
  base::Vector<EGLContext> contexts;
  for (u32 i = 0; i < base::Max(threads, 1u); i++)
    contexts.push_back(CreateGlContext(device.display, EGL_NO_CONTEXT, false));
  base::Mutex features_mutex;
  bool features_ready = false;
  GlWorker workers;
  workers.Start(device.display, contexts, "corpus", [&] {
    base::LockGuard<base::Mutex> lock(features_mutex);
    if (features_ready)
      return;
    features_ready = true;
    features = QueryGlslFeatures();
  });
  base::Atomic<size_t> done{0};
  for (u32 i = 0; i < contexts.size(); i++)
    workers.Post([&] {
      for (size_t n; (n = next++) < files.size();) {
        results[n] = Process(files[n], features, dump_dir);
        const size_t d = ++done;
        if (d % 500 == 0)
          std::fprintf(stderr, "  %zu/%zu\n", d, files.size());
      }
    });
  workers.Stop();

  struct Counts {
    u32 total = 0, ok = 0, reflect = 0, lower = 0, compile = 0, link = 0,
        skipped = 0;
  };
  base::Map<u32, Counts> per_stage;
  base::Map<base::String, u32> reasons;
  base::Map<base::String, u32> skips;
  base::Map<base::String, base::String> example;
  for (size_t i = 0; i < results.size(); i++) {
    const Result& r = results[i];
    Counts& c = per_stage[r.stage];
    c.total++;
    switch (r.outcome) {
      case Outcome::kOk:
        c.ok++;
        continue;
      case Outcome::kReflect:
        c.reflect++;
        break;
      case Outcome::kLower:
        c.lower++;
        break;
      case Outcome::kCompile:
        c.compile++;
        break;
      case Outcome::kLink:
        c.link++;
        break;
      case Outcome::kSkipped:
        c.skipped++;
        skips[base::String(StageName(r.stage)) + ": " + r.reason]++;
        continue;
    }
    const base::String key = base::String(StageName(r.stage)) + ": " + r.reason;
    reasons[key]++;
    example.emplace(key, files[i]);
  }
  std::printf("\n%-10s %7s %7s %7s %7s %7s %7s %7s\n", "stage", "total", "ok",
              "reflect", "lower", "compile", "link", "skipped");
  Counts all;
  for (const auto& [stage, c] : per_stage) {
    std::printf("%-10s %7u %7u %7u %7u %7u %7u %7u\n", StageName(stage),
                c.total, c.ok, c.reflect, c.lower, c.compile, c.link,
                c.skipped);
    all.total += c.total;
    all.ok += c.ok;
    all.reflect += c.reflect;
    all.lower += c.lower;
    all.compile += c.compile;
    all.link += c.link;
    all.skipped += c.skipped;
  }
  std::printf("%-10s %7u %7u %7u %7u %7u %7u %7u\n", "all", all.total, all.ok,
              all.reflect, all.lower, all.compile, all.link, all.skipped);

  if (!skips.empty())
    std::printf("\nskipped:\n");
  for (const auto& [reason, n] : skips)
    std::printf("%6u  %s\n", n, reason.c_str());

  base::Vector<base::Pair<u32, base::String>> top;
  for (const auto& [reason, n] : reasons)
    top.push_back({n, reason});
  base::Sort(top.begin(), top.end(),
             [](const auto& a, const auto& b) { return b < a; });
  if (!top.empty())
    std::printf("\ntop failure reasons:\n");
  for (size_t i = 0; i < top.size() && i < 25; i++)
    std::printf("%6u  %s\n        e.g. %s\n", top[i].first,
                top[i].second.c_str(), example[top[i].second].c_str());
  return 0;
}
