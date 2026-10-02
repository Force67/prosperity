#include "main/recent_games.h"

#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <cstdio>

#include "base/environment_variables.h"
#include "base/filesystem/file.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "io/file.h"
#include "logger/logger.h"

namespace cli {
namespace {
constexpr mem_size kMaxRecentGames = 12;
constexpr u64 kMaxImageBytes = 16u << 20;

base::String DataDirectory() {
  base::StringU8 value;
  if (base::GetEnvironmentVariable(u8"XDG_DATA_HOME", value) &&
      !value.empty() && value.front() == '/')
    return base::String(reinterpret_cast<const char*>(value.c_str())) +
           "/prosperity";
  if (base::GetEnvironmentVariable(u8"HOME", value) && !value.empty())
    return base::String(reinterpret_cast<const char*>(value.c_str())) +
           "/.local/share/prosperity";
  return {};
}

bool CreateDirectories(const base::String& path) {
  for (mem_size i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/')
      continue;
    if (::mkdir(path.substr(0, i).c_str(), 0700) != 0 && errno != EEXIST)
      return false;
  }
  return true;
}

base::String Encode(const base::String& text) {
  constexpr char kHex[] = "0123456789abcdef";
  base::String result;
  for (unsigned char byte : text) {
    result += kHex[byte >> 4];
    result += kHex[byte & 15];
  }
  return result;
}

bool Decode(const base::String& text, base::String* result) {
  if (text.size() % 2 != 0)
    return false;
  auto digit = [](char c) {
    return c >= '0' && c <= '9'   ? c - '0'
           : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                  : -1;
  };
  for (mem_size i = 0; i < text.size(); i += 2) {
    const int a = digit(text[i]), b = digit(text[i + 1]);
    if (a < 0 || b < 0 || (a == 0 && b == 0))
      return false;
    *result += static_cast<char>((a << 4) | b);
  }
  return true;
}

base::String ImagePath(const base::String& directory,
                       const base::String& path,
                       const char* kind) {
  u64 hash = 14695981039346656037ull;
  for (unsigned char byte : path) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return base::Format("{}/covers/{:016x}-{}.png", directory, hash, kind);
}

base::Vector<u8> ReadBytes(const base::String& path, u64 limit) {
  io::File file(path, io::FileMode::kRead);
  if (!file.IsOpen() || file.GetSize() == 0 || file.GetSize() > limit)
    return {};
  base::Vector<u8> result(file.GetSize());
  if (file.Read(result.data(), result.size()) != result.size())
    return {};
  return result;
}

bool WriteBytes(const base::String& path, const void* bytes, mem_size size) {
  const auto temporary = path + ".tmp";
  base::File file(base::Path(base::StringRefU8(
                      reinterpret_cast<const char8_t*>(temporary.c_str()))),
                  base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
  if (!file.IsValid() || file.WriteAtCurrentPos(static_cast<const char*>(bytes),
                                                static_cast<int>(size)) !=
                             static_cast<int>(size)) {
    ::unlink(temporary.c_str());
    return false;
  }
  file.Close();
  if (::rename(temporary.c_str(), path.c_str()) == 0)
    return true;
  ::unlink(temporary.c_str());
  return false;
}

base::Vector<ui::HomeGame> ReadEntries(const base::String& directory) {
  base::Vector<ui::HomeGame> result;
  const auto bytes = ReadBytes(directory + "/recent-games", 1u << 20);
  if (bytes.empty())
    return result;
  const base::String text(reinterpret_cast<const char*>(bytes.data()),
                          bytes.size());
  mem_size start = 0;
  while (start < text.size() && result.size() < kMaxRecentGames) {
    mem_size end = text.find('\n', start);
    if (end == base::String::npos)
      end = text.size();
    const auto line = text.substr(start, end - start);
    start = end + 1;
    const auto a = line.find('\t');
    const auto b = a == base::String::npos ? a : line.find('\t', a + 1);
    const auto c = b == base::String::npos ? b : line.find('\t', b + 1);
    if (c == base::String::npos)
      continue;
    ui::HomeGame entry;
    const auto platform = line.substr(c + 1);
    if (!Decode(line.substr(0, a), &entry.path) || entry.path.empty() ||
        !Decode(line.substr(a + 1, b - a - 1), &entry.name) ||
        !Decode(line.substr(b + 1, c - b - 1), &entry.title_id) ||
        (platform != "4" && platform != "5"))
      continue;
    entry.is_ps5 = platform == "5";
    bool duplicate = false;
    for (const auto& previous : result)
      duplicate |= previous.path == entry.path;
    if (!duplicate)
      result.push_back(base::move(entry));
  }
  return result;
}
}  // namespace

base::Vector<ui::HomeGame> ReadRecentGames() {
  const auto directory = DataDirectory();
  if (directory.empty())
    return {};
  auto result = ReadEntries(directory);
  for (auto& game : result) {
    struct stat info{};
    game.available = ::stat(game.path.c_str(), &info) == 0;
    game.icon =
        ReadBytes(ImagePath(directory, game.path, "icon"), kMaxImageBytes);
    game.artwork =
        ReadBytes(ImagePath(directory, game.path, "art"), kMaxImageBytes);
  }
  return result;
}

void RememberGame(const base::String& path,
                  const base::String& name,
                  const base::String& title_id,
                  bool is_ps5,
                  const base::Vector<u8>& icon,
                  const base::Vector<u8>& artwork) {
  const auto directory = DataDirectory();
  if (directory.empty() || !CreateDirectories(directory + "/covers"))
    return;
  char absolute[PATH_MAX];
  if (!::realpath(path.c_str(), absolute))
    return;
  auto entries = ReadEntries(directory);
  const base::String canonical(absolute);
  for (auto it = entries.begin(); it != entries.end();) {
    if (it->path == canonical)
      it = entries.erase(it, it + 1);
    else
      ++it;
  }
  ui::HomeGame game;
  game.path = canonical;
  game.name =
      name.empty() ? canonical.substr(canonical.find_last_of('/') + 1) : name;
  game.title_id = title_id;
  game.is_ps5 = is_ps5;
  entries.insert(entries.begin(), base::move(game));
  if (entries.size() > kMaxRecentGames) {
    const auto& removed = entries.back();
    ::unlink(ImagePath(directory, removed.path, "icon").c_str());
    ::unlink(ImagePath(directory, removed.path, "art").c_str());
    entries.resize(kMaxRecentGames);
  }
  if (!icon.empty())
    WriteBytes(ImagePath(directory, canonical, "icon"), icon.data(),
               icon.size());
  if (!artwork.empty())
    WriteBytes(ImagePath(directory, canonical, "art"), artwork.data(),
               artwork.size());
  base::String index;
  for (const auto& entry : entries)
    base::FormatTo(index, "{}\t{}\t{}\t{}\n", Encode(entry.path),
                   Encode(entry.name), Encode(entry.title_id),
                   entry.is_ps5 ? 5 : 4);
  if (!WriteBytes(directory + "/recent-games", index.data(), index.size()))
    LOG_WARNING("Could not save recent games");
}

}  // namespace cli
