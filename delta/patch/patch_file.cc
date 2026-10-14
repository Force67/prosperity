/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "patch/patch_file.h"

#include <cstring>
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/number_parse.h"

namespace patch {
namespace {

using Ref = base::StringRef;

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\r';
}

Ref Trim(Ref s) {
  while (!s.empty() && IsSpace(s[0]))
    s = s.substr(1);
  while (!s.empty() && IsSpace(s[s.size() - 1]))
    s = s.substr(0, s.size() - 1);
  return s;
}

Ref Unquote(Ref s) {
  if (s.size() >= 2 && (s[0] == '"' || s[0] == '\'') && s[s.size() - 1] == s[0])
    return s.substr(1, s.size() - 2);
  return s;
}

// Index of the first `c` outside quotes, or npos.
mem_size FindUnquoted(Ref s, char c, mem_size from = 0) {
  char quote = 0;
  for (mem_size i = from; i < s.size(); i++) {
    if (quote) {
      if (s[i] == quote)
        quote = 0;
    } else if (s[i] == '"' || s[i] == '\'') {
      quote = s[i];
    } else if (s[i] == c) {
      return i;
    }
  }
  return Ref::npos;
}

Ref StripComment(Ref s) {
  for (mem_size at = FindUnquoted(s, '#'); at != Ref::npos;
       at = FindUnquoted(s, '#', at + 1))
    if (at == 0 || IsSpace(s[at - 1]))
      return s.substr(0, at);
  return s;
}

// "key: value" or "key:"; the colon must be followed by a space or the end.
bool SplitKey(Ref line, Ref* key, Ref* value) {
  for (mem_size at = FindUnquoted(line, ':'); at != Ref::npos;
       at = FindUnquoted(line, ':', at + 1)) {
    if (at + 1 < line.size() && !IsSpace(line[at + 1]))
      continue;
    *key = Unquote(Trim(line.substr(0, at)));
    *value = Trim(line.substr(at + 1));
    return !key->empty();
  }
  return false;
}

bool ParseNumber(Ref text, u64* out) {
  const base::String s(text.data(), text.size());
  const bool hex = s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
  const char* end = nullptr;
  return base::ParseUnsigned(s.c_str() + (hex ? 2 : 0), *out, hex ? 16 : 10,
                             &end) &&
         *end == 0;
}

int HexDigit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

bool Encode(Ref type, Ref value, base::Vector<u8>* out) {
  struct Int {
    const char* name;
    u32 bytes;
    bool big;
  };
  static constexpr Int kInts[] = {{"byte", 1, false}, {"le16", 2, false},
                                  {"le32", 4, false}, {"le64", 8, false},
                                  {"be16", 2, true},  {"be32", 4, true},
                                  {"be64", 8, true}};
  auto put = [&](u64 v, u32 bytes, bool big) {
    for (u32 i = 0; i < bytes; i++)
      out->push_back(static_cast<u8>(v >> (8 * (big ? bytes - 1 - i : i))));
  };
  for (const Int& t : kInts) {
    if (type != Ref(t.name))
      continue;
    u64 v = 0;
    if (!ParseNumber(value, &v) || (t.bytes < 8 && v >> (t.bytes * 8)))
      return false;
    put(v, t.bytes, t.big);
    return true;
  }
  if (type == Ref("lef32") || type == Ref("lef64")) {
    const base::String s(value.data(), value.size());
    f64 d = 0;
    const char* end = nullptr;
    if (!base::ParseFloat(s.c_str(), d, &end) || *end != 0)
      return false;
    if (type == Ref("lef32")) {
      const f32 f = static_cast<f32>(d);
      u32 bits;
      std::memcpy(&bits, &f, sizeof(bits));
      put(bits, 4, false);
    } else {
      u64 bits;
      std::memcpy(&bits, &d, sizeof(bits));
      put(bits, 8, false);
    }
    return true;
  }
  if (type == Ref("utf8") || type == Ref("c_utf8")) {
    for (mem_size i = 0; i < value.size(); i++)
      out->push_back(static_cast<u8>(value[i]));
    if (type == Ref("c_utf8"))
      out->push_back(0);
    return true;
  }
  if (type == Ref("bytes")) {
    if (value.empty() || value.size() % 2)
      return false;
    for (mem_size i = 0; i < value.size(); i += 2) {
      const int hi = HexDigit(value[i]), lo = HexDigit(value[i + 1]);
      if (hi < 0 || lo < 0)
        return false;
      out->push_back(static_cast<u8>(hi << 4 | lo));
    }
    return true;
  }
  return false;
}

// "[ type, address, value ]"
const char* ParseEntry(Ref item, Write* out) {
  if (item.size() < 2 || item[0] != '[' || item[item.size() - 1] != ']')
    return "expected '- [ type, address, value ]'";
  Ref rest = item.substr(1, item.size() - 2);
  Ref fields[3];
  for (int i = 0; i < 3; i++) {
    const mem_size comma = i < 2 ? FindUnquoted(rest, ',') : Ref::npos;
    if (i < 2 && comma == Ref::npos)
      return "an entry needs a type, an address and a value";
    fields[i] = Trim(rest.substr(0, comma));
    if (comma != Ref::npos)
      rest = rest.substr(comma + 1);
  }
  if (FindUnquoted(fields[2], ',') != Ref::npos)
    return "an entry has exactly three fields";
  for (Ref& field : fields)
    field = Unquote(field);
  if (!ParseNumber(fields[1], &out->address))
    return "bad address";
  if (!Encode(fields[0], fields[2], &out->bytes))
    return "unknown type or bad value";
  return nullptr;
}

}  // namespace

bool ParsePatchFile(Ref text, base::Vector<Patch>* out, base::String* error) {
  base::String image;
  mem_size patch_indent = Ref::npos, field_indent = Ref::npos;
  bool in_list = false;
  mem_size line_no = 0;
  auto fail = [&](const char* why) {
    error->clear();
    base::FormatTo(*error, "line {}: {}", line_no, why);
    return false;
  };
  for (mem_size pos = 0; pos < text.size();) {
    mem_size nl = text.find('\n', pos);
    if (nl == Ref::npos)
      nl = text.size();
    const Ref raw = text.substr(pos, nl - pos);
    pos = nl + 1;
    line_no++;
    mem_size indent = 0;
    while (indent < raw.size() && raw[indent] == ' ')
      indent++;
    const Ref line = Trim(StripComment(raw));
    if (line.empty())
      continue;
    if (raw[indent] == '\t')
      return fail("indent with spaces, not tabs");
    Ref key, value;
    if (indent == 0) {
      if (!SplitKey(line, &key, &value))
        return fail("expected 'key:'");
      image = key == Ref("Version") ? base::String()
                                    : base::String(key.data(), key.size());
      patch_indent = field_indent = Ref::npos;
      in_list = false;
      continue;
    }
    if (image.empty())
      return fail("entry outside an executable block");
    if (patch_indent == Ref::npos || indent == patch_indent) {
      if (!SplitKey(line, &key, &value) || !value.empty())
        return fail("expected a patch name");
      patch_indent = indent;
      field_indent = Ref::npos;
      in_list = false;
      out->push_back({image, base::String(key.data(), key.size()), {}});
      continue;
    }
    if (indent < patch_indent)
      return fail("bad indentation");
    if (field_indent == Ref::npos || indent == field_indent) {
      if (!SplitKey(line, &key, &value))
        return fail("expected 'key:'");
      field_indent = indent;
      in_list = key == Ref("Patch");
      if (in_list && !value.empty())
        return fail("Patch takes a list of entries");
      continue;
    }
    if (indent < field_indent)
      return fail("bad indentation");
    if (!in_list)
      continue;
    if (line[0] != '-')
      return fail("expected '- [ type, address, value ]'");
    Write write{};
    if (const char* why = ParseEntry(Trim(line.substr(1)), &write))
      return fail(why);
    out->back().writes.push_back(base::move(write));
  }
  return true;
}

bool ApplyPatch(const Patch& patch, uintptr_t base, u64 end) {
  for (const Write& w : patch.writes)
    if (w.address > end || w.bytes.size() > end - w.address)
      return false;
  for (const Write& w : patch.writes)
    std::memcpy(reinterpret_cast<void*>(base + w.address), w.bytes.data(),
                w.bytes.size());
  return true;
}

}  // namespace patch
