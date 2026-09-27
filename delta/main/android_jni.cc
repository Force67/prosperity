/*
 * PS4Delta : PS4 emulation and research project
 *
 * JNI bridge for the Android launcher (DELTA_ANDROID_APP). The Java
 * LauncherActivity loads libps4delta_app.so and calls these to read pkg
 * metadata (param.sfo title / title-id, icon0.png cover art) for the game
 * library, and to best-effort unpack a firmware .PUP. The emulator itself is
 * still entered through ANativeActivity_onCreate (android_main.cc); this file
 * only adds utility entry points reused by the launcher UI.
 */
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)

#include <jni.h>
#include "base/arch.h"

#include <cstdio>
#include <cstring>

#include <base/strings/xstring.h>

#include "formats/pkg_filesystem.h"
#include "formats/pup_reader.h"
#include <base/containers/vector.h>

namespace {

// Minimal param.sfo reader: returns the value of `key` (UTF-8 string keys, or
// int32 keys rendered as decimal), or "" if absent. The SFO is a small flat
// table; see the PS4 param.sfo layout. All offsets are bounds-checked.
base::String SfoGet(const u8 *d, size_t n, const char *key) {
  if (n < 20)
    return {};
  auto rd32 = [&](size_t o) -> u32 {
    return u32(d[o]) | (u32(d[o + 1]) << 8) |
           (u32(d[o + 2]) << 16) | (u32(d[o + 3]) << 24);
  };
  auto rd16 = [&](size_t o) -> u16 {
    return u16(d[o] | (d[o + 1] << 8));
  };
  if (rd32(0) != 0x46535000u) // "\0PSF"
    return {};
  u32 keyStart = rd32(8), dataStart = rd32(12), count = rd32(16);
  size_t klen = std::strlen(key);
  size_t idx = 20;
  for (u32 i = 0; i < count; i++, idx += 16) {
    if (idx + 16 > n)
      break;
    u16 keyOff = rd16(idx);
    u16 fmt = rd16(idx + 2);
    u32 len = rd32(idx + 4);
    u32 dataOff = rd32(idx + 12);

    size_t kpos = size_t(keyStart) + keyOff;
    if (kpos + klen + 1 > n)
      continue;
    if (std::memcmp(d + kpos, key, klen) != 0 || d[kpos + klen] != '\0')
      continue;

    size_t dpos = size_t(dataStart) + dataOff;
    if (dpos >= n)
      return {};
    if (fmt == 0x0404) { // int32
      if (dpos + 4 > n)
        return {};
      u32 v = u32(d[dpos]) | (u32(d[dpos + 1]) << 8) |
                   (u32(d[dpos + 2]) << 16) | (u32(d[dpos + 3]) << 24);
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%u", v);
      return buf;
    }
    size_t avail = n - dpos;
    size_t l = len < avail ? len : avail;
    base::String s(reinterpret_cast<const char *>(d + dpos), l);
    while (!s.empty() && s.back() == '\0')
      s.pop_back();
    return s;
  }
  return {};
}

// PKG entry ids for the outer metadata the launcher surfaces.
constexpr u32 kEntryParamSfo = 0x1000;
constexpr u32 kEntryIcon0Png = 0x1200;

// Read a well-known outer-PKG metadata entry (param.sfo / icon0.png). These sit
// in the PKG header table, not the inner PFS, so they read even for pkgs whose
// PFS we don't fully mount.
bool readPkgMeta(const char *pkgPath, u32 entryId,
                 base::Vector<u8> &out) {
  formats::PkgFilesystem fs((base::String(pkgPath)));
  return fs.readPkgEntry(entryId, out) > 0;
}

} // namespace

extern "C" {

// Returns "<TITLE_ID>\t<TITLE>" parsed from /sce_sys/param.sfo, or "" when the
// pkg can't be read as a fake-pkg (retail/encrypted) so the launcher falls back
// to the file name.
JNIEXPORT jstring JNICALL Java_com_prosperity_ps4_NativeBridge_pkgInfo(
    JNIEnv *env, jclass, jstring jpath) {
  const char *path = env->GetStringUTFChars(jpath, nullptr);
  base::String result;
  base::Vector<u8> sfo;
  if (readPkgMeta(path, kEntryParamSfo, sfo)) {
    base::String tid = SfoGet(sfo.data(), sfo.size(), "TITLE_ID");
    base::String title = SfoGet(sfo.data(), sfo.size(), "TITLE");
    result = tid + "\t" + title;
  }
  env->ReleaseStringUTFChars(jpath, path);
  return env->NewStringUTF(result.c_str());
}

// Extracts /sce_sys/icon0.png to outPath for cover art. Returns true on success.
JNIEXPORT jboolean JNICALL Java_com_prosperity_ps4_NativeBridge_pkgIcon(
    JNIEnv *env, jclass, jstring jpath, jstring joutPath) {
  const char *path = env->GetStringUTFChars(jpath, nullptr);
  const char *out = env->GetStringUTFChars(joutPath, nullptr);
  bool ok = false;
  base::Vector<u8> png;
  if (readPkgMeta(path, kEntryIcon0Png, png)) {
    if (FILE *f = std::fopen(out, "wb")) {
      ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
      std::fclose(f);
    }
  }
  env->ReleaseStringUTFChars(jpath, path);
  env->ReleaseStringUTFChars(joutPath, out);
  return ok ? JNI_TRUE : JNI_FALSE;
}

// Best-effort firmware PUP unpack into outDir. Returns a human-readable summary
// (see PupReader::ExtractAll); retail PUPs are encrypted, so this can only dump
// container segments, never loadable modules.
JNIEXPORT jstring JNICALL Java_com_prosperity_ps4_NativeBridge_pupExtract(
    JNIEnv *env, jclass, jstring jpup, jstring jout) {
  const char *pup = env->GetStringUTFChars(jpup, nullptr);
  const char *out = env->GetStringUTFChars(jout, nullptr);
  base::String summary;
  formats::PupReader r((base::String(pup)));
  if (!r.load()) {
    summary = "Not a recognized PUP container (magic mismatch). Retail firmware "
              "is encrypted and unsupported here; import a pre-extracted .sprx "
              "module set instead.";
  } else {
    bool encrypted = false;
    summary = r.extractAll(base::String(out), encrypted);
  }
  env->ReleaseStringUTFChars(jpup, pup);
  env->ReleaseStringUTFChars(jout, out);
  return env->NewStringUTF(summary.c_str());
}

} // extern "C"

#endif // __ANDROID__ && DELTA_ANDROID_APP
