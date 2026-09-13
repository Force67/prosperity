
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Ported from tools/pkg_extract/pkg_extract.py: decrypt the PFS of a fake-signed
// .pkg and inflate the inner PFSC image, exposing files on demand.

#include "pkg_object.h"
#include "base/arch.h"

#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <zlib.h>

#include <mbedtls/aes.h>
#include <mbedtls/bignum.h>
#include <mbedtls/md.h>
#include <mbedtls/sha256.h>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>

#include <logger/logger.h>
#include <utl/file.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kPfsDbg, "DELTA_PFS_DBG", nullptr);
DELTA_OPTION(const char *, kQarSelftest, "DELTA_QAR_SELFTEST", nullptr);
DELTA_OPTION(bool, kPkgReadTrace, "DELTA_PKGREAD_TRACE", false);
}  // namespace

namespace vfs {
namespace {
// Public fake-pkg keysets (from LibOrbisPkg): RSA-2048 modulus + private exp.
static const char *kNFake =
    "c6cf71e7e59af0d12a2c458bf92a0ec143058bc37117801dcd497dde359d259b"
    "a0d7a0f27d6c087eaa5502682b23c644b84418eb56cf16a24803c9e74f87eb3d"
    "30c31588bf20e79dff770cde1d241e63a94f8abf5bbe601968333bfced9f474e"
    "5ff8eacb3d00bd6701f92c6dc6ac1364e76714f3dc52696ab9832c4230131bb2"
    "d8a5020d79ed96b10df8cc0cdf81954f035809570e80692efeff5277ea7528a8"
    "fbc9bebf9fbbb7798e1805e180bd50349481d353c269a2d24ccf6cf4572c104a"
    "3ffb22fd8b97e2c95ba62bcdd61b6bdb687f4bc2a05034c005e58def2467ff93"
    "40cf2d62a2a050b1f13aa83dfd80d1f9b80522afc8354590588ee33a7cbd3e27";
static const char *kDFake =
    "7f76cd0ee2d4de051cc6d9a80e8dfa7bca1eaa271a40f8f1228735dddbfdeef8"
    "c2bcbd01fb8be23e63b2b1225c56496e11be07440b9a2666d1492c8fd31bcfa4"
    "a1b8d1fba49ed2212883098af6a00ba3d60f9b6368ccbc0c4e145b27a4a9f42b"
    "b9b87bc0e651ad1d77d46bb9ce20d126667e5e9ea2e96b90f373b8528f441103"
    "0c1397393d132258d5438249da6e7ca1c58ca5b009e0ce3ddff49d3c9715e26a"
    "c72b3c509323dbba4a226644ac78bb0e1a2743b57167aff4ab48469373d042ab"
    "9363e56c9ade5024c0237d99793f2207e0c148561bdf830912b42d456bc9c068"
    "85999079961ad7f54d1f3783404aec3937a680927dc580c7d66ffe8a7989c6b1";
static const char *kNDk3 =
    "d212fc335f6ddb831609628b0356273782d477853529392d526b8c4c8cfb06c1"
    "845be7d4f7bcd24e6245cd2abbd77776453655273fb3f5f98eda4befaa59aeb3"
    "9bea5498d206326a58312ae0d44f90b50a7decf43a9c52672d99318e0c43e682"
    "fe0746e12e50d41f2d2f7ed908ba06b3bf2e203f4e3ffe44ffaa504357916994"
    "49158282e40f4c8d9d2cc95b1d64bf888bd4c594e76547841ee57910fb989347"
    "b97d8512a640982cf792bc951932ede890560d65c1aa78c62e54fd5f54a1f67e"
    "e5e05f61c120b4b9b4330870e4df8956ed012946775f8cb8a9f51e2eb3b9bfe0"
    "09b78d28d4a6c3b81e1f07ebb4120b95b88530fddc3913d07cdc8fedf9c9a3c1";
static const char *kDDk3 =
    "32d903908fbdb08f572b285e0b8db3ea5cd17ea890888cdd6a80bbb1dfc1f70d"
    "aa32f0b77ccb88800e8b64b0be4cd60e9b8c1e2a64e1f35cd77601415e935c94"
    "fedd4662c31b5ae2a0bc2debc3980aa7b7856970682b644ab31fcc7ddc7c26f4"
    "77f65cf2ae5a442dd3ab16620419bafb90ffe23050896ecb56b2ebc09116925e"
    "308eaec7945dfd35e120f8ad3ebc08bfc036749fd5bb5208fd0666f37ab304f4"
    "75295de95faa1030b20f5a1ac12ab3fecb21ad80ec8f20091cdbc55894c29cc6"
    "ce82653e5790bca98b06b4f072f677df9864f1ecfe372dbcae8c08811fc3c989"
    "1ac742824b2edc8e8d73ceb1cc01d90870873c4408ec498f815ae240ff77fc0d";

inline u16 rd16(const u8 *p) {
  return static_cast<u16>(p[0] | (p[1] << 8));
}
inline u32 rd32(const u8 *p) {
  return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
         (static_cast<u32>(p[2]) << 16) |
         (static_cast<u32>(p[3]) << 24);
}
inline u64 rd64(const u8 *p) {
  u64 v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<u64>(p[i]) << (8 * i);
  return v;
}
inline u32 be32(const u8 *p) {
  return (static_cast<u32>(p[0]) << 24) |
         (static_cast<u32>(p[1]) << 16) |
         (static_cast<u32>(p[2]) << 8) | p[3];
}

// RSA-2048: m = ct^D mod N, then strip PKCS#1 v1.5 type-2 padding (like the
// Python _rsa()). Returns the recovered payload, or empty on bad padding.
std::vector<u8> rsaDecrypt(const u8 *ct, size_t ctLen, const char *nHex,
                                const char *dHex) {
  std::vector<u8> result;
  mbedtls_mpi N, D, C, M;
  mbedtls_mpi_init(&N);
  mbedtls_mpi_init(&D);
  mbedtls_mpi_init(&C);
  mbedtls_mpi_init(&M);

  u8 m[0x100];
  bool ok = mbedtls_mpi_read_string(&N, 16, nHex) == 0 &&
            mbedtls_mpi_read_string(&D, 16, dHex) == 0 &&
            mbedtls_mpi_read_binary(&C, ct, ctLen) == 0 &&
            mbedtls_mpi_exp_mod(&M, &C, &D, &N, nullptr) == 0 &&
            mbedtls_mpi_write_binary(&M, m, sizeof(m)) == 0;

  mbedtls_mpi_free(&N);
  mbedtls_mpi_free(&D);
  mbedtls_mpi_free(&C);
  mbedtls_mpi_free(&M);

  if (!ok || m[0] != 0 || m[1] != 2)
    return result;
  size_t sep = 2;
  while (sep < sizeof(m) && m[sep] != 0)
    ++sep;
  if (sep >= sizeof(m))
    return result;
  result.assign(m + sep + 1, m + sizeof(m));
  return result;
}

void sha256(const u8 *in, size_t len, u8 out[32]) {
  mbedtls_sha256_ret(in, len, out, 0);
}

void hmacSha256(const u8 *key, size_t keyLen, const u8 *in,
                size_t len, u8 out[32]) {
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  mbedtls_md_hmac(info, key, keyLen, in, len, out);
}

void aes128CbcDecrypt(const u8 key[16], const u8 iv[16],
                      const u8 *in, size_t len, u8 *out) {
  mbedtls_aes_context c;
  mbedtls_aes_init(&c);
  mbedtls_aes_setkey_dec(&c, key, 128);
  u8 ivCopy[16];
  std::memcpy(ivCopy, iv, 16);
  mbedtls_aes_crypt_cbc(&c, MBEDTLS_AES_DECRYPT, len, ivCopy, in, out);
  mbedtls_aes_free(&c);
}

// ----------------------------------------------------------------------------
// Lazy read chain. Each layer reads exactly [off, off+len) from the layer below
// and writes len bytes into dst.
// ----------------------------------------------------------------------------
struct DataSource {
  virtual ~DataSource() = default;
  virtual void read(u64 off, u64 len, u8 *dst) = 0;
};

// Bottom of the chain: a window into the host pkg file at a fixed base offset.
class RawSource : public DataSource {
public:
  RawSource(utl::File &f, u64 base) : f_(f), base_(base) {}
  void read(u64 off, u64 len, u8 *dst) override {
    std::memset(dst, 0, len); // zero-fill short reads past EOF
    f_.Seek(base_ + off, utl::seekMode::seek_set);
    f_.Read(dst, len);
  }

private:
  utl::File &f_;
  u64 base_;
};

class SubSource : public DataSource {
public:
  SubSource(DataSource *inner, u64 off) : inner_(inner), off_(off) {}
  void read(u64 off, u64 len, u8 *dst) override {
    inner_->read(off_ + off, len, dst);
  }

private:
  DataSource *inner_;
  u64 off_;
};

// AES-XTS over 0x1000-byte sectors. Sectors below skipSectors are plaintext
// (the PFS superblock region).
class XtsSource : public DataSource {
public:
  XtsSource(DataSource *inner, const u8 tweakKey[16],
            const u8 dataKey[16], u32 skipSectors)
      : inner_(inner), skip_(skipSectors) {
    mbedtls_aes_init(&tweak_);
    mbedtls_aes_init(&data_);
    mbedtls_aes_setkey_enc(&tweak_, tweakKey, 128);
    mbedtls_aes_setkey_dec(&data_, dataKey, 128);
  }
  ~XtsSource() override {
    mbedtls_aes_free(&tweak_);
    mbedtls_aes_free(&data_);
  }

  void read(u64 off, u64 len, u8 *dst) override {
    u8 sector[0x1000];
    u64 p = off, end = off + len;
    while (p < end) {
      u64 si = p / 0x1000, so = p % 0x1000;
      u64 take = std::min<u64>(0x1000 - so, end - p);
      inner_->read(si * 0x1000, 0x1000, sector);
      if (si >= skip_)
        decryptSector(si, sector);
      std::memcpy(dst, sector + so, take);
      dst += take;
      p += take;
    }
  }

private:
  void decryptSector(u64 si, u8 *buf) {
    u8 seed[16] = {0};
    for (int i = 0; i < 8; ++i)
      seed[i] = static_cast<u8>(si >> (8 * i));
    u8 tweak[16];
    mbedtls_aes_crypt_ecb(&tweak_, MBEDTLS_AES_ENCRYPT, seed, tweak);
    for (u32 b = 0; b < 0x1000; b += 16) {
      u8 x[16], p[16];
      for (int k = 0; k < 16; ++k)
        x[k] = buf[b + k] ^ tweak[k];
      mbedtls_aes_crypt_ecb(&data_, MBEDTLS_AES_DECRYPT, x, p);
      for (int k = 0; k < 16; ++k)
        buf[b + k] = p[k] ^ tweak[k];
      u8 carry = 0;
      for (int k = 0; k < 16; ++k) {
        u8 v = tweak[k];
        tweak[k] = static_cast<u8>((v << 1) | carry);
        carry = v >> 7;
      }
      if (carry)
        tweak[0] ^= 0x87;
    }
  }

  DataSource *inner_;
  mbedtls_aes_context tweak_;
  mbedtls_aes_context data_;
  u32 skip_;
};

// zlib-compressed inner image with a per-block offset map.
class PfscSource : public DataSource {
public:
  explicit PfscSource(DataSource *inner) : inner_(inner) {
    u8 h[0x30];
    inner_->read(0, sizeof(h), h);
    bs_ = rd64(h + 0x10);
    u64 bo = rd64(h + 0x18);
    u64 dl = rd64(h + 0x28);
    if (kPfsDbg)
      BASE_LOGI("pfsc", "magic={:02x}{:02x}{:02x}{:02x} bs={} bo={} dl={}",
                h[0], h[1], h[2], h[3], (unsigned long long)bs_,
                (unsigned long long)bo, (unsigned long long)dl);
    if (bs_ == 0)
      return;
    n_ = dl / bs_;
    map_.resize(n_ + 1);
    std::vector<u8> raw((n_ + 1) * 8);
    inner_->read(bo, raw.size(), raw.data());
    for (u64 i = 0; i <= n_; ++i)
      map_[i] = rd64(raw.data() + i * 8);
  }

  void read(u64 off, u64 len, u8 *dst) override {
    std::vector<u8> block(bs_);
    std::vector<u8> comp;
    while (len > 0) {
      u64 bi = off / bs_, bo = off % bs_;
      if (bi + 1 > n_) { // out of range
        std::memset(dst, 0, len);
        return;
      }
      u64 a = map_[bi], b = map_[bi + 1];
      u64 clen = b - a;
      comp.resize(clen);
      inner_->read(a, clen, comp.data());
      if (clen == bs_) {
        std::memcpy(block.data(), comp.data(), bs_);
      } else {
        uLongf out = static_cast<uLongf>(bs_);
        uncompress(block.data(), &out, comp.data(),
                   static_cast<uLong>(clen));
      }
      u64 take = std::min<u64>(bs_ - bo, len);
      std::memcpy(dst, block.data() + bo, take);
      dst += take;
      off += take;
      len -= take;
    }
  }

private:
  DataSource *inner_;
  u64 bs_ = 0;
  u64 n_ = 0;
  std::vector<u64> map_;
};
} // namespace

// ----------------------------------------------------------------------------
struct PkgImpl {
  utl::File pkg;
  std::mutex io;
  std::vector<std::unique_ptr<DataSource>> nodes;
  DataSource *inner = nullptr;
  u32 innerBs = 0;
  std::unordered_map<std::string, PkgFilesystem::Node> files;
  // lowercased path -> canonical key in `files`. PFS lookups on the console are
  // case-insensitive for app content; titles rely on it (SotC opens
  // "pakn.psarc" for a shipped "pakN.psarc").
  std::unordered_map<std::string, std::string> filesCI;
  bool valid = false;

  explicit PkgImpl(const base::String &path) : pkg(path, utl::fileMode::read) {
    if (!pkg.Exists() || !pkg.IsOpen()) {
      LOG_ERROR("pkg: cannot open {}", path.c_str());
      return;
    }
    build();
  }

  template <typename T, typename... Args> T *make(Args &&...args) {
    auto p = std::make_unique<T>(std::forward<Args>(args)...);
    T *raw = p.get();
    nodes.emplace_back(std::move(p));
    return raw;
  }

  bool getEkpfs(u8 out[32]) {
    auto R = [&](u64 o, u64 n, u8 *dst) {
      pkg.Seek(o, utl::seekMode::seek_set);
      pkg.Read(dst, n);
    };
    u8 tmp[4];
    R(0x10, 4, tmp);
    u32 ec = be32(tmp);
    R(0x18, 4, tmp);
    u32 to = be32(tmp);
    if (ec == 0 || ec > 0x10000)
      return false;

    std::vector<u8> tb(static_cast<size_t>(ec) * 0x20);
    R(to, tb.size(), tb.data());

    const u8 *row20 = nullptr;
    u32 ikOff = 0, ikSz = 0, ekOff = 0, ekSz = 0;
    for (u32 i = 0; i < ec; ++i) {
      const u8 *e = tb.data() + i * 0x20;
      u32 id = be32(e);
      if (id == 0x20) {
        row20 = e;
        ikOff = be32(e + 16);
        ikSz = be32(e + 20);
      } else if (id == 0x10) {
        ekOff = be32(e + 16);
        ekSz = be32(e + 20);
      }
    }
    if (!row20 || !ekOff || ekSz < 0x500 || ikSz < 0x100 || (ikSz % 16))
      return false;

    std::vector<u8> ek(ekSz);
    R(ekOff, ekSz, ek.data());
    auto dk3 = rsaDecrypt(ek.data() + 0x400, 0x100, kNDk3, kDDk3);
    if (dk3.empty())
      return false;

    std::vector<u8> ivkIn(row20, row20 + 0x20);
    ivkIn.insert(ivkIn.end(), dk3.begin(), dk3.end());
    u8 ivk[32];
    sha256(ivkIn.data(), ivkIn.size(), ivk);

    std::vector<u8> ik(ikSz), imdec(ikSz);
    R(ikOff, ikSz, ik.data());
    aes128CbcDecrypt(ivk + 16, ivk, ik.data(), ikSz, imdec.data());

    auto ekpfs = rsaDecrypt(imdec.data(), imdec.size(), kNFake, kDFake);
    if (ekpfs.size() < 32)
      return false;
    std::memcpy(out, ekpfs.data(), 32);
    return true;
  }

  // Parse a PFS superblock from `raw` and, if encrypted, wrap it in XTS. Returns
  // the reader to use for inode/data reads. signedFlag/nd/ndb come back via out
  // params.
  DataSource *buildPfs(DataSource *raw, const u8 ekpfs[32], u32 &bs,
                       bool &signedFlag, u64 &nd, u64 &ndb) {
    u8 h[0x400];
    raw->read(0, sizeof(h), h);
    bs = rd32(h + 0x20);
    u16 mode = rd16(h + 0x1c);
    nd = rd64(h + 0x30);
    ndb = rd64(h + 0x40);
    signedFlag = mode & 1;

    DataSource *reader = raw;
    if (mode & 4) {
      u8 msg[20] = {1, 0, 0, 0};
      std::memcpy(msg + 4, h + 0x370, 16); // seed
      u8 d[32];
      hmacSha256(ekpfs, 32, msg, sizeof(msg), d);
      reader = make<XtsSource>(raw, d, d + 16, bs / 0x1000);
    }
    return reader;
  }

  struct Inode {
    u16 type;
    u64 size;
    u32 blocks;
    u32 start;
  };
  std::vector<Inode> ino;

  void readInodes(DataSource *r, u32 bs, bool signedFlag, u64 nd,
                  u64 ndb) {
    u32 dsz = signedFlag ? 0x2C8 : 0xA8;
    u32 per = bs / dsz;
    std::vector<u8> blk(bs);
    u64 tot = 0;
    u64 count = ndb < 1 ? 1 : ndb;
    for (u64 bi = 0; bi < count && tot < nd; ++bi) {
      r->read((1 + bi) * bs, bs, blk.data());
      for (u32 j = 0; j < per && tot < nd; ++j) {
        const u8 *d = blk.data() + j * dsz;
        Inode n;
        n.type = rd16(d);
        n.size = rd64(d + 8);
        n.blocks = rd32(d + 0x60);
        n.start = signedFlag ? rd32(d + 0x84) : rd32(d + 0x64);
        ino.push_back(n);
        ++tot;
      }
    }
  }

  // Find a direct child directory of dirInode by name; -1 if absent.
  i64 findChildDir(u32 dirInode, const char *name, DataSource *r,
                       u32 bs) {
    if (dirInode >= ino.size())
      return -1;
    u32 blocks = ino[dirInode].blocks, st = ino[dirInode].start;
    std::vector<u8> d(bs);
    for (u32 bb = 0; bb < blocks; ++bb) {
      r->read((u64)(st + bb) * bs, bs, d.data());
      u64 o = 0;
      while (o + 17 < bs) {
        u32 ch = rd32(d.data() + o);
        u32 ty = rd32(d.data() + o + 4);
        u32 nl = rd32(d.data() + o + 8);
        u32 es = rd32(d.data() + o + 12);
        if (es == 0)
          break;
        if (ty == 3 && nl > 0 && nl < 256 && o + 16 + nl <= bs) {
          std::string nm(reinterpret_cast<const char *>(d.data() + o + 16), nl);
          if (nm == name && ch < ino.size())
            return static_cast<i64>(ch);
        }
        o += es;
      }
    }
    return -1;
  }

  void walk(u32 i, const std::string &pre, DataSource *r, u32 bs) {
    if (i >= ino.size())
      return;
    u32 blocks = ino[i].blocks, st = ino[i].start;
    std::vector<u8> d(bs);
    for (u32 bb = 0; bb < blocks; ++bb) {
      r->read((u64)(st + bb) * bs, bs, d.data());
      u64 o = 0;
      while (o + 17 < bs) {
        u32 ch = rd32(d.data() + o);
        u32 ty = rd32(d.data() + o + 4);
        u32 nl = rd32(d.data() + o + 8);
        u32 es = rd32(d.data() + o + 12);
        if (es == 0)
          break;
        if (nl > 0 && nl < 256 && o + 16 + nl <= bs) {
          std::string nm(reinterpret_cast<const char *>(d.data() + o + 16), nl);
          if (ty == 2 && ch < ino.size()) {
            files[pre + "/" + nm] = {ino[ch].size, ino[ch].start};
               if (const char *dbg = kPfsDbg)
                 if (nm.find(dbg) != std::string::npos)
                   BASE_LOGI("pfs", "{}/{} size={} start={} blocks={} "
                            "blocks*bs={} bs={}",
                            pre.c_str(), nm.c_str(),
                            (unsigned long long)ino[ch].size, ino[ch].start,
                            ino[ch].blocks,
                            (unsigned long long)ino[ch].blocks * bs, bs);
          } else if (ty == 3 && nm != "." && nm != "..") {
            walk(ch, pre + "/" + nm, r, bs);
          }
        }
        o += es;
      }
    }
  }

  void build() {
    u8 ekpfs[32];
    if (!getEkpfs(ekpfs)) {
      LOG_ERROR("pkg: could not recover EKPFS (not a fake pkg?)");
      return;
    }

    // Standard fpkg PFS image offset. Read the header field when it looks sane,
    // else fall back to the well-known constant pkg_extract.py hardcodes.
    u8 off[8];
    pkg.Seek(0x410, utl::seekMode::seek_set);
    pkg.Read(off, 8);
    u64 pfsOff = (static_cast<u64>(be32(off)) << 32) | be32(off + 4);
    if (pfsOff < 0x1000 || pfsOff >= pkg.GetSize())
      pfsOff = 0x700000;

    // Outer PFS: only its block size and (XTS) reader are needed.
    auto *raw = make<RawSource>(pkg, pfsOff);
    u32 outerBs = 0;
    bool outerSigned = false;
    u64 outerNd = 0, outerNdb = 0;
    DataSource *outer =
        buildPfs(raw, ekpfs, outerBs, outerSigned, outerNd, outerNdb);
    if (outerBs == 0)
      return;

    // The inner image is a contiguous, PFSC-compressed stream inside the outer
    // PFS. Its start block depends on the size of the outer metadata (superblock
    // + inodes + dirents + flat_path_table), so locate it by its "PFSC" magic
    // rather than assuming a fixed block. (Isaac lands at block 11, P.T. at 19.)
    u64 innerBlk = 11;
    bool found = false;
    u64 maxBlk = (pkg.GetSize() - pfsOff) / outerBs;
    for (u64 bi = 1; bi < maxBlk; ++bi) {
      u8 magic[4];
      outer->read(bi * outerBs, sizeof(magic), magic);
      if (magic[0] == 'P' && magic[1] == 'F' && magic[2] == 'S' && magic[3] == 'C') {
        innerBlk = bi;
        found = true;
        break;
      }
    }
    if (kPfsDbg)
      BASE_LOGI("pkg", "pfsOff={} outerBs={} PFSC scan: found={} "
               "innerBlk={} maxBlk={}",
               (unsigned long long)pfsOff, outerBs, found,
               (unsigned long long)innerBlk, (unsigned long long)maxBlk);
    auto *sub = make<SubSource>(outer, innerBlk * outerBs);
    auto *pfsc = make<PfscSource>(sub);

    bool innerSigned = false;
    u64 innerNd = 0, innerNdb = 0;
    inner = buildPfs(pfsc, ekpfs, innerBs, innerSigned, innerNd, innerNdb);
    if (innerBs == 0)
      return;

    readInodes(inner, innerBs, innerSigned, innerNd, innerNdb);
    if (ino.empty())
      return;

    // The PFS superroot holds `flat_path_table` + `uroot`; the game files live
    // under uroot. Present them game-relative (uroot == /app0) by walking from
    // there, falling back to the raw root if the image is laid out flat.
    i64 root = findChildDir(0, "uroot", inner, innerBs);
    walk(root >= 0 ? static_cast<u32>(root) : 0, "", inner, innerBs);

    for (const auto &kv : files) {
      std::string lower(kv.first);
      for (auto &c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      filesCI.emplace(std::move(lower), kv.first);
    }

    valid = true;
    LOG_INFO("pkg: mounted image, {} files", files.size());

    // DELTA_QAR_SELFTEST: dump the bytes our read chain returns for a named file
    // at a spread of offsets, so it can be diffed against an independent
    // ground-truth extraction (validates large-file decrypt/decompress). Reads
    // the file whose path contains the env value (default "texture.qar").
    if (const char *st = kQarSelftest) {
      const char *want = *st ? st : "texture.qar";
      const PkgFilesystem::Node *node = nullptr;
      std::string nodePath;
      for (const auto &kv : files)
        if (kv.first.find(want) != std::string::npos) {
          node = &kv.second;
          nodePath = kv.first;
          break;
        }
      if (node) {
         BASE_LOGI("qarself", "{} size={} startBlock={} innerBs={}",
                   nodePath.c_str(), (unsigned long long)node->size,
                   node->startBlock, innerBs);
        u64 offs[] = {0, 1ull << 20, 100ull << 20, 400ull << 20,
                           800ull << 20,
                           node->size > 256 ? node->size - 256 : 0};
        for (u64 o : offs) {
          if (o >= node->size)
            continue;
          u8 buf[64] = {0};
          i64 r = readNode(*node, buf, static_cast<i64>(o), 64);
          base::String bytes;
          base::FormatTo(bytes, "off={} r={}:",
                         (unsigned long long)o, (long long)r);
          for (int i = 0; i < 16 && i < r; ++i)
            base::FormatTo(bytes, " {:02x}", buf[i]);
          base::FormatTo(bytes, " |");
          for (int i = 16; i < 32 && i < r; ++i)
            base::FormatTo(bytes, " {:02x}", buf[i]);
          BASE_LOGI("qarself", "{}", bytes.c_str());
        }
       } else {
         BASE_LOGI("qarself", "no file matching '{}'", want);
       }
    }
  }

  i64 readNode(const PkgFilesystem::Node &n, void *buf, i64 off,
                   i64 len) {
    if (off < 0 || len < 0)
      return -1;
    if (static_cast<u64>(off) >= n.size)
      return 0;
    u64 take =
        std::min<u64>(len, n.size - static_cast<u64>(off));
    std::lock_guard<std::mutex> lk(io);
    u64 innerOff =
        static_cast<u64>(n.startBlock) * innerBs + static_cast<u64>(off);
    inner->read(innerOff, take, static_cast<u8 *>(buf));
    if (kPkgReadTrace && take >= 1) {
      auto *b = static_cast<const u8 *>(buf);
      u32 w = b[0] | (take > 1 ? b[1] << 8 : 0) | (take > 2 ? b[2] << 16 : 0) |
                   (take > 3 ? u32(b[3]) << 24 : 0);
      BASE_LOGI("pkgread", "blk={} off={} len={} -> {}  first4={:08x}",
                n.startBlock, (long long)off, (long long)len,
                (unsigned long long)take, w);
    }
    return static_cast<i64>(take);
  }

  // Outer-PKG entry straight from the header's entry table (plaintext in a fake pkg,
  // outside the PFS). Same big-endian layout getEkpfs walks: count @0x10, table @0x18,
  // 0x20-byte rows of [id @0, offset @16, size @20].
  bool readEntry(u32 wantId, std::vector<u8> &out) {
    if (!pkg.IsOpen())
      return false;
    std::lock_guard<std::mutex> lk(io);
    auto R = [&](u64 o, u64 n, u8 *dst) {
      pkg.Seek(o, utl::seekMode::seek_set);
      pkg.Read(dst, n);
    };
    u8 tmp[4];
    R(0x10, 4, tmp);
    u32 ec = be32(tmp);
    R(0x18, 4, tmp);
    u32 to = be32(tmp);
    if (ec == 0 || ec > 0x10000)
      return false;
    std::vector<u8> tb(static_cast<size_t>(ec) * 0x20);
    R(to, tb.size(), tb.data());
    for (u32 i = 0; i < ec; ++i) {
      const u8 *e = tb.data() + i * 0x20;
      if (be32(e) != wantId)
        continue;
      u32 off = be32(e + 16), sz = be32(e + 20);
      const u32 maxSize =
          wantId == 0x1000 ? 1u << 20
                           : (wantId == 0x1200 ? 16u << 20 : UINT32_MAX);
      if (sz == 0 || sz > maxSize ||
          static_cast<u64>(off) + sz > pkg.GetSize())
        return false;
      out.resize(sz);
      R(off, sz, out.data());
      return true;
    }
    return false;
  }
};

PkgFilesystem::PkgFilesystem(const base::String &pkgPath)
    : impl_(std::make_unique<PkgImpl>(pkgPath)) {}
PkgFilesystem::~PkgFilesystem() = default;

bool PkgFilesystem::valid() const { return impl_ && impl_->valid; }

const PkgFilesystem::Node *PkgFilesystem::find(const char *relPath) const {
  if (!impl_ || !relPath)
    return nullptr;
  // Normalize to one leading slash, collapse repeats: titles concatenate paths and
  // emit doubled separators (Doom64 opens "/app0//DOOMSND.DLS"); the table is keyed
  // on the clean path and POSIX treats // as /, so an exact match must too (else the
  // soundfont open ENOENTs and FMOD aborts the boot).
  std::string key;
  key.reserve(std::strlen(relPath) + 1);
  key.push_back('/');
  for (const char *p = relPath; *p; ++p) {
    if (*p == '/' && (key.empty() || key.back() == '/'))
      continue;
    key.push_back(*p);
  }
  auto it = impl_->files.find(key);
  if (it != impl_->files.end())
    return &it->second;

  // Case-insensitive fallback (PFS app-content semantics): exact match wins,
  // otherwise resolve through the lowercased index.
  std::string lower(key);
  for (auto &c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  auto ci = impl_->filesCI.find(lower);
  if (ci != impl_->filesCI.end()) {
    it = impl_->files.find(ci->second);
    if (it != impl_->files.end())
      return &it->second;
  }
  return nullptr;
}

i64 PkgFilesystem::read(const Node &node, void *buf, i64 off,
                            i64 len) {
  return impl_->readNode(node, buf, off, len);
}

void PkgFilesystem::paths(std::vector<std::string> &out) const {
  if (!impl_)
    return;
  out.reserve(impl_->files.size());
  for (const auto &kv : impl_->files)
    out.push_back(kv.first);
}

i64 PkgFilesystem::readPkgEntry(u32 entryId, std::vector<u8> &out) {
  if (!impl_)
    return -1;
  return impl_->readEntry(entryId, out) ? static_cast<i64>(out.size()) : -1;
}
} // namespace vfs
