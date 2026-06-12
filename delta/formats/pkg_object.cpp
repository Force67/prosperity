
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

#include <logger/logger.h>
#include <utl/file.h>

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

inline uint16_t rd16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
inline uint32_t rd32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t rd64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}
inline uint32_t be32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

// RSA-2048: m = ct^D mod N, then strip PKCS#1 v1.5 type-2 padding (like the
// Python _rsa()). Returns the recovered payload, or empty on bad padding.
std::vector<uint8_t> rsaDecrypt(const uint8_t *ct, size_t ctLen, const char *nHex,
                                const char *dHex) {
  std::vector<uint8_t> result;
  mbedtls_mpi N, D, C, M;
  mbedtls_mpi_init(&N);
  mbedtls_mpi_init(&D);
  mbedtls_mpi_init(&C);
  mbedtls_mpi_init(&M);

  uint8_t m[0x100];
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

void sha256(const uint8_t *in, size_t len, uint8_t out[32]) {
  mbedtls_sha256_ret(in, len, out, 0);
}

void hmacSha256(const uint8_t *key, size_t keyLen, const uint8_t *in,
                size_t len, uint8_t out[32]) {
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  mbedtls_md_hmac(info, key, keyLen, in, len, out);
}

void aes128CbcDecrypt(const uint8_t key[16], const uint8_t iv[16],
                      const uint8_t *in, size_t len, uint8_t *out) {
  mbedtls_aes_context c;
  mbedtls_aes_init(&c);
  mbedtls_aes_setkey_dec(&c, key, 128);
  uint8_t ivCopy[16];
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
  virtual void read(uint64_t off, uint64_t len, uint8_t *dst) = 0;
};

// Bottom of the chain: a window into the host pkg file at a fixed base offset.
class RawSource : public DataSource {
public:
  RawSource(utl::File &f, uint64_t base) : f_(f), base_(base) {}
  void read(uint64_t off, uint64_t len, uint8_t *dst) override {
    std::memset(dst, 0, len); // zero-fill short reads past EOF
    f_.Seek(base_ + off, utl::seekMode::seek_set);
    f_.Read(dst, len);
  }

private:
  utl::File &f_;
  uint64_t base_;
};

class SubSource : public DataSource {
public:
  SubSource(DataSource *inner, uint64_t off) : inner_(inner), off_(off) {}
  void read(uint64_t off, uint64_t len, uint8_t *dst) override {
    inner_->read(off_ + off, len, dst);
  }

private:
  DataSource *inner_;
  uint64_t off_;
};

// AES-XTS over 0x1000-byte sectors. Sectors below skipSectors are plaintext
// (the PFS superblock region).
class XtsSource : public DataSource {
public:
  XtsSource(DataSource *inner, const uint8_t tweakKey[16],
            const uint8_t dataKey[16], uint32_t skipSectors)
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

  void read(uint64_t off, uint64_t len, uint8_t *dst) override {
    uint8_t sector[0x1000];
    uint64_t p = off, end = off + len;
    while (p < end) {
      uint64_t si = p / 0x1000, so = p % 0x1000;
      uint64_t take = std::min<uint64_t>(0x1000 - so, end - p);
      inner_->read(si * 0x1000, 0x1000, sector);
      if (si >= skip_)
        decryptSector(si, sector);
      std::memcpy(dst, sector + so, take);
      dst += take;
      p += take;
    }
  }

private:
  void decryptSector(uint64_t si, uint8_t *buf) {
    uint8_t seed[16] = {0};
    for (int i = 0; i < 8; ++i)
      seed[i] = static_cast<uint8_t>(si >> (8 * i));
    uint8_t tweak[16];
    mbedtls_aes_crypt_ecb(&tweak_, MBEDTLS_AES_ENCRYPT, seed, tweak);
    for (uint32_t b = 0; b < 0x1000; b += 16) {
      uint8_t x[16], p[16];
      for (int k = 0; k < 16; ++k)
        x[k] = buf[b + k] ^ tweak[k];
      mbedtls_aes_crypt_ecb(&data_, MBEDTLS_AES_DECRYPT, x, p);
      for (int k = 0; k < 16; ++k)
        buf[b + k] = p[k] ^ tweak[k];
      uint8_t carry = 0;
      for (int k = 0; k < 16; ++k) {
        uint8_t v = tweak[k];
        tweak[k] = static_cast<uint8_t>((v << 1) | carry);
        carry = v >> 7;
      }
      if (carry)
        tweak[0] ^= 0x87;
    }
  }

  DataSource *inner_;
  mbedtls_aes_context tweak_;
  mbedtls_aes_context data_;
  uint32_t skip_;
};

// zlib-compressed inner image with a per-block offset map.
class PfscSource : public DataSource {
public:
  explicit PfscSource(DataSource *inner) : inner_(inner) {
    uint8_t h[0x30];
    inner_->read(0, sizeof(h), h);
    bs_ = rd64(h + 0x10);
    uint64_t bo = rd64(h + 0x18);
    uint64_t dl = rd64(h + 0x28);
    if (bs_ == 0)
      return;
    n_ = dl / bs_;
    map_.resize(n_ + 1);
    std::vector<uint8_t> raw((n_ + 1) * 8);
    inner_->read(bo, raw.size(), raw.data());
    for (uint64_t i = 0; i <= n_; ++i)
      map_[i] = rd64(raw.data() + i * 8);
  }

  void read(uint64_t off, uint64_t len, uint8_t *dst) override {
    std::vector<uint8_t> block(bs_);
    std::vector<uint8_t> comp;
    while (len > 0) {
      uint64_t bi = off / bs_, bo = off % bs_;
      if (bi + 1 > n_) { // out of range
        std::memset(dst, 0, len);
        return;
      }
      uint64_t a = map_[bi], b = map_[bi + 1];
      uint64_t clen = b - a;
      comp.resize(clen);
      inner_->read(a, clen, comp.data());
      if (clen == bs_) {
        std::memcpy(block.data(), comp.data(), bs_);
      } else {
        uLongf out = static_cast<uLongf>(bs_);
        uncompress(block.data(), &out, comp.data(),
                   static_cast<uLong>(clen));
      }
      uint64_t take = std::min<uint64_t>(bs_ - bo, len);
      std::memcpy(dst, block.data() + bo, take);
      dst += take;
      off += take;
      len -= take;
    }
  }

private:
  DataSource *inner_;
  uint64_t bs_ = 0;
  uint64_t n_ = 0;
  std::vector<uint64_t> map_;
};
} // namespace

// ----------------------------------------------------------------------------
struct PkgImpl {
  utl::File pkg;
  std::mutex io;
  std::vector<std::unique_ptr<DataSource>> nodes;
  DataSource *inner = nullptr;
  uint32_t innerBs = 0;
  std::unordered_map<std::string, PkgFilesystem::Node> files;
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

  bool getEkpfs(uint8_t out[32]) {
    auto R = [&](uint64_t o, uint64_t n, uint8_t *dst) {
      pkg.Seek(o, utl::seekMode::seek_set);
      pkg.Read(dst, n);
    };
    uint8_t tmp[4];
    R(0x10, 4, tmp);
    uint32_t ec = be32(tmp);
    R(0x18, 4, tmp);
    uint32_t to = be32(tmp);
    if (ec == 0 || ec > 0x10000)
      return false;

    std::vector<uint8_t> tb(static_cast<size_t>(ec) * 0x20);
    R(to, tb.size(), tb.data());

    const uint8_t *row20 = nullptr;
    uint32_t ikOff = 0, ikSz = 0, ekOff = 0, ekSz = 0;
    for (uint32_t i = 0; i < ec; ++i) {
      const uint8_t *e = tb.data() + i * 0x20;
      uint32_t id = be32(e);
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

    std::vector<uint8_t> ek(ekSz);
    R(ekOff, ekSz, ek.data());
    auto dk3 = rsaDecrypt(ek.data() + 0x400, 0x100, kNDk3, kDDk3);
    if (dk3.empty())
      return false;

    std::vector<uint8_t> ivkIn(row20, row20 + 0x20);
    ivkIn.insert(ivkIn.end(), dk3.begin(), dk3.end());
    uint8_t ivk[32];
    sha256(ivkIn.data(), ivkIn.size(), ivk);

    std::vector<uint8_t> ik(ikSz), imdec(ikSz);
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
  DataSource *buildPfs(DataSource *raw, const uint8_t ekpfs[32], uint32_t &bs,
                       bool &signedFlag, uint64_t &nd, uint64_t &ndb) {
    uint8_t h[0x400];
    raw->read(0, sizeof(h), h);
    bs = rd32(h + 0x20);
    uint16_t mode = rd16(h + 0x1c);
    nd = rd64(h + 0x30);
    ndb = rd64(h + 0x40);
    signedFlag = mode & 1;

    DataSource *reader = raw;
    if (mode & 4) {
      uint8_t msg[20] = {1, 0, 0, 0};
      std::memcpy(msg + 4, h + 0x370, 16); // seed
      uint8_t d[32];
      hmacSha256(ekpfs, 32, msg, sizeof(msg), d);
      reader = make<XtsSource>(raw, d, d + 16, bs / 0x1000);
    }
    return reader;
  }

  struct Inode {
    uint16_t type;
    uint64_t size;
    uint32_t blocks;
    uint32_t start;
  };
  std::vector<Inode> ino;

  void readInodes(DataSource *r, uint32_t bs, bool signedFlag, uint64_t nd,
                  uint64_t ndb) {
    uint32_t dsz = signedFlag ? 0x2C8 : 0xA8;
    uint32_t per = bs / dsz;
    std::vector<uint8_t> blk(bs);
    uint64_t tot = 0;
    uint64_t count = ndb < 1 ? 1 : ndb;
    for (uint64_t bi = 0; bi < count && tot < nd; ++bi) {
      r->read((1 + bi) * bs, bs, blk.data());
      for (uint32_t j = 0; j < per && tot < nd; ++j) {
        const uint8_t *d = blk.data() + j * dsz;
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
  int64_t findChildDir(uint32_t dirInode, const char *name, DataSource *r,
                       uint32_t bs) {
    if (dirInode >= ino.size())
      return -1;
    uint32_t blocks = ino[dirInode].blocks, st = ino[dirInode].start;
    std::vector<uint8_t> d(bs);
    for (uint32_t bb = 0; bb < blocks; ++bb) {
      r->read((uint64_t)(st + bb) * bs, bs, d.data());
      uint64_t o = 0;
      while (o + 17 < bs) {
        uint32_t ch = rd32(d.data() + o);
        uint32_t ty = rd32(d.data() + o + 4);
        uint32_t nl = rd32(d.data() + o + 8);
        uint32_t es = rd32(d.data() + o + 12);
        if (es == 0)
          break;
        if (ty == 3 && nl > 0 && nl < 256 && o + 16 + nl <= bs) {
          std::string nm(reinterpret_cast<const char *>(d.data() + o + 16), nl);
          if (nm == name && ch < ino.size())
            return static_cast<int64_t>(ch);
        }
        o += es;
      }
    }
    return -1;
  }

  void walk(uint32_t i, const std::string &pre, DataSource *r, uint32_t bs) {
    if (i >= ino.size())
      return;
    uint32_t blocks = ino[i].blocks, st = ino[i].start;
    std::vector<uint8_t> d(bs);
    for (uint32_t bb = 0; bb < blocks; ++bb) {
      r->read((uint64_t)(st + bb) * bs, bs, d.data());
      uint64_t o = 0;
      while (o + 17 < bs) {
        uint32_t ch = rd32(d.data() + o);
        uint32_t ty = rd32(d.data() + o + 4);
        uint32_t nl = rd32(d.data() + o + 8);
        uint32_t es = rd32(d.data() + o + 12);
        if (es == 0)
          break;
        if (nl > 0 && nl < 256 && o + 16 + nl <= bs) {
          std::string nm(reinterpret_cast<const char *>(d.data() + o + 16), nl);
          if (ty == 2 && ch < ino.size()) {
            files[pre + "/" + nm] = {ino[ch].size, ino[ch].start};
          } else if (ty == 3 && nm != "." && nm != "..") {
            walk(ch, pre + "/" + nm, r, bs);
          }
        }
        o += es;
      }
    }
  }

  void build() {
    uint8_t ekpfs[32];
    if (!getEkpfs(ekpfs)) {
      LOG_ERROR("pkg: could not recover EKPFS (not a fake pkg?)");
      return;
    }

    // Standard fpkg PFS image offset. Read the header field when it looks sane,
    // else fall back to the well-known constant pkg_extract.py hardcodes.
    uint8_t off[8];
    pkg.Seek(0x410, utl::seekMode::seek_set);
    pkg.Read(off, 8);
    uint64_t pfsOff = (static_cast<uint64_t>(be32(off)) << 32) | be32(off + 4);
    if (pfsOff < 0x1000 || pfsOff >= pkg.GetSize())
      pfsOff = 0x700000;

    // Outer PFS: only its block size and (XTS) reader are needed.
    auto *raw = make<RawSource>(pkg, pfsOff);
    uint32_t outerBs = 0;
    bool outerSigned = false;
    uint64_t outerNd = 0, outerNdb = 0;
    DataSource *outer =
        buildPfs(raw, ekpfs, outerBs, outerSigned, outerNd, outerNdb);
    if (outerBs == 0)
      return;

    // The inner image is a contiguous, PFSC-compressed stream inside the outer
    // PFS. Its start block depends on the size of the outer metadata (superblock
    // + inodes + dirents + flat_path_table), so locate it by its "PFSC" magic
    // rather than assuming a fixed block. (Isaac lands at block 11, P.T. at 19.)
    uint64_t innerBlk = 11;
    for (uint64_t bi = 1; bi < 256; ++bi) {
      uint8_t magic[4];
      outer->read(bi * outerBs, sizeof(magic), magic);
      if (magic[0] == 'P' && magic[1] == 'F' && magic[2] == 'S' && magic[3] == 'C') {
        innerBlk = bi;
        break;
      }
    }
    auto *sub = make<SubSource>(outer, innerBlk * outerBs);
    auto *pfsc = make<PfscSource>(sub);

    bool innerSigned = false;
    uint64_t innerNd = 0, innerNdb = 0;
    inner = buildPfs(pfsc, ekpfs, innerBs, innerSigned, innerNd, innerNdb);
    if (innerBs == 0)
      return;

    readInodes(inner, innerBs, innerSigned, innerNd, innerNdb);
    if (ino.empty())
      return;

    // The PFS superroot holds `flat_path_table` + `uroot`; the game files live
    // under uroot. Present them game-relative (uroot == /app0) by walking from
    // there, falling back to the raw root if the image is laid out flat.
    int64_t root = findChildDir(0, "uroot", inner, innerBs);
    walk(root >= 0 ? static_cast<uint32_t>(root) : 0, "", inner, innerBs);

    valid = true;
    LOG_INFO("pkg: mounted image, {} files", files.size());
  }

  int64_t readNode(const PkgFilesystem::Node &n, void *buf, int64_t off,
                   int64_t len) {
    if (off < 0 || len < 0)
      return -1;
    if (static_cast<uint64_t>(off) >= n.size)
      return 0;
    uint64_t take =
        std::min<uint64_t>(len, n.size - static_cast<uint64_t>(off));
    std::lock_guard<std::mutex> lk(io);
    uint64_t innerOff =
        static_cast<uint64_t>(n.startBlock) * innerBs + static_cast<uint64_t>(off);
    inner->read(innerOff, take, static_cast<uint8_t *>(buf));
    return static_cast<int64_t>(take);
  }

  // Read a well-known outer-PKG entry (param.sfo = 0x1000, icon0.png = 0x1200)
  // straight from the PKG header's entry table. These entries are plaintext in
  // a fake pkg and live outside the encrypted PFS, so this works regardless of
  // whether the inner image decrypted. Same big-endian header layout getEkpfs
  // walks: entry count @0x10, table offset @0x18, 0x20-byte rows of
  // [id @0, data offset @16, data size @20].
  bool readEntry(uint32_t wantId, std::vector<uint8_t> &out) {
    if (!pkg.IsOpen())
      return false;
    std::lock_guard<std::mutex> lk(io);
    auto R = [&](uint64_t o, uint64_t n, uint8_t *dst) {
      pkg.Seek(o, utl::seekMode::seek_set);
      pkg.Read(dst, n);
    };
    uint8_t tmp[4];
    R(0x10, 4, tmp);
    uint32_t ec = be32(tmp);
    R(0x18, 4, tmp);
    uint32_t to = be32(tmp);
    if (ec == 0 || ec > 0x10000)
      return false;
    std::vector<uint8_t> tb(static_cast<size_t>(ec) * 0x20);
    R(to, tb.size(), tb.data());
    for (uint32_t i = 0; i < ec; ++i) {
      const uint8_t *e = tb.data() + i * 0x20;
      if (be32(e) != wantId)
        continue;
      uint32_t off = be32(e + 16), sz = be32(e + 20);
      if (sz == 0 || static_cast<uint64_t>(off) + sz > pkg.GetSize())
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
  std::string key = relPath[0] == '/' ? relPath : std::string("/") + relPath;
  auto it = impl_->files.find(key);
  return it == impl_->files.end() ? nullptr : &it->second;
}

int64_t PkgFilesystem::read(const Node &node, void *buf, int64_t off,
                            int64_t len) {
  return impl_->readNode(node, buf, off, len);
}

void PkgFilesystem::paths(std::vector<std::string> &out) const {
  if (!impl_)
    return;
  out.reserve(impl_->files.size());
  for (const auto &kv : impl_->files)
    out.push_back(kv.first);
}

int64_t PkgFilesystem::readPkgEntry(uint32_t entryId, std::vector<uint8_t> &out) {
  if (!impl_)
    return -1;
  return impl_->readEntry(entryId, out) ? static_cast<int64_t>(out.size()) : -1;
}
} // namespace vfs
