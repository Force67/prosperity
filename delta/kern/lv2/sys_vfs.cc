
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <unistd.h>
#include <cstdio>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/string_ref.h"
#include "guest_abi.h"

#include "kern/crash.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_vfs.h"
#include "kern/probe/probe_arm.h"
#include "kern/process.h"
#include "kern/ps4/dev/ajm_dev.h"
#include "kern/ps4/dev/authmgr_dev.h"
#include "kern/ps4/dev/av_control_dev.h"
#include "kern/ps4/dev/console_dev.h"
#include "kern/ps4/dev/dce_dev.h"
#include "kern/ps4/dev/deci_stdin_dev.h"
#include "kern/ps4/dev/dipsw_dev.h"
#include "kern/ps4/dev/dir_dev.h"
#include "kern/ps4/dev/dma_dev.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/gc_dev.h"
#include "kern/ps4/dev/hdmi_dev.h"
#include "kern/ps4/dev/hid_dev.h"
#include "kern/ps4/dev/mdctl_dev.h"
#include "kern/ps4/dev/npdrm_dev.h"
#include "kern/ps4/dev/null_dev.h"
#include "kern/ps4/dev/pfsctl_dev.h"
#include "kern/ps4/dev/random_dev.h"
#include "kern/ps4/dev/scegp_dev.h"
#include "kern/ps4/dev/srtc_dev.h"
#include "kern/ps4/dev/tty6_dev.h"
#include "kern/ps4/dev/usbctl_dev.h"
#include "kern/ps4/dev/vtrm_dev.h"
#include "kern/ps4/dev/zero_dev.h"
#include "kern/ps5/dev/dma_dev.h"  // PS5 /dev/dmem (shared-memfd mapping)
#include "kern/ps5/dev/gc_dev.h"   // PS5 AGC /dev/gc device
#include "kern/vfs.h"

#include <sys/select.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/deque.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/lv2/error_table.h"
#include "kern/object_ref.h"
#include "kern/ps4/dev/device.h"
#include "kern/ps4/dev/socket_dev.h"  // FdToSocket, for select over real sockets
#include "logger/logger.h"
#include "options/options.h"
#include "host_memory/host_memory.h"

namespace {
DELTA_OPTION(bool, kQuietGuest, "DELTA_QUIET_GUEST", false);
DELTA_OPTION(bool, kManifestSeq, "DELTA_MANIFEST_SEQ", false);
DELTA_OPTION(bool, kFdStats, "DELTA_FD_STATS", false);
DELTA_OPTION(bool, kFstatTrace, "DELTA_FSTAT_TRACE", false);
DELTA_OPTION(bool, kOpenCaller, "DELTA_OPEN_CALLER", false);
DELTA_OPTION(bool, kRdall, "DELTA_RDALL", false);
DELTA_OPTION(bool, kReadTrace, "DELTA_READ_TRACE", false);
DELTA_OPTION(unsigned, kIoMbps, "DELTA_IO_MBPS", 0);
DELTA_OPTION(bool, kVfsTrace, "DELTA_VFS_TRACE", false);
DELTA_OPTION(bool, kQarBuf, "DELTA_QARBUF", false);
DELTA_OPTION(bool, kIoprogress, "DELTA_IOPROGRESS", false);
}  // namespace

namespace kern {
// Scan the (guest) stack for the first return address inside any guest module's
// .text and print it as <module>+offset, to pin which guest code issued an
// open. Native backend runs handlers on the guest stack. Gated; for tracing
// loops.
static void PrintOpenCaller(const char* path) {
  if (!kOpenCaller)
    return;
  auto* proc = Process::GetActive();
  if (!proc)
    return;
  auto* sp = reinterpret_cast<uintptr_t*>(__builtin_frame_address(0));
  int printed = 0;
  for (int i = 0; i < 768 && printed < 5; i++) {
    uintptr_t v = sp[i];
    for (auto& m : proc->GetModuleList()) {
      auto& mi = m->GetInfo();
      auto base = reinterpret_cast<uintptr_t>(mi.text_seg.addr);
      if (base && v >= base && v < base + mi.text_seg.size) {
        BASE_LOGI("open-caller", "{} : {}+{:#x}", path, mi.name.c_str(),
                  v - base);
        printed++;
        break;
      }
    }
  }
}

static Device* MakeDevice(const char* device_name) {
  base::StringRef xname(device_name);

  Device* dev = nullptr;
  auto* proc = Process::GetActive();
  if (xname == "console")
    dev = new ConsoleDevice(proc->GetObjTable());
  if (xname == "deci_tty6")
    dev = new Tty6Device(proc->GetObjTable());
  if (xname == "deci_stdin")
    dev = new DeciStdinDevice(proc->GetObjTable());
  if (xname == "null")
    dev = new NullDevice(proc->GetObjTable());
  if (xname == "zero")
    dev = new ZeroDevice(proc->GetObjTable());
  if (xname == "mdctl")
    dev = new MdctlDevice(proc->GetObjTable());
  if (xname == "av_control")
    dev = new AvControlDevice(proc->GetObjTable());
  if (xname == "hdmi")
    dev = new HdmiDevice(proc->GetObjTable());
  if (xname == "srtc")
    dev = new SrtcDevice(proc->GetObjTable());
  if (xname == "authmgr")
    dev = new AuthmgrDevice(proc->GetObjTable());
  if (xname == "npdrm")
    dev = new NpdrmDevice(proc->GetObjTable());
  if (xname == "vtrm")
    dev = new VtrmDevice(proc->GetObjTable());
  if (xname == "pfsctldev")
    dev = new PfsctlDevice(proc->GetObjTable());
  if (xname == "usbctl")
    dev = new UsbctlDevice(proc->GetObjTable());
  if (xname == "hid")
    dev = new HidDevice(proc->GetObjTable());
  if (xname == "sceGp")
    dev = new SceGpDevice(proc->GetObjTable());
  if (xname == "gc")
    dev = (proc && proc->GetPlatform() == kern::Process::Platform::kPs5)
              ? static_cast<Device*>(new GcDevicePs5(proc->GetObjTable()))
              : static_cast<Device*>(new GcDevice(proc->GetObjTable()));
  if (xname == "dce")
    dev = new DceDevice(proc->GetObjTable());
  if (xname == "dipsw")
    dev = new DipswDevice(proc->GetObjTable());
  if (xname == "random" || xname == "urandom")
    dev = new RandomDevice(proc->GetObjTable());
  // PS5 only. /dev/rng lets libSceSsl's DT_INIT seed itself; on Orbis that
  // reaches libSceNpMatching2's init, which derefs an NpManager context we
  // leave null (Tomb Raider faults). Widen once PS4 Np bring-up follows.
  if (xname == "rng" && proc &&
      proc->GetPlatform() == kern::Process::Platform::kPs5)
    dev = new RandomDevice(proc->GetObjTable());
  if (xname == "ajm")
    dev = new AjmDevice(proc->GetObjTable());
  /*there are multiple of these*/
  if (xname.find("dmem", 0, 4) != base::StringRef::npos)
    dev = (proc && proc->GetPlatform() == kern::Process::Platform::kPs5)
              ? static_cast<Device*>(new DmaDevicePs5(proc->GetObjTable()))
              : static_cast<Device*>(new DmaDevice(proc->GetObjTable()));

  return dev;
}

int PS4ABI sys_open(const char* path, u32 flags, u32 mode) {
  if (!path)
    return -SysError::eINVAL;

  // Kernel open flag validation:
  //   * accmode (flags & 3) > O_RDWR (2) without O_EXEC (0x40000) is EINVAL.
  //   * O_EXEC with a non-zero accmode (not O_RDONLY) is EINVAL.
  const u32 accmode = flags & O_ACCMODE;
  if (accmode > O_RDWR && !(flags & O_EXEC))
    return -SysError::eINVAL;
  if ((flags & O_EXEC) && accmode != 0)
    return -SysError::eINVAL;

  if (kVfsTrace)
    BASE_LOGI("open", "{} flags={:#x} mode={:#x}", path, flags, mode);
  if (std::strstr(path, ".psarc"))
    PrintOpenCaller(path);

  if (std::strncmp(path, "/dev/", 5) == 0) {
    const char* name = &path[5];

    auto dev = MakeDevice(name);
    if (dev) {
      // Object::name is what every device diagnostic prints, and nothing had
      // ever set it, so an unknown ioctl reported the device it arrived on as
      // an empty string.
      dev->SetName(name);

      if (!dev->Init(name, flags, mode)) {
        dev->ReleaseHandle();
        return -SysError::eNXIO;
      }

      return dev->handle();
    }
    // unknown device: fail soft instead of trapping
    return -SysError::eNOENT;
  }

  // Directory: games open (O_DIRECTORY) then getdents. The guest's flag bits
  // are FreeBSD's, so confirm with a stat when the flag is absent: a read open
  // of a directory must still yield a DirDevice, else getdents reports ENOTDIR
  // and a d_reclen walk never advances (Dead Cells spins on its loading
  // screen). Write opens are never directories, so they skip the stat.
  bool as_dir = (flags & O_DIRECTORY) != 0;
  if (!as_dir && (flags & O_ACCMODE) == O_RDONLY && !(flags & O_CREAT)) {
    i64 dsize = 0;
    bool is_dir = false;
    as_dir = vfs::Stat(path, dsize, is_dir) && is_dir;
  }
  if (as_dir) {
    base::Vector<vfs::DirEntry> entries;
    if (vfs::ListDir(path, entries)) {
      const size_t n = entries.size();
      auto* dir = new DirDevice(Process::GetActive()->GetObjTable(),
                                base::move(entries));
      if (kVfsTrace)
        BASE_LOGI("open", "  -> dir fd={} entries={} {}", dir->handle(), n,
                  path);
      return dir->handle();
    }
    if (kVfsTrace)
      BASE_LOGI("open", "  -> dir ENOENT {}", path);
    return -SysError::eNOENT;
  }

  // Writable open (savedata): a create/write flag on a path under a writable
  // host mount goes to a writable FileDevice. Read-only titles never take this
  // (they open /app0, a read-only virtual mount), so it can't affect them.
  const bool write_intent =
      accmode == O_WRONLY || accmode == O_RDWR || (flags & O_CREAT);
  if (write_intent) {
    base::String host = vfs::ResolveWritable(path);
    if (!host.empty()) {
      auto* file = new FileDevice(Process::GetActive()->GetObjTable());
      if (file->OpenWritable(host, (flags & O_CREAT) != 0,
                             (flags & O_TRUNC) != 0)) {
        if (kVfsTrace)
          BASE_LOGI("open", "  -> writable fd={} {}", file->handle(),
                    host.c_str());
        return file->handle();
      }
      file->ReleaseHandle();
      return -SysError::eNOENT;
    }
  }

  // Regular file: resolve through the VFS (host + virtual mounts).
  io::File vf = vfs::OpenRead(path);
  if (!vf.Exists()) {
    if (kVfsTrace)
      BASE_LOGI("open", "  -> ENOENT {}", path);
    return -SysError::eNOENT;
  }

  i64 fsize = vf.GetSize();
  auto* file = new FileDevice(Process::GetActive()->GetObjTable());
  if (!file->Adopt(base::move(vf))) {
    file->ReleaseHandle();
    return -SysError::eNOENT;
  }
  // SOTTR's TAFS loader reads .manifest.bin with an uninitialised file offset;
  // serve those sequentially so the header (off 0) loads. See SetSeqMode().
  if (kManifestSeq && std::strstr(path, ".manifest.bin"))
    file->SetSeqMode();
  // Flag manifest fds so the read-request setter hook (DELTA_RDOFF_FIX) can
  // force their read offset to 0.
  if (std::strstr(path, ".manifest.bin"))
    probe::MarkManifestFd(file->handle(), true);
  // Flag .qar archive fds for the DELTA_QARBUF read-destination trace.
  if (std::strstr(path, ".qar"))
    MarkQarFd(file->handle(), true);
  if (kVfsTrace)
    BASE_LOGI("open", "  -> fd={} size={} {}", file->handle(), (long long)fsize,
              path);
  return file->handle();
}

// Resolve an fd (object-table handle) back to the device that backs it.
static Device* FdToDevice(u32 fd) {
  auto* obj = Process::GetActive()->GetObjTable().Get(fd);
  if (!obj || obj->type() != Object::Type::kDevice)
    return nullptr;
  return static_cast<Device*>(obj);
}

// DELTA_FD_STATS: bytes read per fd, dumped periodically; "opened but never
// read" is a strong signal that whatever consumes the asset is stuck.
void FdReadStat(u32 fd, i64 n) {
  if (!kFdStats || n <= 0)
    return;
  static base::Atomic<u64> bytes[4096];
  static base::Atomic<u64> calls[4096];
  if (fd >= 4096)
    return;
  bytes[fd].fetch_add(static_cast<u64>(n), base::memory_order_relaxed);
  calls[fd].fetch_add(1, base::memory_order_relaxed);
  static const bool kStarted = [] {
    base::SpawnDetachedThread("sys_vfs", [] {
      for (;;) {
        base::SleepForMilliseconds((20) * 1000);
        BASE_LOGI("fdstats", "--- bytes read per fd ---");
        for (u32 i = 0; i < 4096; i++)
          if (u64 b = bytes[i].load(base::memory_order_relaxed))
            BASE_LOGI("fdstats", "fd={} calls={} bytes={}", i,
                      (unsigned long long)calls[i].load(),
                      (unsigned long long)b);
      }
    });
    return true;
  }();
  (void)kStarted;
}

// DELTA_IO_MBPS=<MiB/s>: cap file-read throughput. A host SSD outruns a loader
// the title tuned to a console drive: a pipeline keeping loaded-but-unfinalized
// data in a fixed CPU budget can be outrun and exhaust it (SotC fills its 1 GiB
// onion heap and dies in its own allocator; the same run survives on a busy
// host). 0 = off.
void ThrottleIo(i64 bytes) {
  const unsigned mbps = kIoMbps;
  if (!mbps || bytes <= 0)
    return;
  static base::Mutex m;
  static base::TimeTicks next{};
  const auto cost = base::Microseconds(
      (i64)((double)bytes * 1e6 / ((double)mbps * 1024.0 * 1024.0)));
  base::TimeTicks until;
  {
    base::LockGuard<base::Mutex> lk(m);
    const auto now = base::TimeTicks::Now();
    if (next < now)
      next = now;
    next = next + cost;
    until = next;
  }
  const base::TimeDelta wait = until - base::TimeTicks::Now();
  if (wait > base::TimeDelta())
    base::SleepForMicroseconds(u64(wait.InMicroseconds()));
}

i64 PS4ABI sys_read(u32 fd, void* buf, size_t nbytes) {
  auto* d = FdToDevice(fd);
  if (!d) {
    // The standard descriptors exist but read nothing: report EOF, not EBADF.
    // Skyrim's INI parser falls back to stderr when the file is missing and its
    // fgets loop only stops on EOF; an error left it reading fd 2 forever at
    // 100% CPU.
    if (fd <= 2)
      return 0;
    if (kRdall)
      BASE_LOGI("rd", "fd={} -> EBADF (no device)", fd);
    return -SysError::eBADF;
  }
  // The host kernel may write the buffer (a file read): open any page a
  // cache protected, or the call fails with EFAULT.
  host_memory::BeforeHostWrite(buf, nbytes);
  i64 r = d->Read(buf, nbytes);
  ThrottleIo(r);
  FdReadStat(fd, r);
  // DELTA_READ_TRACE: log large reads (asset/texture loads) + their target
  // buffer, to see whether texture data lands in the GPU texture region (0x41x)
  // directly or a staging buffer the game later copies from.
  if (kReadTrace && nbytes >= 0x4000)
    BASE_LOGI("read", "fd={} buf={:p} nbytes={:#x} -> {}", fd, buf, nbytes,
              (long long)r);
  if (kRdall) {
    u32 f4 = 0;
    if (buf && r >= 4)
      f4 = *reinterpret_cast<const u32*>(buf);
    BASE_LOGI("rd", "t={} fd={} nbytes={:#x} -> {} buf={:p} first4={:08x}",
              (long)gettid(), fd, nbytes, (long long)r, buf, f4);
  }
  return r;
}

i64 PS4ABI sys_lseek(u32 fd, i64 offset, int whence) {
  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->Lseek(offset, whence);
}

// FreeBSD struct statfs (0x1D8 bytes). Only capacity matters (it decides
// whether a title may write); unhandled, the caller read garbage as "no space"
// and Minecraft refused to open a world. Needs privilege 0x2AC in the kernel.
struct BsdStatfs {
  u32 f_version, f_type;
  u64 f_flags, f_bsize, f_iosize;
  u64 f_blocks, f_bfree;
  i64 f_bavail;
  u64 f_files;
  i64 f_ffree;
  u64 f_syncwrites, f_asyncwrites, f_syncreads, f_asyncreads;
  u64 f_spare[10];
  u32 f_namemax, f_owner;
  i32 f_fsid[2];
  char f_charspare[80];
  char f_fstypename[16];
  char f_mntfromname[88];
  char f_mntonname[88];
};

static void FillStatfs(void* buf, const char* mount) {
  auto* sf = static_cast<BsdStatfs*>(buf);
  std::memset(sf, 0, sizeof(*sf));
  constexpr u64 kBlockSize = 0x8000;     // 32 KiB, as the PS5 fs
  constexpr u64 kBlocks = 0x1000000ull;  // 512 GiB total
  sf->f_version = 0x20140518;            // STATFS_VERSION
  sf->f_bsize = kBlockSize;
  sf->f_iosize = kBlockSize;
  sf->f_blocks = kBlocks;
  sf->f_bfree = kBlocks / 2;
  sf->f_bavail = static_cast<i64>(kBlocks / 2);  // 256 GiB free
  sf->f_files = 0x100000;
  sf->f_ffree = 0x100000 / 2;
  sf->f_namemax = 255;
  std::strncpy(sf->f_fstypename, "exfatfs", sizeof(sf->f_fstypename) - 1);
  std::strncpy(sf->f_mntfromname, "/dev/da0", sizeof(sf->f_mntfromname) - 1);
  std::strncpy(sf->f_mntonname, mount && *mount ? mount : "/",
               sizeof(sf->f_mntonname) - 1);
}

int PS4ABI sys_statfs(const char* path, void* buf) {
  if (kVfsTrace)
    BASE_LOGI("statfs", "'{}'", path ? path : "(null)");
  if (!buf)
    return -SysError::eFAULT;
  FillStatfs(buf, path);
  return 0;
}

int PS4ABI sys_fstatfs(u32 fd, void* buf) {
  if (!buf)
    return -SysError::eFAULT;
  FillStatfs(buf, "/");
  return 0;
}

int PS4ABI sys_fstat(u32 fd, void* stat) {
  // Zero first: a failed/unsupported fstat must not leave the caller's stat
  // buffer uninitialized. Games read st_size from it without checking the
  // return and then allocate that many bytes (garbage -> bad_alloc).
  if (stat)
    std::memset(stat, 0, sizeof(SceKernelStat));
  // shm fds aren't device-backed; size them from the shm backing so a title
  // that fstat()s a shm before mmap'ing it (e.g. libSceAvSetting) gets a real
  // st_size instead of -EBADF + a zero-length map.
  if (size_t sz = ShmFstatSize(fd); sz != SIZE_MAX) {
    if (stat) {
      auto* st = static_cast<SceKernelStat*>(stat);
      st->st_size = static_cast<i64>(sz);
      st->st_mode = 0x8000;  // S_IFREG
      st->st_blksize = 0x4000;
    }
    return 0;
  }
  auto* d = FdToDevice(fd);
  if (!d) {
    // The standard descriptors are not device-backed here; report them as
    // character devices, not EBADF (Skyrim's INI parser stats its stderr
    // fallback; an error makes its stdio layer treat the stream as broken).
    if (fd <= 2) {
      if (stat) {
        auto* st = static_cast<SceKernelStat*>(stat);
        st->st_mode = 0x2000;  // S_IFCHR
        st->st_blksize = 0x4000;
      }
      return 0;
    }
    if (kFstatTrace) {
      static base::Mutex m;
      static base::HashMap<u32, u64> bad;
      base::LockGuard<base::Mutex> lk(m);
      if (bad[fd]++ == 0)
        BASE_LOGI("fstat", "fd={} -> EBADF (unknown descriptor)", fd);
    }
    return -SysError::eBADF;
  }
  int r = d->Fstat(stat);
  if (kRdall && stat)
    BASE_LOGI("fstat", "fd={} -> st_size={}", fd,
              (long long)static_cast<SceKernelStat*>(stat)->st_size);
  return r;
}

int PS4ABI sys_stat(const char* path, void* stat) {
  if (!path || !stat)
    return -SysError::eFAULT;
  // Zero first, for the reason sys_fstat documents: callers read st_size
  // without checking the return and then size a buffer from it. A missing file
  // must leave st_size = 0, not stack garbage (DOOM read a -1 size and
  // crashed).
  std::memset(stat, 0, sizeof(SceKernelStat));
  i64 size = 0;
  bool is_dir = false;
  if (!vfs::Stat(path, size, is_dir)) {
    if (kRdall)
      BASE_LOGI("stat", "{} -> ENOENT", path);
    return -SysError::eNOENT;
  }
  FillStat(*reinterpret_cast<SceKernelStat*>(stat),
           is_dir ? kSceFileModeDir : kSceFileModeReg, size);
  if (kRdall)
    BASE_LOGI("stat", "{} -> size={} dir={}", path, (long long)size,
              (int)is_dir);
  return 0;
}

i64 PS4ABI sys_getdents(u32 fd, void* buf, size_t nbytes) {
  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->Getdents(buf, nbytes);
}

// Regular-file fd slots are released a bounded number of closes late. SOTTR
// opens a file, hands the fd to an async I/O worker, then closes and reopens:
// freeing the slot at once reuses it for the next open, and the pending read
// lands on the wrong file -> garbage archive header -> a ~32 GiB entry-table
// allocation. Keeping the last N closed slots alive lets the lagging read
// finish right. PFS-backed files share one host fd, so this costs no host
// descriptors; char devices close at once.
static base::Mutex g_defer_m;
static base::SimpleDeque<u32> g_deferred;
static constexpr size_t kDeferredCloseWindow = 256;

int PS4ABI sys_close(u32 fd) {
  auto* proc = Process::GetActive();

  if (proc && fd != -1) {
    if (kRdall)
      BASE_LOGI("close", "fd={}", fd);
    auto* d = FdToDevice(fd);
    if (d && d->IsRegularFile()) {
      u32 evict = static_cast<u32>(-1);
      {
        base::LockGuard<base::Mutex> lk(g_defer_m);
        // A deferred fd keeps its slot pinned, so it can't have been reopened
        // as a different file; a second close of it is a redundant double-close
        // and must not queue a second (wrong) release.
        bool already = base::Find(g_deferred.begin(), g_deferred.end(), fd) !=
                       g_deferred.end();
        if (!already) {
          g_deferred.push_back(fd);
          if (g_deferred.size() > kDeferredCloseWindow) {
            evict = g_deferred.front();
            g_deferred.pop_front();
          }
        }
      }
      if (evict != static_cast<u32>(-1))
        proc->GetObjTable().Release(evict);
      return 0;
    }
    proc->GetObjTable().Release(fd);
    return 0;
  }

  LOG_WARNING("failed to release handle {}", fd);
  return -SysError::eBADF;
}
}  // namespace kern

namespace kern {

enum { kSeekSet = 0, kSeekCur = 1 };

struct sce_iovec {  // NOLINT(readability-identifier-naming): guest ABI name
  void* iov_base;
  size_t iov_len;
};

int PS4ABI sys_access(const char* path, int mode) {
  if (!path)
    return -SysError::eINVAL;
  // We model read-only assets, so existence is the only check we can honour;
  // W_OK/X_OK are accepted for anything that exists.
  i64 size = 0;
  bool is_dir = false;
  if (!vfs::Stat(path, size, is_dir))
    return -SysError::eNOENT;
  return 0;
}

int PS4ABI sys_faccessat(int fd, const char* path, int mode, int flag) {
  return sys_access(path, mode);
}

// We have no symbolic links in the VFS. The kernel returns EINVAL when the
// target is not a VLNK vnode, so we do too. The PS4 uses symlinks only for
// /app0 -> the PFS mount root and the base system dirs; our VFS resolves
// those directly, so readlink should never reach a callable path.
int PS4ABI sys_readlink(const char* path, char* buf, size_t bufsize) {
  return -SysError::eINVAL;
}

int PS4ABI sys_readlinkat(int fd, const char* path, char* buf, size_t bufsize) {
  return -SysError::eINVAL;
}

// No symlinks, so lstat is plain stat. Zero the buffer first for the reason
// sys_fstat documents: callers read st_size without checking the return.
int PS4ABI sys_lstat(const char* path, void* stat) {
  if (!path)
    return -SysError::eINVAL;
  if (stat)
    std::memset(stat, 0, sizeof(SceKernelStat));
  i64 size = 0;
  bool is_dir = false;
  if (!vfs::Stat(path, size, is_dir))
    return -SysError::eNOENT;
  FillStat(*reinterpret_cast<SceKernelStat*>(stat),
           is_dir ? kSceFileModeDir : kSceFileModeReg, size);
  return 0;
}

int PS4ABI sys_fstatat(int fd, const char* path, void* stat, int flag) {
  return sys_lstat(path, stat);
}

// sys_fcntl: non-privileged cmds validated against 0x3818 {F_GETFL, F_SETFL,
// F_GETLK, F_SETLK, F_SETLKW}; 7/8/9 (OGETLK/OSETLK/OSETLKW) translate to
// 11/12/13. Flags and advisory locks unmodelled: GETFD/GETFL report 0, locks
// accept silently.
int PS4ABI sys_fcntl(u32 fd, int cmd, i64 arg) {
  enum {
    // FreeBSD fcntl(2) command spelling.
    // NOLINTBEGIN(readability-identifier-naming)
    F_DUPFD = 0,
    F_GETFD = 1,
    F_SETFD = 2,
    F_GETFL = 3,
    F_SETFL = 4,
    F_GETOWN = 5,
    F_SETOWN = 6,
    F_OGETLK = 7,
    F_OSETLK = 8,
    F_OSETLKW = 9,
    F_GETLK = 11,
    F_SETLK = 12,
    F_SETLKW = 13,
    // NOLINTEND(readability-identifier-naming)
  };
  // Normalize legacy OGETLK/OSETLK/OSETLKW (7/8/9) to their modern equivalents.
  int ncmd = cmd;
  switch (cmd) {
    case F_OGETLK:
      ncmd = F_GETLK;
      break;
    case F_OSETLK:
      ncmd = F_SETLK;
      break;
    case F_OSETLKW:
      ncmd = F_SETLKW;
      break;
  }
  switch (ncmd) {
    case F_GETFL:
      // Report O_RDONLY: the device opened with whatever flags the guest
      // passed, but we model read-only access for regular files.
      return 0;
    case F_SETFL:
    case F_GETFD:
    case F_SETFD:
      return 0;
    case F_GETLK:
      // No locks held: return F_UNLCK (type 2) in the caller's flock struct.
      if (arg) {
        auto* fl = reinterpret_cast<i32*>(arg);
        fl[0] = 2;  // l_type = F_UNLCK
      }
      return 0;
    case F_SETLK:
    case F_SETLKW:
      return 0;  // advisory lock accepted, not enforced
    case F_DUPFD:
      return -SysError::eOPNOTSUPP;  // no descriptor duplication in the object
                                     // table
    case F_GETOWN:
    case F_SETOWN:
      return 0;  // no signal delivery so ownership is inert
    default:
      LOG_WARNING("sys_fcntl: unhandled cmd {} on fd {} -> 0", cmd, fd);
      return 0;
  }
}

int PS4ABI sys_dup(u32 fd) {
  LOG_WARNING("sys_dup({}) unsupported", fd);
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_dup2(u32 oldfd, u32 newfd) {
  LOG_WARNING("sys_dup2({}, {}) unsupported", oldfd, newfd);
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_fsync(u32 fd) {
  return 0;
}
int PS4ABI sys_fdatasync(u32 fd) {
  return 0;
}

int PS4ABI sys_getcwd(char* buf, size_t size) {
  if (!buf || size == 0)
    return -SysError::eINVAL;
  const char* cwd = "/app0";  // the single working directory we expose
  size_t n = std::strlen(cwd);
  if (n + 1 > size)
    n = size - 1;
  std::memcpy(buf, cwd, n);
  buf[n] = '\0';
  return 0;
}

// pread/pwrite must not disturb the file pointer. Our devices only offer
// seek+read, so snapshot the current offset, do the positioned I/O, then
// restore it. Without the restore a following read() would resume from the
// wrong place.
static bool g_qar_fd[8192] = {false};
void MarkQarFd(u32 fd, bool v) {
  if (fd < 8192)
    g_qar_fd[fd] = v;
}

void ThrottleIo(i64 bytes);

i64 PS4ABI sys_pread(u32 fd, void* buf, size_t nbytes, i64 offset) {
  auto* d = FdToDevice(fd);
  if (!d) {
    if (kRdall)
      BASE_LOGI("pread", "fd={} off={} -> EBADF (no device)", fd,
                (long long)offset);
    return -SysError::eBADF;
  }
  struct timespec t0;
  if (kQarBuf)
    clock_gettime(CLOCK_MONOTONIC, &t0);
  i64 saved = d->Lseek(0, kSeekCur);
  d->Lseek(offset, kSeekSet);
  // The host kernel may write the buffer (a file read): open any page a
  // cache protected, or the call fails with EFAULT.
  host_memory::BeforeHostWrite(buf, nbytes);
  i64 r = d->Read(buf, nbytes);
  if (saved >= 0)
    d->Lseek(saved, kSeekSet);
  long read_us = 0;
  if (kQarBuf) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    read_us =
        (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
  }
  ThrottleIo(r);
  if (kRdall) {
    u32 f4 = 0;
    if (buf && r >= 4)
      f4 = *reinterpret_cast<const u32*>(buf);
    BASE_LOGI("pread",
              "t={} fd={} off={} nbytes={:#x} -> {} buf={:p} first4={:08x}",
              (long)gettid(), fd, (long long)offset, (size_t)nbytes,
              (long long)r, buf, f4);
  }
  // DELTA_QARBUF: where does streamed .qar data land? Reports the destination
  // buffer for reads on a *.qar fd, so we can tell whether textures stream into
  // a GPU-mapped region (0x81xx, directly bindable) or a low staging buffer
  // that still needs a copy/commit step the engine never performs.
  if (fd < 8192 && g_qar_fd[fd] && kQarBuf) {
    BASE_LOGI("qarbuf", "fd={} off={} nbytes={:#x} -> {} buf={:p} {}us", fd,
              (long long)offset, (size_t)nbytes, (long long)r, buf, read_us);
  }
  // DELTA_IOPROGRESS: throttled per-fd streaming high-water mark: is FIOS2's
  // pread of a large world archive advancing or stalled, without the
  // DELTA_RDALL firehose. One line per fd per ~2s: current + max offset and
  // MB/s since the last line.
  if (kIoprogress) {
    // maxOff/lastMax = streaming high-water. nNew climbing = fetching NEW
    // bytes; nReread = a downstream consume/decompress stage that never drains,
    // so the streamer re-issues the same reads. lastOff catches exact-repeat
    // reads.
    struct FdIo {
      i64 max_off, last_max, last_off;
      long last_ms;
      long n_new, n_reread, n_same;
    };
    static base::Mutex m;
    static base::HashMap<u32, FdIo> tbl;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long now_ms = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    base::LockGuard<base::Mutex> lk(m);
    auto& e = tbl[fd];
    i64 end = offset + (r > 0 ? r : 0);
    if (end > e.max_off)
      e.n_new++;
    else
      e.n_reread++;
    if (offset == e.last_off)
      e.n_same++;
    e.last_off = offset;
    if (end > e.max_off)
      e.max_off = end;
    if (e.last_ms == 0)
      e.last_ms = now_ms;
    if (now_ms - e.last_ms >= 2000) {
      double mb = (e.max_off - e.last_max) / 1048576.0;
      double sec = (now_ms - e.last_ms) / 1000.0;
      BASE_LOGI("ioprog",
                "fd={} off={} max={} ({:.1f} MB) +{:.2f} MB/s  new={} "
                "reread={} same={}",
                fd, (long long)offset, (long long)e.max_off,
                e.max_off / 1048576.0, sec > 0 ? mb / sec : 0.0, e.n_new,
                e.n_reread, e.n_same);
      e.last_max = e.max_off;
      e.last_ms = now_ms;
      e.n_new = e.n_reread = e.n_same = 0;
    }
  }
  return r;
}

i64 PS4ABI sys_pwrite(u32 fd, const void* buf, size_t nbytes, i64 offset) {
  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  i64 saved = d->Lseek(0, kSeekCur);
  d->Lseek(offset, kSeekSet);
  i64 r = d->Write(buf, nbytes);
  if (saved >= 0)
    d->Lseek(saved, kSeekSet);
  return r;
}

i64 PS4ABI sys_writev(u32 fd, const void* iov, int iovcnt) {
  auto* segs = static_cast<const sce_iovec*>(iov);
  if (!segs || iovcnt < 0)
    return -SysError::eINVAL;

  if (fd == 1 || fd == 2) {  // stdout / stderr, like sys_write
    // Through the logger rather than printf: a title's own diagnostics are the
    // most direct account of what it is doing, and interleaving them with ours
    // by timestamp is what makes them usable.
    i64 total = 0;
    base::String out;
    for (int i = 0; i < iovcnt; ++i) {
      out.append(static_cast<const char*>(segs[i].iov_base), segs[i].iov_len);
      total += static_cast<i64>(segs[i].iov_len);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
      out.pop_back();
    if (!out.empty())
      BASE_LOGI("guest", "{}", out.c_str());
    if (const char* m = std::getenv("DELTA_GUEST_LOG_STACK");
        m && out.find(m) != base::String::npos)
      GuestStackTrace("guestlog", 12);
    return total;
  }

  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  i64 total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    i64 r = d->Write(segs[i].iov_base, segs[i].iov_len);
    if (r < 0)
      return r;
    total += r;
  }
  return total;
}

i64 PS4ABI sys_readv(u32 fd, const void* iov, int iovcnt) {
  auto* segs = static_cast<const sce_iovec*>(iov);
  if (!segs || iovcnt < 0)
    return -SysError::eINVAL;

  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  i64 total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    host_memory::BeforeHostWrite(segs[i].iov_base, segs[i].iov_len);
    i64 r = d->Read(segs[i].iov_base, segs[i].iov_len);
    if (r < 0)
      return r;
    total += r;
  }
  return total;
}

// Positional vectored I/O. These were stubbed to return 0, which reads as a
// clean end-of-file to the caller: a title that loads through preadv gets empty
// buffers and no error to notice it by. Offsets advance across the segments and
// the file position is left alone, like pread/pwrite.
i64 PS4ABI sys_preadv(u32 fd, const void* iov, int iovcnt, i64 offset) {
  auto* segs = static_cast<const sce_iovec*>(iov);
  if (!segs || iovcnt < 0 || offset < 0)
    return -SysError::eINVAL;
  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  const i64 saved = d->Lseek(0, kSeekCur);
  i64 total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    d->Lseek(offset + total, kSeekSet);
    host_memory::BeforeHostWrite(segs[i].iov_base, segs[i].iov_len);
    i64 r = d->Read(segs[i].iov_base, segs[i].iov_len);
    if (r < 0) {
      if (saved >= 0)
        d->Lseek(saved, kSeekSet);
      return total ? total : r;
    }
    total += r;
    if (static_cast<size_t>(r) < segs[i].iov_len)
      break;  // short read: end of file
  }
  if (saved >= 0)
    d->Lseek(saved, kSeekSet);
  return total;
}

i64 PS4ABI sys_pwritev(u32 fd, const void* iov, int iovcnt, i64 offset) {
  auto* segs = static_cast<const sce_iovec*>(iov);
  if (!segs || iovcnt < 0 || offset < 0)
    return -SysError::eINVAL;
  auto* d = FdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  const i64 saved = d->Lseek(0, kSeekCur);
  i64 total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    d->Lseek(offset + total, kSeekSet);
    i64 r = d->Write(segs[i].iov_base, segs[i].iov_len);
    if (r < 0) {
      if (saved >= 0)
        d->Lseek(saved, kSeekSet);
      return total ? total : r;
    }
    total += r;
    if (static_cast<size_t>(r) < segs[i].iov_len)
      break;
  }
  if (saved >= 0)
    d->Lseek(saved, kSeekSet);
  return total;
}

// We have no pollable fds. Returning 0 (zero ready) immediately would turn a
// timed poll into a busy-spin, so honour the caller's timeout by sleeping it
// first (capped). timeout is in milliseconds; negative means "wait forever",
// which we treat as the cap rather than hanging.
int PS4ABI sys_poll(void* fds, u32 nfds, int timeout) {
  int ms = timeout;
  if (ms < 0 || ms > 50)
    ms = 50;
  if (ms > 0)
    ::usleep(static_cast<useconds_t>(ms) * 1000);
  return 0;
}

// FreeBSD and Linux lay an fd_set out the same way: a bitmap of 64-bit words,
// bit n for fd n. nfds bounds it, so only ceil(nfds/64) words are ours to read.
namespace {
constexpr int kFdSetWords = (1024 + 63) / 64;

struct GuestTimeval {
  i64 sec, usec;
};

bool FdIsSet(const void* set, int fd) {
  if (!set || fd < 0 || fd >= kFdSetWords * 64)
    return false;
  return (static_cast<const u64*>(set)[fd / 64] >> (fd % 64)) & 1;
}

void FdSet(void* set, int fd) {
  if (set && fd >= 0 && fd < kFdSetWords * 64)
    static_cast<u64*>(set)[fd / 64] |= 1ull << (fd % 64);
}
}  // namespace

// select(): sockets ask the host; everything else (file, device) is always
// ready and never blocks. The old stub returned "nothing ready" without waiting
// or clearing the sets, so a title selecting with a timeout spun: GTA:SA's
// Gameface thread made 1.8 billion calls in 78s and left the render loop at 0.1
// fps.
int PS4ABI sys_select(int nfds,
                      void* readfds,
                      void* writefds,
                      void* exceptfds,
                      void* timeout) {
  if (nfds < 0)
    return -SysError::eINVAL;
  if (nfds > kFdSetWords * 64)
    nfds = kFdSetWords * 64;

  fd_set host_read, host_write, host_except;
  FD_ZERO(&host_read);
  FD_ZERO(&host_write);
  FD_ZERO(&host_except);
  int host_max = -1, ready = 0;
  // The always-ready fds, collected before the wait: with one of them in the
  // set there is nothing to wait for.
  u64 always_read[kFdSetWords] = {}, always_write[kFdSetWords] = {};

  for (int fd = 0; fd < nfds; fd++) {
    const bool r = FdIsSet(readfds, fd), w = FdIsSet(writefds, fd),
               e = FdIsSet(exceptfds, fd);
    if (!r && !w && !e)
      continue;
    auto* s = FdToSocket(static_cast<u32>(fd));
    if (!s) {
      if (r) {
        FdSet(always_read, fd);
        ready++;
      }
      if (w) {
        FdSet(always_write, fd);
        ready++;
      }
      continue;
    }
    const int h = s->HostFd();
    if (r)
      FD_SET(h, &host_read);
    if (w)
      FD_SET(h, &host_write);
    if (e)
      FD_SET(h, &host_except);
    if (h > host_max)
      host_max = h;
  }

  // A null timeout means "wait forever". We cap it (as sys_poll does) so a
  // title that parks a thread there stays interruptible.
  timeval tv{0, 50 * 1000};
  if (auto* gt = static_cast<const GuestTimeval*>(timeout)) {
    if (gt->sec > 0 || gt->usec >= 50 * 1000)
      tv = {0, 50 * 1000};
    else
      tv = {0, static_cast<suseconds_t>(gt->usec)};
  }
  if (ready)
    tv = {0, 0};

  if (host_max >= 0) {
    const int n =
        ::select(host_max + 1, &host_read, &host_write, &host_except, &tv);
    if (n > 0) {
      for (int fd = 0; fd < nfds; fd++) {
        auto* s = FdToSocket(static_cast<u32>(fd));
        if (!s)
          continue;
        const int h = s->HostFd();
        if (FdIsSet(readfds, fd) && FD_ISSET(h, &host_read)) {
          FdSet(always_read, fd);
          ready++;
        }
        if (FdIsSet(writefds, fd) && FD_ISSET(h, &host_write)) {
          FdSet(always_write, fd);
          ready++;
        }
      }
    }
  } else if (!ready && (tv.tv_sec || tv.tv_usec)) {
    ::usleep(static_cast<useconds_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
  }

  // select reports its answer by rewriting the sets, so the ones it was given
  // have to be cleared even when nothing is ready.
  if (readfds)
    std::memcpy(readfds, always_read, sizeof(always_read));
  if (writefds)
    std::memcpy(writefds, always_write, sizeof(always_write));
  if (exceptfds)
    std::memset(exceptfds, 0, sizeof(always_read));
  return ready;
}

int PS4ABI sys_openat(int fd, const char* path, u32 flags, u32 mode) {
  return sys_open(path, flags, mode);
}

int PS4ABI sys_chdir(const char* path) {
  return 0;
}
int PS4ABI sys_fchdir(u32 fd) {
  return 0;
}

// The host tree stays read-only. We report success so installers and savedata
// setup proceed, but log every call: if a title relies on a file it "created"
// here being readable back, that read returns stale VFS data and this trace is
// the only sign of why.
int PS4ABI sys_unlink(const char* path) {
  // Under a writable mount (savedata, /download0) do the real thing: a title
  // that rewrites a file by unlink+create reads back stale content otherwise.
  if (path && vfs::RemoveFile(path))
    return 0;
  BASE_LOGI("vfs", "unlink('{}') ignored (read-only host)",
            path ? path : "(null)");
  return 0;
}
int PS4ABI sys_rmdir(const char* path) {
  BASE_LOGI("vfs", "rmdir('{}') ignored (read-only host)",
            path ? path : "(null)");
  return 0;
}
int PS4ABI sys_mkdir(const char* path, u32 mode) {
  (void)mode;
  // Real directory creation under a writable mount (savedata); otherwise a
  // no-op success as before (the read-only host content the game expects to
  // exist already does).
  if (path && vfs::MakeDir(path)) {
    if (kVfsTrace)
      BASE_LOGI("vfs", "mkdir('{}') -> host", path);
    return 0;
  }
  return 0;
}
int PS4ABI sys_rename(const char* from, const char* to) {
  base::String hf = from ? vfs::ResolveWritable(from) : base::String();
  base::String ht = to ? vfs::ResolveWritable(to) : base::String();
  if (!hf.empty() && !ht.empty() && std::rename(hf.c_str(), ht.c_str()) == 0)
    return 0;
  BASE_LOGI("vfs", "rename('{}' -> '{}') ignored (read-only host)",
            from ? from : "(null)", to ? to : "(null)");
  return 0;
}

// sys_unlinkat (503): flag bit 0x800 (AT_REMOVEDIR) means rmdir, else unlink.
int PS4ABI sys_unlinkat(int fd, const char* path, int flag) {
  if (flag & 0x800)
    return sys_rmdir(path);
  return sys_unlink(path);
}

// sys_mkdirat (496): identical to mkdir (the dirfd is always AT_FDCWD here).
int PS4ABI sys_mkdirat(int fd, const char* path, u32 mode) {
  return sys_mkdir(path, mode);
}

// sys_renameat (501): identical to rename (both dirfds are AT_FDCWD here).
int PS4ABI sys_renameat(int fd_old,
                        const char* old,
                        int fd_new,
                        const char* to) {
  return sys_rename(old, to);
}

i64 PS4ABI sys_getdirentries(u32 fd, void* buf, size_t nbytes, i64* basep) {
  auto* d = FdToDevice(fd);
  if (!d) {
    if (kVfsTrace)
      BASE_LOGI("getdirentries", "fd={} BADF", fd);
    return -SysError::eBADF;
  }
  // The kernel validates buflen and returns EINVAL on a negative value.
  if (nbytes == 0)
    return -SysError::eINVAL;
  i64 r = d->Getdents(buf, nbytes);
  // On success write the next seek offset to *basep; the byte count consumed is
  // the cookie. Our DirDevice serves all entries on the first call, so it's
  // just the total.
  if (r >= 0 && basep)
    *basep = r;
  if (kVfsTrace)
    BASE_LOGI("getdirentries", "fd={} buf={:p} n={} -> {} basep={}", fd, buf,
              nbytes, (long long)r, basep ? (long long)*basep : -1);
  return r;
}

int PS4ABI sys_closefrom(u32 lowfd) {
  return 0;
}

// dup a descriptor into another process; no multi-proc, so deny.
int PS4ABI sys_rdup() {
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_resume_internal_hdd() {
  return 0;
}

int PS4ABI sys_sync() {
  return 0;
}

int PS4ABI sys_flock() {
  return 0;
}

int PS4ABI sys_utimes() {
  return 0;
}
int PS4ABI sys_futimes() {
  return 0;
}

// pathconf/fpathconf/lpathconf: concrete values, not the -1 sentinel, which is
// indistinguishable from an errno in rax and would misread as failure when
// sizing a buffer. Values = FreeBSD defaults for a UFS-like filesystem.
static i64 PathconfValue(int name) {
  switch (name) {
    case 1:
      return 32767;  // _PC_LINK_MAX
    case 2:
      return 255;  // _PC_MAX_CANON
    case 3:
      return 255;  // _PC_MAX_INPUT
    case 4:
      return 255;  // _PC_NAME_MAX
    case 5:
      return 1024;  // _PC_PATH_MAX
    case 6:
      return 512;  // _PC_PIPE_BUF
    case 7:
      return 1;  // _PC_CHOWN_RESTRICTED
    case 8:
      return 1;  // _PC_NO_TRUNC
    case 9:
      return 255;  // _PC_VDISABLE
    case 11:
      return 64;  // _PC_ACL_PATH_MAX
    case 12:
      return 64;  // _PC_FILESIZEBITS -> at least 64-bit offsets
    default:
      return -SysError::eINVAL;
  }
}
int PS4ABI sys_pathconf(const char* path, int name) {
  (void)path;
  return static_cast<int>(PathconfValue(name));
}
int PS4ABI sys_fpathconf(int fd, int name) {
  (void)fd;
  return static_cast<int>(PathconfValue(name));
}
int PS4ABI sys_lpathconf(const char* path, int name) {
  (void)path;
  return static_cast<int>(PathconfValue(name));
}

int PS4ABI sys_posix_fallocate() {
  return 0;
}
int PS4ABI sys_posix_fadvise() {
  return 0;
}

// sys_randomized_path (602): args {set_path@0, out@8, out_len@16}. Non-null
// set_path stores the new randomized prefix (priv 0x2AF); the current prefix
// (<=256 bytes) always copies to out. It's the per-title sandbox component
// under /system_data; we have no mapping, so report empty (len 0) and the guest
// uses the plain path.
int PS4ABI sys_randomized_path(const char* set_path,
                               char* out,
                               size_t* out_len) {
  (void)set_path;
  if (out && out_len) {
    if (*out_len >= 1)
      out[0] = '\0';
    *out_len = 0;
  }
  return 0;
}

int PS4ABI sys_write(u32 fd, const void* buf, size_t nbytes) {
  if (fd == 1 || fd == 2)  // stdout, stderr
  {
    // DELTA_QUIET_GUEST: the game's debug prints (per-frame message-pump
    // chatter) flood stdout char-by-char and corrupt our diagnostic logs via
    // interleaving. Suppress guest fd1/2 output while diagnosing the host-side
    // render path.
    if (kQuietGuest)
      return static_cast<int>(nbytes);
    // A line at a time through the logger, like sys_writev: host stdout is
    // usually a redirected file, and a plain fwrite sits in stdio a killed run
    // never flushes, making "stopped printing" and "lost the last 4 KiB"
    // identical. Titles write this fd a character at a time, hence the
    // accumulator.
    static base::Mutex mtx;
    static base::String line;
    base::LockGuard<base::Mutex> lk(mtx);
    line.append(static_cast<const char*>(buf), nbytes);
    for (size_t nl; (nl = line.find('\n')) != base::String::npos;) {
      base::String one = line.substr(0, nl);
      line.erase(0, nl + 1);
      while (!one.empty() && one.back() == '\r')
        one.pop_back();
      if (!one.empty())
        BASE_LOGI("guest", "{}", one.c_str());
      if (const char* m = std::getenv("DELTA_GUEST_LOG_STACK");
          m && one.find(m) != base::String::npos)
        GuestStackTrace("guestlog", 12);
    }
    return static_cast<int>(nbytes);
  }

  // A device-backed fd (console/tty): let it handle the write. Otherwise just
  // accept the bytes; libkernel writes debug output to fds we don't model, and
  // trapping there kills the boot.
  if (auto* proc = Process::GetActive()) {
    auto* obj = proc->GetObjTable().Get(fd);
    if (obj && obj->type() == Object::Type::kDevice) {
      i64 r = static_cast<Device*>(obj)->Write(buf, nbytes);
      if (r >= 0)
        return static_cast<int>(r);
    }
  }
  return static_cast<int>(nbytes);
}

}  // namespace kern
