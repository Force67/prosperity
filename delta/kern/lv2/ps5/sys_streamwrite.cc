/*
 * PS4Delta : PS4 emulation and research project
 *
 * streamwrite (0x2c5): the PS5's SSD ring buffer behind sceKernelStreamWrite*.
 * A title creates one buffer of up to 2 GiB, opens it for I/O, appends 64 KiB
 * blocks to it and reads back what the ring still holds. The kernel keeps it
 * in the SSD's SLC area; here it is lazily backed host memory with the same
 * states, checks and replies.
 */

#include <sys/mman.h>
#include <cstring>
#include "base/arch.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/process.h"

namespace kern {
namespace {
enum Op {
  kCreate = 1,  // {size, u32* handle}
  kDelete = 17,
  kStat = 18,   // {stat*}
  kOpen = 19,   // {u32 must be 0}
  kWrite = 20,  // {buf, len, u32 flags, u64* offset, u64* held}
  kRead = 21,   // {buf, offset, len, u32 flags}
  kClose = 22,  // {u32 must be 0}
};

constexpr u64 kMaxSize = 0x80000000;
constexpr u64 kBlock = 0x10000;
// The handle of the first (and here only) app slot.
constexpr u32 kHandle = 0x10000;

enum State { kNone, kCreated, kOpened };

struct Ring {
  State state = kNone;
  u8* data = nullptr;
  u64 capacity = 0;
  u64 head = 0;  // bytes ever written; the ring holds the last `capacity`

  u64 Tail() const { return head > capacity ? head - capacity : 0; }
  u64 Held() const { return head - Tail(); }
};

base::Mutex g_mutex;
Ring g_ring;

void Release() {
  if (g_ring.data)
    ::munmap(g_ring.data, g_ring.capacity);
  g_ring = Ring{};
}

const guest::SessionReset g_reset([] { Release(); });

// Copies between the ring and guest memory at logical offset `at`, wrapping.
void Transfer(u64 at, u8* guest, u64 len, bool to_ring) {
  while (len) {
    const u64 pos = at % g_ring.capacity;
    const u64 n = len < g_ring.capacity - pos ? len : g_ring.capacity - pos;
    if (to_ring)
      std::memcpy(g_ring.data + pos, guest, n);
    else
      std::memcpy(guest, g_ring.data + pos, n);
    at += n;
    guest += n;
    len -= n;
  }
}

bool Mapped(const void* p, size_t n) {
  return p && host_memory::IsMemoryRangeMapped(p, n);
}

template <typename T>
T Arg(const u8* a, size_t off) {
  T v;
  std::memcpy(&v, a + off, sizeof(T));
  return v;
}

int Create(int fd, const u8* a) {
  const u64 size = Arg<u64>(a, 0);
  auto* handle = Arg<u32*>(a, 8);
  if (fd != -1 || size > kMaxSize)
    return -SysError::eINVAL;
  if (g_ring.state != kNone)
    return -SysError::eBUSY;
  const u64 capacity = (size + kBlock - 1) & ~(kBlock - 1);
  if (!capacity)
    return -SysError::eINVAL;
  if (!Mapped(handle, sizeof(u32)))
    return -SysError::eFAULT;
  void* p = ::mmap(nullptr, capacity, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED)
    return -SysError::eNOSPC;
  g_ring = Ring{kCreated, static_cast<u8*>(p), capacity, 0};
  *handle = kHandle;
  return 0;
}

int Stat(const u8* a) {
  auto* out = Arg<u64*>(a, 0);
  if (!Mapped(out, 0x40))
    return -SysError::eFAULT;
  std::memset(out, 0, 0x40);
  out[0] = g_ring.capacity;
  out[2] = g_ring.head;
  out[3] = g_ring.Held();
  return 0;
}

int Write(const u8* a) {
  auto* buf = Arg<u8*>(a, 0);
  const u64 len = Arg<u64>(a, 8);
  auto* offset = Arg<u64*>(a, 0x18);
  auto* held = Arg<u64*>(a, 0x20);
  if (Arg<u32>(a, 0x10) || g_ring.state != kOpened)
    return -SysError::eINVAL;
  if ((len & (kBlock - 1)) || len > g_ring.capacity)
    return -SysError::eINVAL;
  if (len && !Mapped(buf, len))
    return -SysError::eFAULT;
  const u64 at = g_ring.head;
  Transfer(at, buf, len, true);
  g_ring.head += len;
  if (!Mapped(offset, sizeof(u64)) || !Mapped(held, sizeof(u64)))
    return -SysError::eFAULT;
  *offset = at;
  *held = g_ring.Held();
  return 0;
}

// Only what the ring still holds can be read back.
int Read(const u8* a) {
  auto* buf = Arg<u8*>(a, 0);
  const u64 offset = Arg<u64>(a, 8);
  const u64 len = Arg<u64>(a, 0x10);
  if (Arg<u32>(a, 0x18) || (len & (kBlock - 1)) || g_ring.state != kOpened)
    return -SysError::eINVAL;
  if (offset < g_ring.Tail() || len > g_ring.head - offset)
    return -SysError::eINVAL;
  if (len && !Mapped(buf, len))
    return -SysError::eFAULT;
  Transfer(offset, buf, len, false);
  return 0;
}
}  // namespace

int PS4ABI Ps5Streamwrite(int op, int fd, size_t size, const void* arg) {
  // Categories 0x10000..0xffffff are refused unless they are 0x01xxxx.
  const Process* proc = Process::GetActive();
  if (!proc)
    return -SysError::eNOENT;
  const u32 category = proc->GetAppCategory();
  if (category - 0x1000000 >= 0xff010000 && (category & 0xff0000) != 0x10000)
    return -SysError::eINVAL;
  if (size < 1 || size > 64)
    return -SysError::eINVAL;
  if (!Mapped(arg, size))
    return -SysError::eFAULT;
  u8 a[64] = {};
  std::memcpy(a, arg, size);

  // Every operation names the buffer by its handle, except the one making it.
  base::LockGuard<base::Mutex> lk(g_mutex);
  const bool known = g_ring.state != kNone && static_cast<u32>(fd) == kHandle;
  switch (op) {
    case kCreate:
      return size == 0x18 ? Create(fd, a) : -SysError::eINVAL;
    case kDelete:
      if (size != 8 || !known)
        return -SysError::eINVAL;
      if (g_ring.state == kOpened)
        return -SysError::eBUSY;
      Release();
      return 0;
    case kStat:
      return size == 0x10 && known ? Stat(a) : -SysError::eINVAL;
    case kOpen:
      if (size != 8 || !known || Arg<u32>(a, 0) || g_ring.state != kCreated)
        return -SysError::eINVAL;
      g_ring.state = kOpened;
      return 0;
    case kWrite:
      return size == 0x28 && known ? Write(a) : -SysError::eINVAL;
    case kRead:
      return size == 0x20 && known ? Read(a) : -SysError::eINVAL;
    case kClose:
      if (size != 8 || !known || Arg<u32>(a, 0) || g_ring.state != kOpened)
        return -SysError::eINVAL;
      g_ring.state = kCreated;
      return 0;
  }
  return -SysError::eNOSYS;  // the kernel's 15 unassigned operations
}

}  // namespace kern
