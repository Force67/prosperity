// Copyright (C) Force67 2019

#include <base.h>
#include <cstdio>
#include "sys_net.h"
#include "error_table.h"

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void* buffer,
        uint32_t size) {

    if (size > 160)
    return -SysError::eINVAL;

    if (op == 20) {
      *static_cast<uint32_t *>(buffer) = 0xF00D;
      return 0;
    }

    return -SysError::eINVAL;
}

int PS4ABI sys_socketex(const char* name, int32_t domain, int32_t type,
    int32_t protocol) {
    // TOO lazy for now
  return 0;
}

// We host no network stack and none of the system-service processes the guest
// reaches over AF_UNIX sockets (NP, ShellCore, ...). Returning fake success
// (the old null_handler) leaves the guest blocked on a reply that never comes;
// failing the socket up front makes it fall back to its offline/no-service path.
int PS4ABI sys_socket(int32_t domain, int32_t type, int32_t protocol) {
  std::printf("[net] socket(domain=%d type=%d proto=%d) -> EAFNOSUPPORT\n",
              domain, type, protocol);
  return -SysError::eAFNOSUPPORT;
}

int PS4ABI sys_connect(int32_t fd, const void *addr, uint32_t addrlen) {
  return -SysError::eCONNREFUSED;
}

int PS4ABI sys_recvmsg(int32_t fd, void *msg, int32_t flags) {
  return -SysError::eBADF;
}
}