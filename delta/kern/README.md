# delta/kern

The guest's kernel: one emulated process, its loaded modules, the lv2 syscall
layer and the devices behind /dev. Namespace `kern`.

| unit | hides |
|---|---|
| `process` | the running title: modules, object table, guest memory map, entry |
| `module` | loading, relocating and linking one SELF/ELF image |
| `object`, `object_table`, `object_ref` | refcounted kernel objects behind guest handles |
| `vm_map` | the guest's mappings and their SCE protections (FreeBSD's vm_map) |
| `guest_va_space` | host address ranges reserved up front for the guest's fixed maps |
| `vfs`, `vfs_providers` | the guest path namespace and the mounts behind /app0 and friends |
| `thread_names` | naming host threads after the guest thread they run |
| `crash` | the fatal-signal handler and its guest-aware dump |
| `lv2/` | syscalls, one `sys_<topic>` unit per FreeBSD/SCE topic; `ps4/` and `ps5/` hold the number tables |
| `ps4/dev`, `ps5/dev` | the /dev devices, one per node |
| `ps4/audio_daemon` | the LLE audio daemon feeding the host sink |
| `ipmi/` | the IPMI service registry and its services |
| `probe/` | opt-in research instrumentation (write watches, heap profiles, traces) |

Guest ABI spellings (errno and flag constants, BSD/SCE struct names) stay as
the SDK and FreeBSD write them, behind NOLINT markers.
