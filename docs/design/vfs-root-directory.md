# VFS root directory
Status: implemented

## Context
Prosperity exposes mounted guest paths such as `/app0` and `/savedata0`, but no
mount owns `/` itself. As a result, `stat("/")` returns `ENOENT` even though the
guest filesystem root must exist. FIOS uses the root as the base case while
creating parent directories, so a missing root can turn that walk into
unbounded recursion.

## Decision
Treat the exact path `/` as a synthetic directory in `vfs::stat`, with size
zero. Leave path resolution and mounted-file behavior unchanged.

## Alternatives
- Mount a host directory at `/`: rejected because it would expose host content
  and interfere with longest-prefix mount resolution.
- Special-case FIOS or the title: rejected because root-directory existence is
  a VFS contract.
- Add full synthetic-root enumeration: deferred because the observed failure
  only requires correct metadata.

## Consequences
Guest code can stat the filesystem root without assigning it a host backing.
Opening or enumerating `/` remains unsupported.

## Acceptance
- A kernel test verifies that `vfs::stat("/")` succeeds and reports a zero-size
  directory.
- Existing tests pass.
- Uncharted 2 no longer recursively probes `/` until its FIOS fiber overflows.
