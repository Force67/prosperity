#pragma once

#include "base/arch.h"
#include "base/strings/xstring.h"
#include "io/file.h"

namespace kern::save_storage {
i32 Info(const base::String& root, u64& total, u64& free);
bool OpenWritable(io::File& file,
                  const base::String& root,
                  const base::String& path,
                  bool create,
                  bool truncate);
bool OpenRead(io::File& file,
              const base::String& root,
              const base::String& path);
void Close(io::File& file);
i64 Write(io::File& file,
          const base::String& root,
          const void* data,
          size_t size);
i32 Truncate(io::File& file, const base::String& root, i64 length);
i32 Truncate(const base::String& root, const base::String& path, i64 length);
i32 Replace(const base::String& root,
            const base::String& path,
            const void* data,
            size_t size);
i32 Remove(const base::String& root, const base::String& path);
i32 RemoveSave(const base::String& root);
i32 Backup(const base::String& root);
i32 CheckBackup(const base::String& root);
i32 RestoreBackup(const base::String& root);
i32 Rename(const base::String& root,
           const base::String& from,
           const base::String& to);
bool MakeDir(const base::String& root, const base::String& path);
}  // namespace kern::save_storage
