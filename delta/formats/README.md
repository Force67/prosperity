# delta/formats

Parsers for the containers titles and firmware ship in. Depends on nothing
else in delta; everything lives in namespace `formats`.

| unit | hides |
|---|---|
| `pkg_filesystem` | PS4 fake-signed PKG: entry table, PFS decryption, the inner filesystem |
| `ufs2_filesystem` | UFS2 images (PS5 ffpkg backups) |
| `archive_filesystem` | titles read straight out of .rar/.zip, decompressed on demand |
| `archive_backend`, `archive_rar`, `archive_zip` | one backend per archive format behind `ArchiveBackend` |
| `pup_reader` | decrypted PS4/PS5 firmware PUPs |
| `slb2_reader` | SLB2 firmware bundles |
| `fself` | fake-signed SELF to plain ELF |
| `title_metadata` | param.sfo / param.json fields and the SDK version string |
