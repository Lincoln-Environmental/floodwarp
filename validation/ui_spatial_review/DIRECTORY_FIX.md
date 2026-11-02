# IDE archive-directory correction

The reported launch used `src/floodwarp` as the working directory. Archive
lookup previously searched only that directory and its children; a blank
archive field then resolved to the source directory and failed on DATA_LICENSE.md.

Checkout launches now resolve the project root for browsing/default outputs,
and archive discovery checks the launch directory and ancestors. Explicit empty
or wrong archive selections fail with a useful message. Input checksums and
licence requirements are retained; the source directory is never treated as an
archive. The map layout is unchanged.

Main source `1617351dfab5038639f41bd1a8b9b108a70be998`: 1,027 CPU passes, 634 skips.
Dev source `96ca6e311d24e4d27cda90a4c4696f238827f033`: 1,046 CPU passes, 642 skips.
Both retain 350 experimental warnings. Focused main server/spatial controls:
23 passed. No numerical source change or new hydraulic qualification is claimed.

An actual Firefox check launched the server from the reported source directory
on a separate test port: auto-loaded the complete Orton Bradley setup, displayed
the existing map and 212 contacts, rejected empty/source folders clearly, then
reloaded the correct archive. The user's server/jobs were not stopped.
The corrected installed v0.1.3 wheel also resolved that working directory and
loaded all 212 contacts; pip check passed.

[Record and updated artifact checksums](directory_fix.json). The corrected local
v0.1.3 files supersede the earlier preparation, which is preserved under
`outputs/release/v0.1.3-before-directory-fix`. Older release tags/assets remain
unchanged. Earlier main verification describes its original source/distributions;
this record identifies the directory fix separately. Restart the UI process to
load updated Python code.
