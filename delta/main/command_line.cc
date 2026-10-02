#include "main/command_line.h"

#include "base/standard_streams.h"
#include "base/strings/format.h"

namespace cli {
namespace {

void PrintHelp() {
  constexpr char kHelp[] = R"(
                           @@@@@@
                          @@@@@@
                          @@@@@@
                        @@@@@@::::
                        %%%%%%::::
                      %%%%%%    ::::
                      %%%%%%    ::::
                    ######        ::::
                    ######    ##  ::::
                  ######      ##    ::::
                  ######    ######  ::::
                ######      ######    ::::
                ######    ==========  ::::
              ######      ==========    ::::
              ######    ==============  ::::
            ======      ==============    ::::
            ======    ==================  ::::
          ======      ==================    ::::
          ======    ======================  ::::
        ======    ================================

  Prosperity | PlayStation 4 & 5 emulator

Usage: ps4delta [options] <game> [-- guest arguments]

Game: .pkg, .ffpkg, .zip, .rar, app directory, or eboot.bin

  -h, --help                Show this help
  -v, --version             Show the build version and CPU backend
  --data-dir PATH           Find modules/ and game_profiles/ here
  --configure-ps4-fw DIR    Import or replace stored PS4 firmware modules
  --configure-ps5-fw DIRS   Import or replace stored PS5 firmware modules
  --ps4-modules DIR         Override the PS4 module folder for this run
  --ps5-modules DIRS        PS5 module directories, separated by ':'
  --headless                Render without presenting to a window
  --dump-frames DIR         Write captured frames to this directory
  --profile FILE|off        Use a game profile, or disable profiles
  --options FILE            Load runtime options (also --options=FILE)
  --dump-options[=FILE]     Dump runtime options, then exit if no game given
  +DELTA_NAME=VALUE         Override a runtime option
  --                        Pass remaining arguments through unchanged

Examples:
  ps4delta --configure-ps4-fw /dumps/ps4/modules
  ps4delta /games/game.pkg
  ps4delta --data-dir ~/.local/share/prosperity /games/game.pkg
  ps4delta --ps5-modules /dumps/ps5/common/lib /games/game.ffpkg
  ps4delta --profile off /games/app -- -debug

PS4 modules: use --configure-ps4-fw to import decrypted firmware modules.
             Otherwise use modules/ next to the binary or inside --data-dir.
Keep PS5 modules in a separate directory.
Setup: docs/installation.md
)";
  base::WriteStandardOutput(kHelp, sizeof(kHelp) - 1);
}

}  // namespace

CommandLine Parse(int argc, char** argv) {
  CommandLine result;
  if (argc == 1) {
    PrintHelp();
    result.exit = true;
    return result;
  }

  bool passthrough = false;
  for (int i = 1; i < argc; ++i) {
    const base::String arg(argv[i]);
    if (!passthrough && arg == "--") {
      passthrough = true;
      continue;
    }
    if (!passthrough) {
      if (arg == "--help" || arg == "-h") {
        PrintHelp();
        result.exit = true;
        return result;
      }
      if (arg == "--version" || arg == "-v") {
#if defined(DELTA_BACKEND_NATIVE)
        constexpr const char* backend = "NATIVE";
#else
        constexpr const char* backend = "FEX";
#endif
        const auto version = base::Format("Prosperity (ps4delta) {} | {}\n",
                                          rsc_productversion, backend);
        base::WriteStandardOutput(version.data(), version.size());
        result.exit = true;
        return result;
      }
      if (arg == "--headless") {
        result.options.emplace_back("+DELTA_GPU_NOPRESENT=1");
        continue;
      }
      const char* flags[] = {"--data-dir",         "--ps5-modules",
                             "--profile",          "--options",
                             "--dump-frames",      "--configure-ps4-fw",
                             "--configure-ps5-fw", "--ps4-modules"};
      const char* names[] = {"+DELTA_DATA_DIR=",
                             "+DELTA_PS5_MODULES=",
                             "+DELTA_PROFILE=",
                             "--options=",
                             "+DELTA_GPU_DUMP_DIR=",
                             nullptr,
                             nullptr,
                             "+DELTA_PS4_MODULES="};
      bool matched = false;
      for (size_t f = 0; f < 8; ++f) {
        const size_t len = base::CountStringLength(flags[f]);
        if (arg != flags[f] &&
            arg.compare(0, len + 1, base::String(flags[f]) + "=") != 0)
          continue;
        base::String value;
        if (arg.size() > len) {
          value = arg.substr(len + 1);
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
          value = argv[++i];
        }
        if (value.empty()) {
          const auto error =
              base::Format("ps4delta: {} needs a value\n", flags[f]);
          base::WriteStandardError(error.data(), error.size());
          result.exit = true;
          result.exit_code = 2;
          return result;
        }
        if (arg.starts_with("--dump-frames"))
          result.options.emplace_back("+DELTA_GPU_DUMP=1");
        if (arg.starts_with("--configure-ps5-fw"))
          result.configure_ps5_fw = value;
        else if (arg.starts_with("--configure-ps4-fw"))
          result.configure_ps4_fw = value;
        else
          result.options.emplace_back(base::String(names[f]) + value);
        matched = true;
        break;
      }
      if (matched)
        continue;
      if (arg == "--dump-options" || arg.starts_with("--dump-options=")) {
        result.dump_options = true;
        result.options.push_back(arg);
        continue;
      }
      if (arg.starts_with("+")) {
        result.options.push_back(arg);
        continue;
      }
      if (result.game.empty() && arg.starts_with("-")) {
        const auto error =
            base::Format("ps4delta: unknown option '{}' (see --help)\n", arg);
        base::WriteStandardError(error.data(), error.size());
        result.exit = true;
        result.exit_code = 2;
        return result;
      }
    }
    if (result.game.empty())
      result.game = arg;
    else
      result.guest_args.push_back(arg);
  }
  if (result.game.empty() && !result.dump_options &&
      result.configure_ps4_fw.empty() && result.configure_ps5_fw.empty()) {
    constexpr char kError[] = "ps4delta: missing game path (see --help)\n";
    base::WriteStandardError(kError, sizeof(kError) - 1);
    result.exit = true;
    result.exit_code = 2;
  }
  return result;
}

}  // namespace cli
