if(UNIX AND NOT ANDROID)
  add_delta_test(settings_test
    SOURCES settings_test.cc
    DEPS delta_deps)
  add_delta_test(firmware_config_test
    SOURCES firmware_config_test.cc ../firmware_config.cc ../game_firmware.cc ../game_directory.cc
    DEPS delta_ui delta_kern delta_formats)
  add_delta_test(recent_games_test
    SOURCES recent_games_test.cc command_line_test.cc ../recent_games.cc ../command_line.cc
    DEPS delta_ui)
endif()
