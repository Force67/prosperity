if(UNIX AND NOT ANDROID)
  add_delta_test(firmware_config_test
    SOURCES firmware_config_test.cc ../firmware_config.cc
    DEPS delta_ui)
  add_delta_test(recent_games_test
    SOURCES recent_games_test.cc command_line_test.cc ../recent_games.cc ../command_line.cc
    DEPS delta_ui)
endif()
