add_delta_test(object_table_test
  SOURCES object_table_test.cc
  DEPS delta_kern)

add_delta_test(appcontent_test
  SOURCES appcontent_test.cc
  DEPS delta_kern)

add_delta_test(ime_test
  SOURCES ime_test.cc
  DEPS delta_kern)
add_delta_test(common_dialog_test
  SOURCES common_dialog_test.cc
  DEPS delta_kern)
add_delta_test(savedata_test
  SOURCES savedata_test.cc
  DEPS delta_kern)
add_delta_test(savedata_lle_test
  SOURCES savedata_lle_test.cc
  DEPS delta_kern)
add_delta_test(flexible_memory_lle_test
  SOURCES flexible_memory_lle_test.cc
  DEPS delta_kern xbyak)
add_delta_test(ret_trace_test
  SOURCES ret_trace_test.cc
  DEPS delta_kern xbyak)

add_delta_test(usb_dev_test
  SOURCES usb_dev_test.cc
  DEPS delta_kern)
add_delta_test(pipe_dev_test
  SOURCES pipe_dev_test.cc
  DEPS delta_kern xbyak)

add_delta_test(dma_dev_test
  SOURCES dma_dev_test.cc
  DEPS delta_kern)

add_delta_test(gc_dev_test
  SOURCES gc_dev_test.cc
  DEPS delta_kern)
add_delta_test(dce_ps5_test
  SOURCES dce_ps5_test.cc
  DEPS delta_kern)
add_delta_test(dce_ps4_test
  SOURCES dce_ps4_test.cc
  DEPS delta_kern)
set_tests_properties(dce_ps5_test PROPERTIES ENVIRONMENT
  "DELTA_GPU_RENDER_THREAD=0;DELTA_GPU_NOPRESENT=1;SDL_VIDEODRIVER=offscreen")
set_tests_properties(gc_dev_test PROPERTIES ENVIRONMENT
  "DELTA_GPU_RENDER_THREAD=0;DELTA_GPU_NOPRESENT=1;SDL_VIDEODRIVER=offscreen")

add_delta_test(audio_queue_test
  SOURCES audio_queue_test.cc
  DEPS delta_deps)
add_delta_test(audio_daemon_test
  SOURCES audio_daemon_test.cc
  DEPS delta_kern)
add_delta_test(audio_in_lle_test
  SOURCES audio_in_lle_test.cc
  DEPS delta_kern xbyak)
add_delta_test(agc_lle_test
  SOURCES agc_lle_test.cc
  DEPS delta_kern xbyak)
add_delta_test(audio_in_test
  SOURCES audio_in_test.cc
  DEPS delta_kern)

add_delta_test(sys_info_test
  SOURCES sys_info_test.cc
  DEPS delta_kern)
set_tests_properties(sys_info_test PROPERTIES ENVIRONMENT "DELTA_PS4_NEO=0")
add_test(NAME sys_info_default_test
  COMMAND ${CMAKE_COMMAND} -E env --unset=DELTA_PS4_NEO $<TARGET_FILE:sys_info_test>)
add_test(NAME sys_info_neo_test COMMAND sys_info_test)
set_tests_properties(sys_info_neo_test PROPERTIES ENVIRONMENT "DELTA_PS4_NEO=2")

add_delta_test(vfs_test
  SOURCES vfs_test.cc
  DEPS delta_kern)

add_delta_test(umtx_op_test
  SOURCES umtx_op_test.cc
  DEPS delta_kern)
add_delta_test(netctl_test
  SOURCES netctl_test.cc
  DEPS delta_kern)
add_delta_test(playgo_test
  SOURCES playgo_test.cc
  DEPS delta_kern)
add_delta_test(user_service_test
  SOURCES user_service_test.cc
  DEPS delta_kern)

add_delta_test(mdbg_service_test
  SOURCES mdbg_service_test.cc
  DEPS delta_kern)
set_tests_properties(mdbg_service_test PROPERTIES TIMEOUT 10)

# A device now takes the object table it registers with rather than the whole
# process, so these two build the unit under test into the test binary and link
# no emulator at all -- which is the acceptance test for the boundary.
add_delta_test(console_dev_test
  SOURCES console_dev_test.cc
    ../ps4/dev/console_dev.cc ../ps4/dev/file_dev.cc ../save_storage.cc
    ../object.cc ../object_table.cc
  DEPS delta_deps)

add_delta_test(hid_dev_test
  SOURCES hid_dev_test.cc
    ../ps4/dev/hid_dev.cc ../ps4/dev/file_dev.cc ../ps4/controller_ports.cc
    ../save_storage.cc
    ../object.cc ../object_table.cc
  DEPS delta_deps)

add_delta_test(mbus_test
  SOURCES mbus_test.cc
  DEPS delta_kern)

add_delta_test(ajm_dev_test
  SOURCES ajm_dev_test.cc
  DEPS delta_kern)

add_delta_test(ajm_decode_test
  SOURCES ajm_decode_test.cc
  DEPS delta_kern)

add_delta_test(shm_test
  SOURCES shm_test.cc
  DEPS delta_kern)

add_delta_test(system_settings_test
  SOURCES system_settings_test.cc
  DEPS delta_kern)

add_delta_test(mapped_prefix_test
  SOURCES mapped_prefix_test.cc
  DEPS delta_kern)
