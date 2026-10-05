# Times SPIR-V validate + optimize over DELTA_GPU_SPV_RAW_DUMP modules.
add_executable(spv_bench spv_bench.cc)
target_link_libraries(spv_bench PRIVATE delta_gpu)
if(TARGET mimalloc-obj)
  target_sources(spv_bench PRIVATE $<TARGET_OBJECTS:mimalloc-obj>)
endif()
