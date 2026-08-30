# Machine-enforced module layering: which module may include which. Every
# module shares one include root, so a wrong-way dependency compiles silently
# and only this rejects it. See check_module_layering.py for the rules.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME module_layering
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_module_layering.py
            ${CMAKE_SOURCE_DIR})
endif()
