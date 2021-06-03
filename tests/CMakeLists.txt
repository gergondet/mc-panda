add_executable(test_panda_module test_module.cpp)
target_link_libraries(test_panda_module PRIVATE mc_rtc::mc_rbdyn)
configure_file(test_module.in.h ${CMAKE_CURRENT_BINARY_DIR}/panda/test_module.cmake.h)
file(GENERATE
     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/panda/$<CONFIGURATION>/test_module.h
     INPUT  ${CMAKE_CURRENT_BINARY_DIR}/panda/test_module.cmake.h)
target_include_directories(test_panda_module PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/panda/$<CONFIGURATION>)
add_test(test_panda_module test_panda_module)

add_executable(test_linking test_linking.cpp)
target_link_libraries(test_linking PRIVATE mc_panda)
add_test(test_linking_panda_module test_linking)
