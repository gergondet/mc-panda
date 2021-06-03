configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/config.h")

set(MC_PANDA_PREFIX "MC_PANDA")
configure_file(api.in.h "${CMAKE_CURRENT_BINARY_DIR}/include/mc_panda/api.h")

set(MC_PANDA_PREFIX "MC_PANDA_DEVICES")
configure_file(api.in.h "${CMAKE_CURRENT_BINARY_DIR}/include/mc_panda/devices/api.h")

# Panda custom devices library
set(devices_HDR
  devices/Robot.h
  devices/Pump.h
  "${CMAKE_CURRENT_BINARY_DIR}/include/mc_panda/devices/api.h"
)
set(devices_SRC
  devices/Robot.cpp
  devices/Pump.cpp
)

add_library(mc_panda_devices SHARED ${devices_SRC} ${devices_HDR})
add_library(mc_panda::devices ALIAS mc_panda_devices)
set_target_properties(mc_panda_devices PROPERTIES EXPORT_NAME devices)
target_include_directories(mc_panda_devices PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>$<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
target_link_libraries(mc_panda_devices PUBLIC mc_rtc::mc_rbdyn Franka::Franka)
target_compile_definitions(mc_panda_devices PRIVATE -DMC_PANDA_DEVICES_EXPORTS)
install(FILES ${devices_HDR} DESTINATION include/mc_panda/devices)
install(TARGETS mc_panda_devices
  EXPORT mc_pandaTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Panda robot module library
set(panda_HDR
  panda.h
  "${CMAKE_CURRENT_BINARY_DIR}/include/mc_panda/api.h"
)

set(panda_SRC
  panda.cpp
)

add_library(mc_panda SHARED panda.cpp ${panda_SRC} ${panda_HDR})
add_library(mc_panda::panda ALIAS mc_panda)
set_target_properties(mc_panda PROPERTIES EXPORT_NAME panda)
target_include_directories(mc_panda PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
target_include_directories(mc_panda INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(mc_panda PUBLIC mc_panda::devices)
target_compile_definitions(mc_panda PRIVATE -DMC_PANDA_EXPORTS)
install(FILES ${panda_HDR} DESTINATION include/mc_panda)
install(TARGETS mc_panda
  EXPORT mc_pandaTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

add_robot(panda module.cpp "")
target_link_libraries(panda PRIVATE mc_panda::panda)

# Panda custom states library
add_subdirectory(states)

install(EXPORT mc_pandaTargets
  FILE mc_pandaTargets.cmake
  NAMESPACE mc_panda::
  DESTINATION lib/cmake/mc_panda)
