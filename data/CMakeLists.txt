find_program(XACRO xacro)

find_package(choreonoid QUIET)
if(choreonoid_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENHRP REQUIRED openhrp3.1)
  pkg_check_modules(HRPSYS_BASE REQUIRED hrpsys-base)
  set(MODEL_DESTINATION "${OPENHRP_PREFIX}/share/OpenHRP-3.1/robot/panda")
  # Install the VRML model
  install(DIRECTORY model/panda/ DESTINATION ${MODEL_DESTINATION} FILES_MATCHING PATTERN "*.wrl")
endif()

set(PANDA_DESCRIPTION_PATH "${CMAKE_INSTALL_PREFIX}/share/mc_panda")
set(URDF_DESTINATION "${PANDA_DESCRIPTION_PATH}/")
set(CONVEX_DESTINATION "${URDF_DESTINATION}/convex")
set(MESHES_DESTINATION "${URDF_DESTINATION}/meshes")

if(XACRO)
  # Generate a xacro file and the associated urdf
  macro(GENERATE_ROBOT XACRO_IN)
    get_filename_component(XACRO_OUT "${XACRO_IN}" NAME)
    set(XACRO_OUT "${CMAKE_BINARY_DIR}/xacro/${XACRO_OUT}")
    configure_file("${XACRO_IN}" "${XACRO_OUT}" @ONLY)
    get_filename_component(NAME "${XACRO_OUT}" NAME_WE)
    set(URDF_OUT "${CMAKE_BINARY_DIR}/urdf/${NAME}.urdf")
    set(URDF_DEPENDS "${XACRO_OUT}")
    set(EXTRA_DEPENDS "${ARGN}")
    foreach(EXTRA IN LISTS EXTRA_DEPENDS)
      list(APPEND URDF_DEPENDS "${EXTRA}")
    endforeach()
    add_custom_command(OUTPUT "${URDF_OUT}" COMMAND ${XACRO} "${XACRO_OUT}" -o "${URDF_OUT}" DEPENDS ${URDF_DEPENDS})
    add_custom_target(generate_${NAME}_urdf ALL DEPENDS ${URDF_OUT})
    install(FILES "${URDF_OUT}" DESTINATION "${URDF_DESTINATION}")
  endmacro()
else()
  macro(GENERATE_ROBOT XACRO_IN)
    get_filename_component(XACRO_OUT "${XACRO_IN}" NAME)
    get_filename_component(NAME "${XACRO_OUT}" NAME_WE)
    set(URDF_OUT "${CMAKE_BINARY_DIR}/urdf/${NAME}.urdf")
    configure_file(urdf/${NAME}.urdf "${URDF_OUT}" @ONLY)
    install(FILES "${URDF_OUT}" DESTINATION "${URDF_DESTINATION}")
  endmacro()
  install(DIRECTORY meshes/visual DESTINATION "${MESHES_DESTINATION}" FILES_MATCHING PATTERN "*.dae")
endif()

configure_file("xacro/main.xacro" "${CMAKE_BINARY_DIR}/xacro/main.xacro" @ONLY)
configure_file("xacro/main.gazebo.xacro" "${CMAKE_BINARY_DIR}/xacro/main.gazebo.xacro" @ONLY)
configure_file("xacro/panda.xacro" "${CMAKE_BINARY_DIR}/xacro/panda.xacro" @ONLY)
configure_file("xacro/panda.gazebo.xacro" "${CMAKE_BINARY_DIR}/xacro/panda.gazebo.xacro" @ONLY)
configure_file("xacro/foot.xacro" "${CMAKE_BINARY_DIR}/xacro/foot.xacro" @ONLY)
configure_file("xacro/pump.xacro" "${CMAKE_BINARY_DIR}/xacro/pump.xacro" @ONLY)

set(XACRO_DEPENDS "xacro/main.xacro" "xacro/main.gazebo.xacro" "xacro/panda.xacro" "xacro/panda.gazebo.xacro")
generate_robot("xacro/panda_default.urdf.xacro" ${XACRO_DEPENDS})
generate_robot("xacro/panda_hand.urdf.xacro" ${XACRO_DEPENDS})
generate_robot("xacro/panda_foot.urdf.xacro" ${XACRO_DEPENDS} "xacro/foot.xacro")
generate_robot("xacro/panda_pump.urdf.xacro" ${XACRO_DEPENDS} "xacro/pump.xacro")

# Install extra meshes
install(FILES meshes/foot.stl meshes/pump.dae DESTINATION "${MESHES_DESTINATION}")

# Install convex
install(DIRECTORY convex DESTINATION "${URDF_DESTINATION}" FILES_MATCHING PATTERN "*-ch.txt")

# Install rsdf
install(DIRECTORY rsdf DESTINATION "${URDF_DESTINATION}" FILES_MATCHING PATTERN "*.rsdf")
install(DIRECTORY DESTINATION "${URDF_DESTINATION}/rsdf/panda_hand")

macro(GENERATE_CNOID_PROJECT PROJECT_FOLDER)
  set(PROJECT_DESTINATION "${HRPSYS_BASE_PREFIX}/share/hrpsys/samples/${PROJECT_FOLDER}")
  set(MODEL_OUT "${MODEL_DESTINATION}/${PROJECT_FOLDER}.wrl")
  file(READ cnoid/${PROJECT_FOLDER}/joint_positions.cnoid JOINT_POSITIONS)
  string(STRIP "${JOINT_POSITIONS}" JOINT_POSITIONS)
  set(USE_UDP False)
  set(SIM_SCRIPT sim_mc.py)
  set(ROBOT_NAME "${PROJECT_FOLDER}")
  set(SENSOR_PORT_A "LeftHandForceSensor")
  set(CONF_FILE_NAME "Virtual-${PROJECT_FOLDER}-RTC.conf")
  install(FILES "cnoid/${PROJECT_FOLDER}/Virtual-${PROJECT_FOLDER}-RTC.conf" DESTINATION "${CHOREONOID_PLUGIN_DIR}/rtc/")
  set(PROJECT_FILES
    cnoid/common/clear-omninames.sh
    cnoid/common/robot.conf
    cnoid/common/rtc.conf.choreonoid
    cnoid/common/sim_mc.cnoid
    cnoid/common/sim_mc_withTable.cnoid
    cnoid/${PROJECT_FOLDER}/sim_mc.py
    cnoid/${PROJECT_FOLDER}/PDcontroller.conf.choreonoid
    cnoid/${PROJECT_FOLDER}/PDgains_sim.dat
  )
  foreach(F ${PROJECT_FILES})
    get_filename_component(F_NAME "${F}" NAME)
    configure_file(${F} "${CMAKE_BINARY_DIR}/cnoid/${PROJECT_FOLDER}/${F_NAME}" @ONLY)
  endforeach()
  set(USE_UDP True)
  set(SIM_SCRIPT sim_mc_udp.py)
  set(SENSOR_PORT_A "lhsensor")
  configure_file(cnoid/${PROJECT_FOLDER}/sim_mc.py "${CMAKE_BINARY_DIR}/cnoid/${PROJECT_FOLDER}/${SIM_SCRIPT}" @ONLY)
  configure_file(cnoid/common/sim_mc.cnoid "${CMAKE_BINARY_DIR}/cnoid/${PROJECT_FOLDER}/sim_mc_udp.cnoid" @ONLY)
  configure_file(cnoid/common/sim_mc_withTable.cnoid "${CMAKE_BINARY_DIR}/cnoid/${PROJECT_FOLDER}/sim_mc_udp.cnoid" @ONLY)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/cnoid/${PROJECT_FOLDER}/" DESTINATION "${PROJECT_DESTINATION}")
  install(PROGRAMS cnoid/common/clear-omninames.sh DESTINATION "${PROJECT_DESTINATION}")
endmacro()

if(choreonoid_FOUND)
  generate_cnoid_project(panda_default)
  generate_cnoid_project(panda_hand)
  generate_cnoid_project(panda_foot)
  generate_cnoid_project(panda_pump)
endif()

set(PANDA_DESCRIPTION_PATH "${PANDA_DESCRIPTION_PATH}" PARENT_SCOPE)
