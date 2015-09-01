# remove rpath from unit tests and exes
macro(fix_rpath target)
if (APPLE)
get_target_property(_target_exe ${target} LOCATION)
add_custom_command(TARGET ${target} POST_BUILD
                          COMMAND install_name_tool ARGS -change @executable_path/../lib/hoomd/python-module/hoomd.so ${HOOMD_LIB} ${_target_exe})
add_custom_command(TARGET ${target} POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/_metadynamics.so ${CMAKE_CURRENT_BINARY_DIR}/../cppmodule/_metadynamics.so ${_target_exe})
endif (APPLE)
endmacro()


# Need to define NO_IMPORT_ARRAY in every file but hoomd_module.cc
set_source_files_properties(test_communicator_grid.cc PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

add_executable(test_communicator_grid test_communicator_grid.cc)
target_link_libraries(test_communicator_grid ${HOOMD_LIBRARIES} _metadynamics ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
fix_cudart_rpath(test_communicator_grid)
fix_rpath(test_communicator_grid)

SET_TARGET_PROPERTIES (test_communicator_grid
    PROPERTIES BUILD_WITH_INSTALL_RPATH 1
    INSTALL_NAME_DIR "@executable_path/../cppmodule"
)
#if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
add_definitions(-DBOOST_TEST_DYN_LINK)
#endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cppmodule)

include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
  set_target_properties(test_communicator_grid PROPERTIES
  COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
set_target_properties(test_communicator_grid PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

#add_test(test_communicator_grid "test_communicator_grid")

