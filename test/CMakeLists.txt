add_executable(test_communicator_grid test_communicator_grid.cc)
target_link_libraries(test_communicator_grid _metadynamics ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

if("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")
add_definitions(-DBOOST_TEST_DYN_LINK)
endif("${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" MATCHES "\\.so$")

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
add_test(test_communicator_grid "test_communicator_grid")

