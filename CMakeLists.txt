project(metadynamics)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

find_package(HOOMD 2.6.0 REQUIRED)

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)
# add subdirectories
add_subdirectory(${PROJECT_NAME})
