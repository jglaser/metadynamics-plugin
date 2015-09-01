project(metadynamics)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

# remove rpath from unit tests and exes
macro(fix_hoomd_rpath target)
if (APPLE)
get_target_property(_target_exe ${target} LOCATION)
add_custom_command(TARGET ${target} POST_BUILD
                          COMMAND install_name_tool ARGS -change @executable_path/../lib/hoomd/python-module/hoomd.so ${HOOMD_LIB} ${_target_exe})
endif (APPLE)
endmacro()


SET(CMAKE_MACOSX_RPATH 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(HOOMD)

# find DFFT lib
if(ENABLE_MPI)
find_package(DFFT REQUIRED)
endif(ENABLE_MPI)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

# add subdirectories
add_subdirectory(cppmodule)
add_subdirectory(pymodule)
add_subdirectory(doc)
add_subdirectory(test)

# overide the CMAKE_INSTALL_PREFIX on the first run to the env HOOMD_PLUGINS_DIR if defined or ${HOOMD_ROOT}
# ${HOOMD_ROOT} will likely be the most common installation directory, but users may want to override
if(DEFINED HOOMD_PLUGINS_DIR)
    set(_default_install_path ${HOOMD_PLUGINS_DIR})
else(DEFINED HOOMD_PLUGINS_DIR)
    set(_default_install_path ${HOOMD_ROOT}/lib/hoomd/python-module)
endif(DEFINED HOOMD_PLUGINS_DIR)

if (NOT _first_run)
    if (HOOMD_FOUND)
        set(CMAKE_INSTALL_PREFIX ${_default_install_path} CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
        SET(_first_run ON CACHE INTERNAL "First configure has run: CMAKE_INSTALL_PREFIX has had its defaults changed" FORCE)
    endif (HOOMD_FOUND)
endif (NOT _first_run)

# install the __init__.py file 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION hoomd_plugins)
