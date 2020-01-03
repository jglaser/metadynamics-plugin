# Maintainer: jglaser

set(COMPONENT_NAME metadynamics)

set(_${COMPONENT_NAME}_sources
    module.cc
    IntegratorMetaDynamics.cc
    LamellarOrderParameter.cc
    LamellarOrderParameterGPU.cc
    OrderParameterMesh.cc
    OrderParameterMeshGPU.cc
    WellTemperedEnsemble.cc
    CollectiveWrapper.cc
    IndexGrid.cc
    CollectiveVariable.cc
    AspectRatio.cc
    Density.cc
    SteinhardtQl.cc
    )

set(_${COMPONENT_NAME}_cu_sources
    IntegratorMetaDynamics.cu
    LamellarOrderParameterGPU.cu
    OrderParameterMeshGPU.cu
    WellTemperedEnsemble.cu
    )

include_directories(${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/dfftlib/src)

pybind11_add_module (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_${COMPONENT_NAME}_cu_sources} NO_EXTRAS)

fix_cudart_rpath(_${COMPONENT_NAME})

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd HOOMD::_md )

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# install the library
install(TARGETS _${COMPONENT_NAME}
        EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files
    __init__.py
    integrate.py
    cv.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
endif()
