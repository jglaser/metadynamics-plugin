find_library(DFFT_LIBRARY
    dfft
    HINTS
    $ENV{DFFT_LIB}
    $ENV{DFFT_ROOT}/lib
    ${LIB_INSTALL_DIR}
    )

find_path(DFFT_INCLUDE_DIR
    NAMES dfft_common.h
    HINTS
    $ENV{DFFT_INCLUDE}
    $ENV{DFFT_ROOT}/include
    ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES include
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DFFT DEFAULT_MSG DFFT_LIBRARY DFFT_INCLUDE_DIR)
mark_as_advanced(DFFT_LIBRARY DFFT_INCLUDE_DIR)
