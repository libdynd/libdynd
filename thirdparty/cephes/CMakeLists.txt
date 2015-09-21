cmake_minimum_required(VERSION 2.6)
project(cephes)

set(CMAKE_VERBOSE_MAKEFILE 1)

# The ${CMAKE_CURRENT_SOURCE_DIR}/ is required for the static lib builds,
# where this source is included directly in the libdynd build instead
# of being linked separately.
set(cephes_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/airy.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cbrt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/const.c
    ${CMAKE_CURRENT_SOURCE_DIR}/gamma.c
    ${CMAKE_CURRENT_SOURCE_DIR}/j0.c
    ${CMAKE_CURRENT_SOURCE_DIR}/j1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/jv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/polevl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mtherr.c
    ${CMAKE_CURRENT_SOURCE_DIR}/sf_error.c
    ${CMAKE_CURRENT_SOURCE_DIR}/struve.c
    ${CMAKE_CURRENT_SOURCE_DIR}/yn.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mconf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/protos.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rename.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sf_error.h
    )

# Just supporting static library build for now (no .so/.dll yet)
add_library(cephes STATIC ${cephes_SRC})

# On some 64-bit unix platforms it appears to be necessary to
# explicitly say -fPIC to make static linking work
if(UNIX AND (${CMAKE_SIZEOF_VOID_P} MATCHES 8))
    set_target_properties(cephes PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
