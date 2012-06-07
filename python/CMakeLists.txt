project(pydnd)

cmake_minimum_required(VERSION 2.8)

# Force the default build type to be Release, because a Debug
# build doesn't work properly with the default Python build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(NumPy REQUIRED)
include(UseCython)

# Default install location for Python packages
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PYTHON_SITE_PACKAGES}" CACHE STRING
      "Choose the Python module directory (default site-packages)" FORCE)
endif()

# Require version >= 1.6
if(NUMPY_VERSION_DECIMAL LESS 10600)
    message(FATAL_ERROR,
        "pydnd requires NumPy >= 1.6")
endif()

# TODO: introduce a variable for the DND include dir
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
    ../../include
    src
    )

set(pydnd_CPP_SRC
    src/placement_wrappers.hpp
    src/dtype_functions.hpp
    )
set(pydnd_CYTHON_SRC
    src/_pydnd.pyx
    src/dnd.pxd
    )
set_source_files_properties(${pydnd_CYTHON_SRC} PROPERTIES CYTHON_IS_CXX 1)

source_group("Cython Source" REGULAR_EXPRESSION ".*pyx$")
source_group("Cython Headers" REGULAR_EXPRESSION ".*pxd$")

cython_add_module(_pydnd ${pydnd_CPP_SRC} ${pydnd_CYTHON_SRC})

set_target_properties(_pydnd PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(_pydnd dynamicndarray)

# Install all the Python scripts
install(DIRECTORY pydnd DESTINATION "${CMAKE_INSTALL_PREFIX}"
    FILES_MATCHING PATTERN "*.py")
# Install the module
install(TARGETS _pydnd DESTINATION "${CMAKE_INSTALL_PREFIX}/pydnd")
