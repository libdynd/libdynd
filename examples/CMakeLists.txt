#
# Copyright (C) 2011-16 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

include_directories(
    ../include
  )

add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example libdynd libdyndt)

add_executable(abi_sparse_example abi_sparse.cpp)
target_link_libraries(abi_sparse_example dynd_abi)
set_property(TARGET abi_sparse_example PROPERTY CXX_STANDARD 17)
