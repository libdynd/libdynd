//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "inc_gtest.hpp"

#include "dynd/buffer.hpp"
#include "dynd/func/functor_arrfunc.hpp"

using namespace std;
using namespace dynd;

struct func_auxiliary_buffer : nd::auxiliary_buffer {
};

struct func_thread_local_buffer : nd::thread_local_buffer {
};

// thread_local_buffer *, thread_local_buffer &

TEST(Buffer, Untitled) {
    typedef int (*func_type)(int, func_auxiliary_buffer *, func_thread_local_buffer *);

    std::cout << is_auxiliary_buffered<func_type>::value << std::endl;
    std::cout << is_thread_local_buffered<func_type>::value << std::endl;

//    std::cout << dynd::count<func_type, func_thread_local_buffer *>::value << std::endl;
//    std::cout << is_thread_local_buffered<func_type>::value << std::endl;
//    std::cout << is_auxiliary_buffered<func_type>::value << std::endl;

    std::exit(-1);
}
