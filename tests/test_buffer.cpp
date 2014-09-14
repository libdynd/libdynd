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

struct func_aux_buffer : aux_buffer {
    int val;
};

struct func_thread_aux_buffer : thread_aux_buffer {
    func_thread_aux_buffer(func_aux_buffer *) {
    }
};

void func(int &dst, int src, func_aux_buffer *aux, func_thread_aux_buffer *) {
    dst = src + aux->val;
}

TEST(Buffer, Simple) {
    typedef void (*func_type)(int &, int, func_aux_buffer *, func_thread_aux_buffer *);
    typedef void (*func_type)(int &, int, func_aux_buffer *, func_thread_aux_buffer *);

    func_aux_buffer aux;
    aux.val = 5;

    nd::arrfunc af = nd::make_functor_arrfunc(func);
    EXPECT_EQ(af(5, &aux).as<int>(), 10);
}
