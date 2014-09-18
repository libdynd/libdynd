//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "inc_gtest.hpp"

#include <dynd/buffer.hpp>
#include <dynd/func/functor_arrfunc.hpp>

using namespace std;
using namespace dynd;

struct func_aux_buffer : aux_buffer {
    int val;
};

struct func_thread_aux_buffer : thread_aux_buffer {
    int val;

    func_thread_aux_buffer() : val(9) {
    }

    func_thread_aux_buffer(func_aux_buffer *aux) : val(aux->val + 3) {
    }
};

int func_with_aux(int src, func_aux_buffer *aux) {
    return src + aux->val;
}

void func_with_thread_aux(int &dst, int src, func_thread_aux_buffer *thread_aux) {
    dst = src + thread_aux->val;
}

void func_with_aux_and_thread_aux(int &dst, int src, func_aux_buffer *aux, func_thread_aux_buffer *thread_aux) {
    dst = src + aux->val + thread_aux->val;
}

TEST(Buffer, Simple) {
    func_aux_buffer aux;
    aux.val = 7;

    nd::arrfunc af = nd::make_functor_arrfunc(func_with_aux);
    EXPECT_EQ(12, af(5, &aux).as<int>());

#if __has_include(<type_traits>)
    std::cout << "(DEBUG) has type_traits" << std::endl;
#else
#endif

//#ifdef DYND_CXX_11
  //  std::cout << "(DEBUG) CXX_11 " << __cplusplus << std::endl;
//#endif

/*
    af = nd::make_functor_arrfunc(func_with_thread_aux);
    EXPECT_EQ(14, af(5).as<int>());

    af = nd::make_functor_arrfunc(func_with_aux_and_thread_aux);
    EXPECT_EQ(22, af(5, &aux).as<int>());
*/
}
