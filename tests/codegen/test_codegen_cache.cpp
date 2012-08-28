//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dnd/codegen/codegen_cache.hpp>

using namespace std;
using namespace dnd;

static int int_float_fn1(float x) {
    return (int)(x * 2);
}

static int int_float_fn2(float x) {
    return (int)(x + 2);
}

static unsigned int uint_float_fn1(float x) {
    return (unsigned int)(x - 2);
}

TEST(CodeGenCache, UnaryCaching) {
    codegen_cache cgcache;
    unary_specialization_kernel_instance op_int_float1, op_int_float2;
    // Generate two adapted functions with different function pointers
    cgcache.codegen_unary_function_adapter(make_dtype<int>(),
                                           make_dtype<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&int_float_fn1),
                                           NULL,
                                           op_int_float1);
    
    cgcache.codegen_unary_function_adapter(make_dtype<int>(),
                                           make_dtype<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&int_float_fn2),
                                           NULL,
                                           op_int_float2);

    // The adapter kernel should have been reused
    EXPECT_EQ(op_int_float1.specializations[0], op_int_float2.specializations[0]);

    unary_specialization_kernel_instance op_uint_float1;
    cgcache.codegen_unary_function_adapter(make_dtype<unsigned int>(),
                                           make_dtype<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&uint_float_fn1),
                                           NULL,
                                           op_uint_float1);

    // int and uint look the same at the assembly level, so it should have reused the kernel
    EXPECT_EQ(op_int_float1.specializations[0], op_uint_float1.specializations[0]);
}
