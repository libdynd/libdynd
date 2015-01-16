//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#if 0 // TODO reenable
#include <dynd/codegen/codegen_cache.hpp>

using namespace std;
using namespace dynd;

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
    kernel_instance<unary_operation_pair_t> op_int_float1, op_int_float2;
    // Generate two adapted functions with different function pointers
    cgcache.codegen_unary_function_adapter(ndt::make_type<int>(),
                                           ndt::make_type<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&int_float_fn1),
                                           NULL,
                                           op_int_float1);
    
    cgcache.codegen_unary_function_adapter(ndt::make_type<int>(),
                                           ndt::make_type<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&int_float_fn2),
                                           NULL,
                                           op_int_float2);

    // The adapter kernel should have been reused
    EXPECT_EQ(op_int_float1.specializations[0], op_int_float2.specializations[0]);

    kernel_instance<unary_operation_pair_t> op_uint_float1;
    cgcache.codegen_unary_function_adapter(ndt::make_type<unsigned int>(),
                                           ndt::make_type<float>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&uint_float_fn1),
                                           NULL,
                                           op_uint_float1);

    // int and uint look the same at the assembly level, so it should have reused the kernel
    EXPECT_EQ(op_int_float1.specializations[0], op_uint_float1.specializations[0]);
}
#endif // TODO reenable

