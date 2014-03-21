//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/platform_definitions.hpp>

#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/codegen/codegen_cache.hpp>

#if 0 // TODO reenable

using namespace std;
using namespace dynd;

template<class S, class T1, class T2>
static S multiply_values(T1 value1, T2 value2) {
    return (S)(value1 * value2);
}

template int multiply_values<int, float, double>(float, double);
template float multiply_values<float, float, float>(float, float);
template float multiply_values<float, double, int>(double, int);

TEST(BinaryKernelAdapter, BasicOperations) {
    codegen_cache cgcache;
    kernel_instance<binary_operation_t> op_int_float_double, op_float_float_float, op_float_double_int;
    // NOTE: Cannot cast directly to <void*>, because of a compile error on MSVC:
    //         "Context does not allow for disambiguation of overloaded function"
    cgcache.codegen_binary_function_adapter(ndt::make_type<int>(),
                                            ndt::make_type<float>(),
                                            ndt::make_type<double>(),
                                            cdecl_callconv,
                                            (void*)static_cast<int (*)(float, double)>(&multiply_values<int, float, double>),
                                            NULL,
                                            op_int_float_double);
    cgcache.codegen_binary_function_adapter(ndt::make_type<float>(),
                                            ndt::make_type<float>(),
                                            ndt::make_type<float>(),
                                            cdecl_callconv,
                                            (void*)static_cast<float (*)(float, float)>(&multiply_values<float, float, float>),
                                            NULL,
                                            op_float_float_float);
    cgcache.codegen_binary_function_adapter(ndt::make_type<float>(),
                                            ndt::make_type<double>(),
                                            ndt::make_type<int>(),
                                            cdecl_callconv,
                                            (void*)static_cast<float (*)(double, int)>(&multiply_values<float, double, int>),
                                            NULL,
                                            op_float_double_int);

    int int_vals[3];
    float float_vals[3];
    double double_vals[3];

    float_vals[0] = 1.f;
    float_vals[1] = 2.5f;
    float_vals[2] = 3.25f;
    double_vals[0] = 3.0;
    double_vals[1] = -2.0;
    double_vals[2] = 4.0;
    op_int_float_double.kernel((char *)int_vals, sizeof(int),
                    (const char *)float_vals, sizeof(float),
                    (const char *)double_vals, sizeof(double),
                    3, op_int_float_double.auxdata);
    EXPECT_EQ(3, int_vals[0]);
    EXPECT_EQ(-5, int_vals[1]);
    EXPECT_EQ(13, int_vals[2]);

    op_float_float_float.kernel((char *)float_vals, sizeof(float),
                    (const char *)float_vals, sizeof(float),
                    (const char *)float_vals, sizeof(float),
                    3, op_float_float_float.auxdata);
    EXPECT_EQ(1.f, float_vals[0]);
    EXPECT_EQ(6.25f, float_vals[1]);
    EXPECT_EQ(10.5625f, float_vals[2]);

    double_vals[0] = -1.f;
    double_vals[1] = 3.5f;
    double_vals[2] = -2.25f;
    op_float_double_int.kernel((char *)float_vals, sizeof(float),
                    (const char *)double_vals, sizeof(double),
                    (const char *)int_vals, sizeof(int),
                    3, op_float_double_int.auxdata);
    EXPECT_EQ(-3.f, float_vals[0]);
    EXPECT_EQ(-17.5f, float_vals[1]);
    EXPECT_EQ(-29.25f, float_vals[2]);

}


#if defined(DYND_CALL_MSFT_X64)

class raise_if_greater_exception : public std::runtime_error {
public:
    raise_if_greater_exception()
        : std::runtime_error("raise_if_greater_exception")
    {
    }
};

static int raise_if_greater(int value1, int value2) {
    if (value1 <= value2) {
        return value1 - value2;
    } else {
        throw raise_if_greater_exception();
    }
}

TEST(BinaryKernelAdapter, UnwindException) {
    codegen_cache cgcache;
    kernel_instance<binary_operation_t> rig;
    cgcache.codegen_binary_function_adapter(ndt::make_type<int>(),
                                            ndt::make_type<int>(),
                                            ndt::make_type<int>(),
                                            cdecl_callconv,
                                            reinterpret_cast<void*>(&raise_if_greater),
                                            NULL,
                                            rig);
    int in1[3], in2[3], out[3];
    // First call it with no exception raised
    in1[0] = 0;
    in1[1] = 10;
    in1[2] = 10000;
    in2[0] = 0;
    in2[1] = 11;
    in2[2] = 100000;
    rig.kernel((char *)out, sizeof(int), (const char *)in1, sizeof(int), (const char *)in2, sizeof(int),
                    3, rig.auxdata);
    EXPECT_EQ(0, out[0]);
    EXPECT_EQ(-1, out[1]);
    EXPECT_EQ(-90000, out[2]);

    // Call it with an input stride that skips the negative value
    in2[1] = 9;
    rig.kernel((char *)out, sizeof(int), (const char *)in1, 2*sizeof(int), (const char *)in2, 2*sizeof(int),
                    2, rig.auxdata);
    EXPECT_EQ(0, out[0]);
    EXPECT_EQ(-90000, out[1]);
    EXPECT_EQ(-90000, out[2]);

    // Call it with a negative value
    EXPECT_THROW(rig.kernel((char *)out, sizeof(int), (const char *)in1, sizeof(int), (const char *)in2, sizeof(int),
                    3, rig.auxdata),
            raise_if_greater_exception);
}

#endif

#endif // TODO reenable

