//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable

#include <dynd/platform_definitions.hpp>
#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/codegen/codegen_cache.hpp>

using namespace std;
using namespace dynd;

template<class S, class T>
static S double_value(T value) {
    return (S)(2 * value);
}

template int double_value<int, float>(float);
template float double_value<float, float>(float);
template float double_value<float, double>(double);

TEST(UnaryKernelAdapter, BasicOperations) {
    codegen_cache cgcache;
    kernel_instance<unary_operation_pair_t> op_int_float, op_float_float, op_float_double;
    // NOTE: Cannot cast directly to <void*>, because of a compile error on MSVC:
    //         "Context does not allow for disambiguation of overloaded function"
    cgcache.codegen_unary_function_adapter(ndt::make_type<int>(), ndt::make_type<float>(), cdecl_callconv,
                    (void*)static_cast<int (*)(float)>(&double_value<int, float>), NULL, op_int_float);
    cgcache.codegen_unary_function_adapter(ndt::make_type<float>(), ndt::make_type<float>(), cdecl_callconv,
                    (void*)static_cast<float (*)(float)>(&double_value<float, float>), NULL, op_float_float);
    cgcache.codegen_unary_function_adapter(ndt::make_type<float>(), ndt::make_type<double>(), cdecl_callconv,
                    (void*)static_cast<float (*)(double)>(&double_value<float, double>), NULL, op_float_double);

    int int_vals[3];
    float float_vals[3];
    double double_vals[3];

    float_vals[0] = 1.f;
    float_vals[1] = 2.5f;
    float_vals[2] = 3.25f;
    op_int_float.specializations[0]((char *)int_vals, sizeof(int), (char *)float_vals, sizeof(float), 3,
                    op_int_float.auxdata);
    EXPECT_EQ(2, int_vals[0]);
    EXPECT_EQ(5, int_vals[1]);
    EXPECT_EQ(6, int_vals[2]);

    op_float_float.specializations[0]((char *)float_vals, sizeof(float), (char *)float_vals, sizeof(float), 3,
                    op_float_float.auxdata);

    EXPECT_EQ(2.f, float_vals[0]);
    EXPECT_EQ(5.f, float_vals[1]);
    EXPECT_EQ(6.5f, float_vals[2]);

    double_vals[0] = -1.f;
    double_vals[1] = 3.5f;
    double_vals[2] = -2.25f;
    op_float_double.specializations[0]((char *)float_vals, sizeof(float), (char *)double_vals, sizeof(double), 3,
                    op_float_double.auxdata);
    EXPECT_EQ(-2.f, float_vals[0]);
    EXPECT_EQ(7.f, float_vals[1]);
    EXPECT_EQ(-4.5f, float_vals[2]);

}

#if defined(DYND_CALL_MSFT_X64)

class raise_if_negative_exception : public std::runtime_error {
public:
    raise_if_negative_exception()
        : std::runtime_error("raise_if_negative_exception")
    {
    }
};

static int raise_if_negative(int value) {
    if (value >= 0) {
        return value;
    } else {
        throw raise_if_negative_exception();
    }
}

TEST(UnaryKernelAdapter, UnwindException) {
    codegen_cache cgcache;
    kernel_instance<unary_operation_pair_t> rin;
    cgcache.codegen_unary_function_adapter(ndt::make_type<int>(),
                                           ndt::make_type<int>(),
                                           cdecl_callconv,
                                           reinterpret_cast<void*>(&raise_if_negative),
                                           NULL,
                                           rin);
    int in[3], out[3];
    // First call it with no exception raised
    in[0] = 0;
    in[1] = 10;
    in[2] = 10000;
    rin.specializations[0]((char *)out, sizeof(int), (const char *)in, sizeof(int), 3, rin.auxdata);
    EXPECT_EQ(0, out[0]);
    EXPECT_EQ(10, out[1]);
    EXPECT_EQ(10000, out[2]);

    // Call it with an input stride that skips the negative value
    in[1] = -1;
    rin.specializations[0]((char *)out, sizeof(int), (const char *)in, 2*sizeof(int), 2, rin.auxdata);
    EXPECT_EQ(0, out[0]);
    EXPECT_EQ(10000, out[1]);
    EXPECT_EQ(10000, out[2]);

    // Call it with a negative value
    EXPECT_THROW(rin.specializations[0]((char *)out, sizeof(int), (const char *)in, sizeof(int), 3, rin.auxdata),
            raise_if_negative_exception);
}
#endif

#endif // TODO reenable

