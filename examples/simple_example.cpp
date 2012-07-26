//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/ndarray_arange.hpp>
#include <dnd/codegen/codegen_cache.hpp>
#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>

using namespace std;
using namespace dnd;

typedef complex<double> A0;
typedef float R;

#define EXPECT_EQ(a, b) \
    cout << "first   : " << (a) << endl \
         << "second  : " << (b) << endl

template<class S, class T>
S double_value(T value) {
    return (S)(2 * value);
}

int main()
{
    try {
    float v0[5] = {3.5f, 1.3f, -2.4999f, -2.999, 1000.50001f};
    ndarray a = v0, b;

    b = a.as_dtype<int>(assign_error_overflow);
    b = b.as_dtype<float>(assign_error_inexact);
    // Multiple as_dtype operations should make a chained conversion dtype
    EXPECT_EQ(make_convert_dtype(make_dtype<float>(),
                                    make_convert_dtype<int, float>(assign_error_overflow), assign_error_inexact),
              b.get_dtype());

    // Evaluating the values should truncate them to integers
    b = b.vals();
    // Now it's just the value dtype, no chaining
    EXPECT_EQ(make_dtype<float>(), b.get_dtype());
    EXPECT_EQ(3, b(0).as<float>());
    EXPECT_EQ(1, b(1).as<float>());
    EXPECT_EQ(-2, b(2).as<float>());
    EXPECT_EQ(-2, b(3).as<float>());
    EXPECT_EQ(1000, b(4).as<float>());

    // Now try it with longer chaining through multiple element sizes
    b = a.as_dtype<int16_t>(assign_error_overflow);
    b = b.as_dtype<int32_t>(assign_error_overflow);
    b = b.as_dtype<int16_t>(assign_error_overflow);
    b = b.as_dtype<int64_t>(assign_error_overflow);
    b = b.as_dtype<float>(assign_error_overflow);
    b = b.as_dtype<int32_t>(assign_error_overflow);

    EXPECT_EQ(make_convert_dtype(make_dtype<int32_t>(),
                    make_convert_dtype(make_dtype<float>(),
                        make_convert_dtype(make_dtype<int64_t>(),
                            make_convert_dtype(make_dtype<int16_t>(),
                                make_convert_dtype(make_dtype<int32_t>(),
                                    make_convert_dtype<int16_t, float>(
                                    assign_error_overflow),
                                assign_error_overflow),
                            assign_error_overflow),
                        assign_error_overflow),
                    assign_error_overflow),
                assign_error_overflow),
            b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_dtype<int32_t>(), b.get_dtype());
    EXPECT_EQ(3, b(0).as<int32_t>());
    EXPECT_EQ(1, b(1).as<int32_t>());
    EXPECT_EQ(-2, b(2).as<int32_t>());
    EXPECT_EQ(-2, b(3).as<int32_t>());
    EXPECT_EQ(1000, b(4).as<int32_t>());

    } catch(int) { //std::exception& e) {
        //cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
