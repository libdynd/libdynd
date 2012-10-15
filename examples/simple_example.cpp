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
using namespace dynd;

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

    float v0[4] = {3.5, 1.0, 0, 1000};
    ndarray a = v0, b;

	a.debug_dump(cout);
    cout << a << endl;

    //b = a.as_dtype(make_dtype<int>());
    // This triggers the conversion from float to int,
    // but the default assign policy is 'fractional'

    // Allow truncation of fractional part
    //b = a.as_dtype(make_dtype<int>(), assign_error_overflow);
    b = a.as_dtype(make_dtype<int>(), assign_error_overflow);
	b.debug_dump(cout);
    cout << b << endl;
    b = b.vals();
	b.debug_dump(cout);
    cout << b << endl;
    EXPECT_EQ(3, b(0).as<int>());
    EXPECT_EQ(1, b(1).as<int>());
    EXPECT_EQ(0, b(2).as<int>());
    EXPECT_EQ(1000, b(3).as<int>());

    } catch(int) { //std::exception& e) {
        //cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
