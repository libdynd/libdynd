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
    ndarray a;

    a = ndarray(make_fixedstring_dtype(string_encoding_utf_16, 16));
    // Fill up the string with values
    a.vals() = std::string("0123456789012345");
    EXPECT_EQ("0123456789012345", a.as<std::string>());
    // Confirm that now assigning a smaller string works
    a.vals() = std::string("abc");
    EXPECT_EQ("abc", a.as<std::string>());


    } catch(int) { //std::exception& e) {
        //cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
