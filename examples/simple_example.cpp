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
    // Buffering the first operand
    a = ndarray(2) * ndarray(3.f);
    EXPECT_EQ(elementwise_node_category, a.get_expr_tree()->get_category());
    EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_dtype());
    EXPECT_EQ((make_convert_dtype<float, int>()), a.get_expr_tree()->get_opnode(0)->get_dtype());
    EXPECT_EQ(make_dtype<float>(), a.get_expr_tree()->get_opnode(1)->get_dtype());
    a.debug_dump(cout);
    ndarray b = a.vals();
    b.debug_dump(cout);
    cout << b << endl;
    EXPECT_EQ(6, a.as<float>());


    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
