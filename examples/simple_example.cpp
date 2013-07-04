//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/ndobject_range.hpp>
#include <dynd/codegen/codegen_cache.hpp>
#include <dynd/codegen/unary_kernel_adapter_codegen.hpp>

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
        intptr_t shape[] = {2,3,4};
        int axisperm[] = {0,2,1};

        nd::array a = nd::make_strided_array(make_dtype<int>(), 3, shape,
                        nd::read_access_flag|nd::write_access_flag, axisperm);

        a.debug_print(cout);

        nd::array b = empty_like(a);

        b.debug_print(cout);

        nd::array c = empty_like(a, make_dtype<double>());

        c.debug_print(cout);

    } catch(int) { //std::exception& e) {
        //cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
