//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/ndarray_arange.hpp>

using namespace std;
using namespace dnd;

int main1()
{
    try {
        ndarray a;

        //a = {1, 5, 7};
        int avals[] = {1, 5, 7};
        a = avals;

        cout << a << endl;

        ndarray a2 = a.as_dtype<float>();

        cout << a2 << endl;

        ndarray a3 = a.as_dtype<double>();

        cout << a3 << endl;

        ndarray a4 = a2.as_dtype<double>();

        cout << a4 << endl;
        return 0;

        float avals2[2][3] = {{1,2,3}, {3,2,9}};
        ndarray b = avals2;

        ndarray c = a + b;

        c.debug_dump(cout);
        cout << c << endl;

        cout << c(0,1) << endl;
        a(1).val_assign(1.5f);
        cout << c(0,1) << endl;

        return 0;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}

#define EXPECT_EQ(a, b) \
    cout << "first   : " << (a) << endl \
         << "second  : " << (b) << endl

int main()
{
    try {
        ndarray a;

        // Default-constructed ndarray is NULL and will crash if access is attempted
        EXPECT_EQ(NULL, a.get_expr_tree().get());

        // Scalar ndarray
        a = ndarray(make_dtype<float>());
        cout << (void *)a.get_expr_tree().get() << endl;
        a.debug_dump(cout);
        EXPECT_EQ(1, a.get_num_elements());
        EXPECT_EQ(0, a.get_ndim());

        // One-dimensional ndarray with one element
        a = ndarray(1, make_dtype<float>());
        EXPECT_EQ(1, a.get_num_elements());
        EXPECT_EQ(1, a.get_ndim());
        EXPECT_EQ(1, a.get_shape()[0]);
        EXPECT_EQ(0, a.get_strides()[0]);

        // One-dimensional ndarray
        a = ndarray(3, make_dtype<float>());
        EXPECT_EQ(3, a.get_num_elements());
        EXPECT_EQ(1, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[0]);

        // Two-dimensional ndarray with a size-one dimension
        a = ndarray(3, 1, make_dtype<float>());
        EXPECT_EQ(3, a.get_num_elements());
        EXPECT_EQ(2, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ(1, a.get_shape()[1]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[0]);
        EXPECT_EQ(0, a.get_strides()[1]);

        // Two-dimensional ndarray with a size-one dimension
        a = ndarray(1, 3, make_dtype<float>());
        EXPECT_EQ(3, a.get_num_elements());
        EXPECT_EQ(2, a.get_ndim());
        EXPECT_EQ(1, a.get_shape()[0]);
        EXPECT_EQ(3, a.get_shape()[1]);
        EXPECT_EQ(0, a.get_strides()[0]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[1]);

        // Two-dimensional ndarray
        a = ndarray(3, 5, make_dtype<float>());
        EXPECT_EQ(15, a.get_num_elements());
        EXPECT_EQ(2, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ(5, a.get_shape()[1]);
        EXPECT_EQ(5*(int)sizeof(float), a.get_strides()[0]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[1]);

        // Three-dimensional ndarray with size-one dimension
        a = ndarray(1, 5, 4, make_dtype<float>());
        EXPECT_EQ(20, a.get_num_elements());
        EXPECT_EQ(3, a.get_ndim());
        EXPECT_EQ(1, a.get_shape()[0]);
        EXPECT_EQ(5, a.get_shape()[1]);
        EXPECT_EQ(4, a.get_shape()[2]);
        EXPECT_EQ(0, a.get_strides()[0]);
        EXPECT_EQ(4*(int)sizeof(float), a.get_strides()[1]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[2]);

        // Three-dimensional ndarray with size-one dimension
        a = ndarray(3, 1, 4, make_dtype<float>());
        EXPECT_EQ(12, a.get_num_elements());
        EXPECT_EQ(3, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ(1, a.get_shape()[1]);
        EXPECT_EQ(4, a.get_shape()[2]);
        EXPECT_EQ(4*(int)sizeof(float), a.get_strides()[0]);
        EXPECT_EQ(0, a.get_strides()[1]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[2]);

        // Three-dimensional ndarray with size-one dimension
        a = ndarray(3, 5, 1, make_dtype<float>());
        EXPECT_EQ(15, a.get_num_elements());
        EXPECT_EQ(3, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ(5, a.get_shape()[1]);
        EXPECT_EQ(1, a.get_shape()[2]);
        EXPECT_EQ(5*(int)sizeof(float), a.get_strides()[0]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[1]);
        EXPECT_EQ(0, a.get_strides()[2]);

        // Three-dimensional ndarray
        a = ndarray(3, 5, 4, make_dtype<float>());
        EXPECT_EQ(60, a.get_num_elements());
        EXPECT_EQ(3, a.get_ndim());
        EXPECT_EQ(3, a.get_shape()[0]);
        EXPECT_EQ(5, a.get_shape()[1]);
        EXPECT_EQ(4, a.get_shape()[2]);
        EXPECT_EQ(5*4*(int)sizeof(float), a.get_strides()[0]);
        EXPECT_EQ(4*(int)sizeof(float), a.get_strides()[1]);
        EXPECT_EQ((int)sizeof(float), a.get_strides()[2]);
    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
