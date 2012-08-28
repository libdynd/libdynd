//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArray, Constructors) {
    ndarray a;

    // Default-constructed ndarray is NULL and will crash if access is attempted
    EXPECT_EQ(NULL, a.get_node().get());

    // Scalar ndarray
    a = ndarray(make_dtype<float>());
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
}

TEST(NDArray, ScalarConstructor) {
    ndarray a = 3;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());

    a = (int8_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<int8_t>(), a.get_dtype());

    a = (int16_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<int16_t>(), a.get_dtype());

    a = (int32_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<int32_t>(), a.get_dtype());

    a = (int64_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<int64_t>(), a.get_dtype());

    a = (uint8_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<uint8_t>(), a.get_dtype());

    a = (uint16_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<uint16_t>(), a.get_dtype());

    a = (uint32_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<uint32_t>(), a.get_dtype());

    a = (uint64_t)1;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<uint64_t>(), a.get_dtype());

    a = 3.14f;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<float>(), a.get_dtype());

    a = 3.14;
    EXPECT_EQ(0, a.get_ndim());
    EXPECT_EQ(make_dtype<double>(), a.get_dtype());
}

TEST(NDArray, ConstructorMemoryLayouts) {
    ndarray a, b;
    dtype dt(int16_type_id), dt2(int32_type_id);
    intptr_t shape[6];
    int axisperm[6];

    // The strides are set to zero for size-one dimensions
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    axisperm[0] = 0;
    axisperm[1] = 1;
    axisperm[2] = 2;
    a = ndarray(dt, 3, shape, axisperm);
    EXPECT_EQ(1, a.get_num_elements());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);
    EXPECT_EQ(0, a.get_strides()[2]);
    b = empty_like(a);
    EXPECT_EQ(1, b.get_num_elements());
    EXPECT_EQ(0, b.get_strides()[0]);
    EXPECT_EQ(0, b.get_strides()[1]);
    EXPECT_EQ(0, b.get_strides()[2]);

    // Test all permutations of memory layouts from 1 through 6 dimensions
    for (int ndim = 1; ndim <= 6; ++ndim) {
        // Go through all the permutations on ndim elements
        // to check every memory layout
        intptr_t num_elements = 1;
        for (int i = 0; i < ndim; ++i) {
            shape[i] = i + 2;
            axisperm[i] = i;
            num_elements *= shape[i];
        }
        do {
            // Test constructing the array using the perm
            a = ndarray(dt, ndim, shape, axisperm);
            EXPECT_EQ(num_elements, a.get_num_elements());
            intptr_t s = dt.element_size();
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(s, a.get_strides()[axisperm[i]]);
                s *= shape[axisperm[i]];
            }
            // Test constructing the array using empty_like, which preserves the memory layout
            b = empty_like(a);
            EXPECT_EQ(num_elements, b.get_num_elements());
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(a.get_strides()[i], b.get_strides()[i]);
            }
            // Test constructing the array using empty_like with a different dtype, which preserves the memory layout
            b = empty_like(a, dt2);
            EXPECT_EQ(num_elements, b.get_num_elements());
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(2 * a.get_strides()[i], b.get_strides()[i]);
            }
            //cout << "perm " << axisperm[0] << " " << axisperm[1] << " " << axisperm[2] << "\n";
            //cout << "strides " << a.get_strides(0) << " " << a.get_strides(1) << " " << a.get_strides(2) << "\n";
        } while(next_permutation(&axisperm[0], &axisperm[0] + ndim));
    }
}

TEST(NDArray, AsScalar) {
    ndarray a;

    a = ndarray(make_dtype<float>());
    EXPECT_EQ(1, a.get_num_elements());
    a.val_assign(3.14f);
    EXPECT_EQ(3.14f, a.as<float>());
    EXPECT_EQ(3.14f, a.as<double>());
    EXPECT_THROW(a.as<int64_t>(), runtime_error);
    EXPECT_EQ(3, a.as<int64_t>(assign_error_overflow));
    EXPECT_THROW(a.as<dnd_bool>(), runtime_error);
    EXPECT_THROW(a.as<dnd_bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<dnd_bool>(assign_error_none));
    EXPECT_THROW(a.as<bool>(), runtime_error);
    EXPECT_THROW(a.as<bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<bool>(assign_error_none));

    a = ndarray(make_dtype<double>());
    a.val_assign(3.141592653589);
    EXPECT_EQ(3.141592653589, a.as<double>());
    EXPECT_THROW(a.as<float>(assign_error_inexact), runtime_error);
    EXPECT_EQ(3.141592653589f, a.as<float>());
}

TEST(NDArray, CharArrayConstructor) {
    ndarray a;
    char values[8] = {1,2,3,4,5,6,7,8};

    // Constructor assignment
    a = values;
    EXPECT_EQ(1, a.get_ndim());
    EXPECT_EQ(8, a.get_shape()[0]);
    EXPECT_EQ(make_dtype<char>(), a.get_dtype());
    EXPECT_EQ(1, a(0).as<char>());
    EXPECT_EQ(2, a(1).as<char>());
    EXPECT_EQ(3, a(2).as<char>());
    EXPECT_EQ(4, a(3).as<char>());
    EXPECT_EQ(5, a(4).as<char>());
    EXPECT_EQ(6, a(5).as<char>());
    EXPECT_EQ(7, a(6).as<char>());
    EXPECT_EQ(8, a(7).as<char>());

    // Value assignment
    a.vals() = 0;
    EXPECT_EQ(0, a(0).as<char>());
    a.vals() = values;
    EXPECT_EQ(1, a(0).as<char>());
    EXPECT_EQ(2, a(1).as<char>());
    EXPECT_EQ(3, a(2).as<char>());
    EXPECT_EQ(4, a(3).as<char>());
    EXPECT_EQ(5, a(4).as<char>());
    EXPECT_EQ(6, a(5).as<char>());
    EXPECT_EQ(7, a(6).as<char>());
    EXPECT_EQ(8, a(7).as<char>());
}

#ifdef DND_INIT_LIST
TEST(NDArray, InitializerLists) {
    ndarray a = {1, 2, 3, 4, 5};
    EXPECT_EQ(5, a.get_num_elements());
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(1, a.get_ndim());
    EXPECT_EQ(5, a.get_shape()[0]);
    EXPECT_EQ((int)sizeof(int), a.get_strides()[0]);
    const int *ptr_i = (const int *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);

    ndarray b = {{1., 2., 3.}, {4., 5., 6.25}};
    EXPECT_EQ(6, b.get_num_elements());
    EXPECT_EQ(make_dtype<double>(), b.get_dtype());
    EXPECT_EQ(2, b.get_ndim());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(3*(int)sizeof(double), b.get_strides()[0]);
    EXPECT_EQ((int)sizeof(double), b.get_strides()[1]);
    const double *ptr_d = (const double *)b.get_readonly_originptr();
    EXPECT_EQ(1, ptr_d[0]);
    EXPECT_EQ(2, ptr_d[1]);
    EXPECT_EQ(3, ptr_d[2]);
    EXPECT_EQ(4, ptr_d[3]);
    EXPECT_EQ(5, ptr_d[4]);
    EXPECT_EQ(6.25, ptr_d[5]);

    // Testing assignment operator with initializer list (and 3D nested list)
    a = {{{1LL, 2LL}, {-1LL, -2LL}}, {{4LL, 5LL}, {6LL, 1LL}}};
    EXPECT_EQ(8, a.get_num_elements());
    EXPECT_EQ(make_dtype<long long>(), a.get_dtype());
    EXPECT_EQ(3, a.get_ndim());
    EXPECT_EQ(2, a.get_shape()[0]);
    EXPECT_EQ(2, a.get_shape()[1]);
    EXPECT_EQ(2, a.get_shape()[2]);
    EXPECT_EQ(4*(int)sizeof(long long), a.get_strides()[0]);
    EXPECT_EQ(2*(int)sizeof(long long), a.get_strides()[1]);
    EXPECT_EQ((int)sizeof(long long), a.get_strides()[2]);
    const long long *ptr_ll = (const long long *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_ll[0]);
    EXPECT_EQ(2, ptr_ll[1]);
    EXPECT_EQ(-1, ptr_ll[2]);
    EXPECT_EQ(-2, ptr_ll[3]);
    EXPECT_EQ(4, ptr_ll[4]);
    EXPECT_EQ(5, ptr_ll[5]);
    EXPECT_EQ(6, ptr_ll[6]);
    EXPECT_EQ(1, ptr_ll[7]);

    // If the shape is jagged, should throw an error
    EXPECT_THROW((a = {{1,2,3}, {1,2}}), runtime_error);
    EXPECT_THROW((a = {{{1},{2},{3}}, {{1},{2},{3, 4}}}), runtime_error);
}
#endif // DND_INIT_LIST

TEST(NDArray, InitFromNestedCArray) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    ndarray a = i0;
    EXPECT_EQ(6, a.get_num_elements());
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(2, a.get_ndim());
    EXPECT_EQ(2, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_shape()[1]);
    EXPECT_EQ(3*(int)sizeof(int), a.get_strides()[0]);
    EXPECT_EQ((int)sizeof(int), a.get_strides()[1]);
    const int *ptr_i = (const int *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);
    EXPECT_EQ(6, ptr_i[5]);

    float i1[2][2][3] = {{{1,2,3}, {1.5f, 2.5f, 3.5f}}, {{-10, 0, -3.1f}, {9,8,7}}};
    a = i1;
    EXPECT_EQ(12, a.get_num_elements());
    EXPECT_EQ(make_dtype<float>(), a.get_dtype());
    EXPECT_EQ(3, a.get_ndim());
    EXPECT_EQ(2, a.get_shape()[0]);
    EXPECT_EQ(2, a.get_shape()[1]);
    EXPECT_EQ(3, a.get_shape()[2]);
    EXPECT_EQ(6*(int)sizeof(float), a.get_strides()[0]);
    EXPECT_EQ(3*(int)sizeof(float), a.get_strides()[1]);
    EXPECT_EQ((int)sizeof(float), a.get_strides()[2]);
    const float *ptr_f = (float *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_f[0]);
    EXPECT_EQ(2, ptr_f[1]);
    EXPECT_EQ(3, ptr_f[2]);
    EXPECT_EQ(1.5, ptr_f[3]);
    EXPECT_EQ(2.5, ptr_f[4]);
    EXPECT_EQ(3.5, ptr_f[5]);
    EXPECT_EQ(-10, ptr_f[6]);
    EXPECT_EQ(0, ptr_f[7]);
    EXPECT_EQ(-3.1f, ptr_f[8]);
    EXPECT_EQ(9, ptr_f[9]);
    EXPECT_EQ(8, ptr_f[10]);
    EXPECT_EQ(7, ptr_f[11]);
}
