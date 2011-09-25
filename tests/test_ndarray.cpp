#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArray, Constructors) {
    ndarray a;

    // Default-constructed ndarray has size zero
    EXPECT_EQ(0, a.num_elements());
    EXPECT_EQ(1, a.ndim());
    EXPECT_EQ(0, a.shape()[0]);
    EXPECT_EQ(0, a.strides()[0]);

    // Scalar ndarray
    a = ndarray(make_dtype<float>());
    EXPECT_EQ(1, a.num_elements());
    EXPECT_EQ(0, a.ndim());

    // One-dimensional ndarray with one element
    a = ndarray(1, make_dtype<float>());
    EXPECT_EQ(1, a.num_elements());
    EXPECT_EQ(1, a.ndim());
    EXPECT_EQ(1, a.shape()[0]);
    EXPECT_EQ(0, a.strides()[0]);

    // One-dimensional ndarray
    a = ndarray(3, make_dtype<float>());
    EXPECT_EQ(3, a.num_elements());
    EXPECT_EQ(1, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(sizeof(float), a.strides()[0]);

    // Two-dimensional ndarray with a size-one dimension
    a = ndarray(3, 1, make_dtype<float>());
    EXPECT_EQ(3, a.num_elements());
    EXPECT_EQ(2, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(1, a.shape()[1]);
    EXPECT_EQ(sizeof(float), a.strides()[0]);
    EXPECT_EQ(0, a.strides()[1]);

    // Two-dimensional ndarray with a size-one dimension
    a = ndarray(1, 3, make_dtype<float>());
    EXPECT_EQ(3, a.num_elements());
    EXPECT_EQ(2, a.ndim());
    EXPECT_EQ(1, a.shape()[0]);
    EXPECT_EQ(3, a.shape()[1]);
    EXPECT_EQ(0, a.strides()[0]);
    EXPECT_EQ(sizeof(float), a.strides()[1]);

    // Two-dimensional ndarray
    a = ndarray(3, 5, make_dtype<float>());
    EXPECT_EQ(15, a.num_elements());
    EXPECT_EQ(2, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(5, a.shape()[1]);
    EXPECT_EQ(5*sizeof(float), a.strides()[0]);
    EXPECT_EQ(sizeof(float), a.strides()[1]);

    // Three-dimensional ndarray with size-one dimension
    a = ndarray(1, 5, 4, make_dtype<float>());
    EXPECT_EQ(20, a.num_elements());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(1, a.shape()[0]);
    EXPECT_EQ(5, a.shape()[1]);
    EXPECT_EQ(4, a.shape()[2]);
    EXPECT_EQ(0, a.strides()[0]);
    EXPECT_EQ(4*sizeof(float), a.strides()[1]);
    EXPECT_EQ(sizeof(float), a.strides()[2]);

    // Three-dimensional ndarray with size-one dimension
    a = ndarray(3, 1, 4, make_dtype<float>());
    EXPECT_EQ(12, a.num_elements());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(1, a.shape()[1]);
    EXPECT_EQ(4, a.shape()[2]);
    EXPECT_EQ(4*sizeof(float), a.strides()[0]);
    EXPECT_EQ(0, a.strides()[1]);
    EXPECT_EQ(sizeof(float), a.strides()[2]);

    // Three-dimensional ndarray with size-one dimension
    a = ndarray(3, 5, 1, make_dtype<float>());
    EXPECT_EQ(15, a.num_elements());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(5, a.shape()[1]);
    EXPECT_EQ(1, a.shape()[2]);
    EXPECT_EQ(5*sizeof(float), a.strides()[0]);
    EXPECT_EQ(sizeof(float), a.strides()[1]);
    EXPECT_EQ(0, a.strides()[2]);

    // Three-dimensional ndarray
    a = ndarray(3, 5, 4, make_dtype<float>());
    EXPECT_EQ(60, a.num_elements());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(3, a.shape()[0]);
    EXPECT_EQ(5, a.shape()[1]);
    EXPECT_EQ(4, a.shape()[2]);
    EXPECT_EQ(5*4*sizeof(float), a.strides()[0]);
    EXPECT_EQ(4*sizeof(float), a.strides()[1]);
    EXPECT_EQ(sizeof(float), a.strides()[2]);
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
    EXPECT_EQ(1, a.num_elements());
    EXPECT_EQ(0, a.strides(0));
    EXPECT_EQ(0, a.strides(1));
    EXPECT_EQ(0, a.strides(2));
    b = empty_like(a);
    EXPECT_EQ(1, b.num_elements());
    EXPECT_EQ(0, b.strides(0));
    EXPECT_EQ(0, b.strides(1));
    EXPECT_EQ(0, b.strides(2));

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
            EXPECT_EQ(num_elements, a.num_elements());
            intptr_t s = dt.itemsize();
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(s, a.strides(axisperm[i]));
                s *= shape[axisperm[i]];
            }
            // Test constructing the array using empty_like, which preserves the memory layout
            b = empty_like(a);
            EXPECT_EQ(num_elements, b.num_elements());
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(a.strides(i), b.strides(i));
            }
            // Test constructing the array using empty_like with a different dtype, which preserves the memory layout
            b = empty_like(a, dt2);
            EXPECT_EQ(num_elements, b.num_elements());
            for (int i = 0; i < ndim; ++i) {
                EXPECT_EQ(2 * a.strides(i), b.strides(i));
            }
            //cout << "perm " << axisperm[0] << " " << axisperm[1] << " " << axisperm[2] << "\n";
            //cout << "strides " << a.strides(0) << " " << a.strides(1) << " " << a.strides(2) << "\n";
        } while(next_permutation(&axisperm[0], &axisperm[0] + ndim));
    }
}

TEST(NDArray, AsScalar) {
    ndarray a;

    a = ndarray(make_dtype<float>());
    EXPECT_EQ(1, a.num_elements());
    a.vassign(3.14f);
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
    a.vassign(3.141592653589);
    EXPECT_EQ(3.141592653589, a.as<double>());
    EXPECT_THROW(a.as<float>(assign_error_inexact), runtime_error);
    EXPECT_EQ(3.141592653589f, a.as<float>());
}

TEST(NDArray, InitializerLists) {
    ndarray a = {1, 2, 3, 4, 5};
    EXPECT_EQ(5, a.num_elements());
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(1, a.ndim());
    EXPECT_EQ(5, a.shape()[0]);
    EXPECT_EQ(sizeof(int), a.strides()[0]);
    int *ptr_i = (int *)a.originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);

    ndarray b = {{1., 2., 3.}, {4., 5., 6.25}};
    EXPECT_EQ(6, b.num_elements());
    EXPECT_EQ(make_dtype<double>(), b.get_dtype());
    EXPECT_EQ(2, b.ndim());
    EXPECT_EQ(2, b.shape()[0]);
    EXPECT_EQ(3, b.shape()[1]);
    EXPECT_EQ(3*sizeof(double), b.strides()[0]);
    EXPECT_EQ(sizeof(double), b.strides()[1]);
    double *ptr_d = (double *)b.originptr();
    EXPECT_EQ(1, ptr_d[0]);
    EXPECT_EQ(2, ptr_d[1]);
    EXPECT_EQ(3, ptr_d[2]);
    EXPECT_EQ(4, ptr_d[3]);
    EXPECT_EQ(5, ptr_d[4]);
    EXPECT_EQ(6.25, ptr_d[5]);

    // Testing assignment operator with initializer list (and 3D nested list)
    a = {{{1LL, 2LL}, {-1LL, -2LL}}, {{4LL, 5LL}, {6LL, 1LL}}};
    EXPECT_EQ(8, a.num_elements());
    EXPECT_EQ(make_dtype<long long>(), a.get_dtype());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(2, a.shape()[0]);
    EXPECT_EQ(2, a.shape()[1]);
    EXPECT_EQ(2, a.shape()[2]);
    EXPECT_EQ(4*sizeof(long long), a.strides()[0]);
    EXPECT_EQ(2*sizeof(long long), a.strides()[1]);
    EXPECT_EQ(sizeof(long long), a.strides()[2]);
    long long *ptr_ll = (long long *)a.originptr();
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

TEST(NDArray, InitFromNestedCArray) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    ndarray a = i0;
    EXPECT_EQ(6, a.num_elements());
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(2, a.ndim());
    EXPECT_EQ(2, a.shape()[0]);
    EXPECT_EQ(3, a.shape()[1]);
    EXPECT_EQ(3*sizeof(int), a.strides()[0]);
    EXPECT_EQ(sizeof(int), a.strides()[1]);
    int *ptr_i = (int *)a.originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);
    EXPECT_EQ(6, ptr_i[5]);

    float i1[2][2][3] = {{{1,2,3}, {1.5f, 2.5f, 3.5f}}, {{-10, 0, -3.1f}, {9,8,7}}};
    a = i1;
    EXPECT_EQ(12, a.num_elements());
    EXPECT_EQ(make_dtype<float>(), a.get_dtype());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(2, a.shape()[0]);
    EXPECT_EQ(2, a.shape()[1]);
    EXPECT_EQ(3, a.shape()[2]);
    EXPECT_EQ(6*sizeof(float), a.strides()[0]);
    EXPECT_EQ(3*sizeof(float), a.strides()[1]);
    EXPECT_EQ(sizeof(float), a.strides()[2]);
    float *ptr_f = (float *)a.originptr();
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
