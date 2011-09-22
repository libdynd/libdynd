#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(NDArray, AsScalar) {
    ndarray a;

    a = ndarray(make_dtype<float>());
    a.vassign(3.14f);
    EXPECT_EQ(3.14f, a.as_scalar<float>());
    EXPECT_EQ(3.14f, a.as_scalar<double>());
    EXPECT_THROW(a.as_scalar<int64_t>(), runtime_error);
    EXPECT_EQ(3, a.as_scalar<int64_t>(assign_error_overflow));
    EXPECT_THROW(a.as_scalar<dnd_bool>(), runtime_error);
    EXPECT_THROW(a.as_scalar<dnd_bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as_scalar<dnd_bool>(assign_error_none));

    a = ndarray(make_dtype<double>());
    a.vassign(3.141592653589);
    EXPECT_EQ(3.141592653589, a.as_scalar<double>());
    EXPECT_THROW(a.as_scalar<float>(assign_error_inexact), runtime_error);
    EXPECT_EQ(3.141592653589f, a.as_scalar<float>());
}

TEST(NDArray, InitializerLists) {
    ndarray a = {1, 2, 3, 4, 5};
    EXPECT_EQ(make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(1, a.ndim());
    EXPECT_EQ(5, a.shape()[0]);
    EXPECT_EQ(sizeof(int), a.strides()[0]);
    int *ptr_i = (int *)a.data();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);

    ndarray b = {{1., 2., 3.}, {4., 5., 6.25}};
    EXPECT_EQ(make_dtype<double>(), b.get_dtype());
    EXPECT_EQ(2, b.ndim());
    EXPECT_EQ(2, b.shape()[0]);
    EXPECT_EQ(3, b.shape()[1]);
    EXPECT_EQ(3*sizeof(double), b.strides()[0]);
    EXPECT_EQ(sizeof(double), b.strides()[1]);
    double *ptr_d = (double *)b.data();
    EXPECT_EQ(1, ptr_d[0]);
    EXPECT_EQ(2, ptr_d[1]);
    EXPECT_EQ(3, ptr_d[2]);
    EXPECT_EQ(4, ptr_d[3]);
    EXPECT_EQ(5, ptr_d[4]);
    EXPECT_EQ(6.25, ptr_d[5]);

    // Testing assignment operator with initializer list (and 3D nested list)
    a = {{{1LL, 2LL}, {-1LL, -2LL}}, {{4LL, 5LL}, {6LL, 1LL}}};
    EXPECT_EQ(make_dtype<long long>(), a.get_dtype());
    EXPECT_EQ(3, a.ndim());
    EXPECT_EQ(2, a.shape()[0]);
    EXPECT_EQ(2, a.shape()[1]);
    EXPECT_EQ(2, a.shape()[2]);
    EXPECT_EQ(4*sizeof(long long), a.strides()[0]);
    EXPECT_EQ(2*sizeof(long long), a.strides()[1]);
    EXPECT_EQ(sizeof(long long), a.strides()[2]);
    long long *ptr_ll = (long long *)a.data();
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
