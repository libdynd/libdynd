//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/dtypes/strided_dim_type.hpp>
#include <dynd/dtypes/fixedbytes_type.hpp>
#include <dynd/dtypes/string_type.hpp>

using namespace std;
using namespace dynd;

TEST(Array, NullConstructor) {
    nd::array a;

    // Default-constructed nd::array is NULL and will crash if access is attempted
    EXPECT_EQ(NULL, a.get_memblock().get());
}


TEST(Array, ScalarConstructor) {
    // Scalar nd::array
    nd::array a = nd::empty(ndt::make_dtype<float>());
    EXPECT_EQ(ndt::make_dtype<float>(), a.get_dtype());
    EXPECT_TRUE(a.is_scalar());
    // Constructing an empty array with too many dimensions should raise an error
    EXPECT_THROW(nd::empty(1, ndt::make_dtype<double>()), runtime_error);
}

TEST(Array, OneDimConstructor) {
    // One-dimensional strided nd::array with one element
    nd::array a = nd::empty(1, make_strided_dim_type(ndt::make_dtype<float>()));
    EXPECT_EQ(make_strided_dim_type(ndt::make_dtype<float>()), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // One-dimensional nd::array
    a = nd::empty(3, make_strided_dim_type(ndt::make_dtype<float>()));
    EXPECT_EQ(make_strided_dim_type(ndt::make_dtype<float>()), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[0]);
}

TEST(Array, TwoDimConstructor) {
    // Two-dimensional nd::array with a size-one dimension
    nd::array a = nd::empty(3, 1, make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(1, a.get_shape()[1]);
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);

    // Two-dimensional nd::array with a size-one dimension
    a = nd::empty(1, 3, make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_shape()[1]);
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);

    // Two-dimensional nd::array
    a = nd::empty(3, 5, make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>())), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(5 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);
}

TEST(Array, ThreeDimConstructor) {
    // Three-dimensional nd::array with size-one dimension
    nd::array a = nd::empty(1, 5, 4, make_strided_dim_type(
                    make_strided_dim_type(
                        make_strided_dim_type(ndt::make_dtype<float>()))));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>()))), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);

    // Three-dimensional nd::array with size-one dimension
    a = nd::empty(3, 1, 4, make_strided_dim_type(
                    make_strided_dim_type(
                        make_strided_dim_type(ndt::make_dtype<float>()))));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>()))), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(1, a.get_shape()[1]);
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);

    // Three-dimensional nd::array with size-one dimension
    a = nd::empty(3, 5, 1, make_strided_dim_type(
                    make_strided_dim_type(
                        make_strided_dim_type(ndt::make_dtype<float>()))));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>()))), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(1, a.get_shape()[2]);
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(5 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(0, a.get_strides()[2]);

    // Three-dimensional nd::array
    a = nd::empty(3, 5, 4, make_strided_dim_type(
                    make_strided_dim_type(
                        make_strided_dim_type(ndt::make_dtype<float>()))));
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>()))), a.get_dtype());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(5 * 4 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);
}

TEST(Array, IntScalarConstructor) {
    stringstream ss;

    nd::array a = 3;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<int>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3, int32)", ss.str());

    a = (int8_t)1;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<int8_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(1, int8)", ss.str());

    a = (int16_t)2;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<int16_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(2, int16)", ss.str());

    a = (int32_t)3;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<int32_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3, int32)", ss.str());

    a = (int64_t)4;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<int64_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(4, int64)", ss.str());
}

TEST(Array, UIntScalarConstructor) {
    stringstream ss;

    nd::array a = (uint8_t)5;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<uint8_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(5, uint8)", ss.str());

    a = (uint16_t)6;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<uint16_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(6, uint16)", ss.str());

    a = (uint32_t)7;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<uint32_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(7, uint32)", ss.str());

    a = (uint64_t)8;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<uint64_t>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(8, uint64)", ss.str());
}

TEST(Array, FloatScalarConstructor) {
    stringstream ss;

    nd::array a = 3.25f;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<float>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3.25, float32)", ss.str());

    a = 3.5;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<double>(), a.get_dtype());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3.5, float64)", ss.str());

    a = complex<float>(3.14f, 1.0f);
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<complex<float> >(), a.get_dtype());

    a = complex<double>(3.14, 1.0);
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_dtype<complex<double> >(), a.get_dtype());
}

TEST(Array, StdVectorConstructor) {
    nd::array a;
    std::vector<float> v;

    // Empty vector
    a = v;
    EXPECT_EQ(make_strided_dim_type(ndt::make_dtype<float>()), a.get_dtype());
    EXPECT_EQ(1u, a.get_dtype().get_undim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(0, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // Size-10 vector
    for (int i = 0; i < 10; ++i) {
        v.push_back(i/0.5f);
    }
    a = v;
    EXPECT_EQ(make_strided_dim_type(ndt::make_dtype<float>()), a.get_dtype());
    EXPECT_EQ(1u, a.get_dtype().get_undim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(10, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ((int)sizeof(float), a.get_strides()[0]);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i/0.5f, a(i).as<float>());
    }
}

TEST(Array, StdVectorStringConstructor) {
    nd::array a;
    std::vector<std::string> v;

    // Empty vector
    a = v;
    EXPECT_EQ(make_strided_dim_type(make_string_type(string_encoding_utf_8)), a.get_dtype());
    EXPECT_EQ(1u, a.get_dtype().get_undim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(0, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // Size-5 vector
    v.push_back("this");
    v.push_back("is a test of");
    v.push_back("string");
    v.push_back("vectors");
    v.push_back("testing testing testing testing testing testing testing testing testing");
    a = v;
    EXPECT_EQ(make_strided_dim_type(make_string_type(string_encoding_utf_8)), a.get_dtype());
    EXPECT_EQ(1u, a.get_dtype().get_undim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(5, a.get_shape()[0]);
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ((intptr_t)a.get_dtype().at(0).get_data_size(), a.get_strides()[0]);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(v[i], a(i).as<string>());
    }
}

TEST(Array, AsScalar) {
    nd::array a;

    a = nd::empty(ndt::make_dtype<float>());
    a.val_assign(3.14f);
    EXPECT_EQ(3.14f, a.as<float>());
    EXPECT_EQ(3.14f, a.as<double>());
    EXPECT_THROW(a.as<int64_t>(), runtime_error);
    EXPECT_EQ(3, a.as<int64_t>(assign_error_overflow));
    EXPECT_THROW(a.as<dynd_bool>(), runtime_error);
    EXPECT_THROW(a.as<dynd_bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<dynd_bool>(assign_error_none));
    EXPECT_THROW(a.as<bool>(), runtime_error);
    EXPECT_THROW(a.as<bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<bool>(assign_error_none));

    a = nd::empty(ndt::make_dtype<double>());
    a.val_assign(3.141592653589);
    EXPECT_EQ(3.141592653589, a.as<double>());
    EXPECT_THROW(a.as<float>(assign_error_inexact), runtime_error);
    EXPECT_EQ(3.141592653589f, a.as<float>());
}

TEST(Array, ConstructorMemoryLayouts) {
    nd::array a, b;
    ndt::type dt(int16_type_id), dt2(int32_type_id);
    intptr_t shape[6];
    int axisperm[6];

    // The strides are set to zero for size-one dimensions
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    axisperm[0] = 0;
    axisperm[1] = 1;
    axisperm[2] = 2;
    a = nd::make_strided_array(dt, 3, shape, nd::read_access_flag|nd::write_access_flag, axisperm);
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);
    EXPECT_EQ(0, a.get_strides()[2]);
    b = empty_like(a);
    EXPECT_EQ(3u, b.get_strides().size());
    EXPECT_EQ(0, b.get_strides()[0]);
    EXPECT_EQ(0, b.get_strides()[1]);
    EXPECT_EQ(0, b.get_strides()[2]);

    // Test all permutations of memory layouts from 1 through 6 dimensions
    for (size_t ndim = 1; ndim <= 6; ++ndim) {
        // Go through all the permutations on ndim elements
        // to check every memory layout
        intptr_t num_elements = 1;
        for (size_t i = 0; i < ndim; ++i) {
            shape[i] = i + 2;
            axisperm[i] = int(i);
            num_elements *= shape[ i];
        }
        do {
            // Test constructing the array using the perm
            a = nd::make_strided_array(dt, ndim, shape, nd::read_access_flag|nd::write_access_flag, axisperm);
            EXPECT_EQ(ndim, a.get_strides().size());
            intptr_t s = dt.get_data_size();
            for (size_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(s, a.get_strides()[axisperm[i]]);
                s *= shape[axisperm[i]];
            }
            // Test constructing the array using empty_like, which preserves the memory layout
            b = empty_like(a);
            EXPECT_EQ(ndim, b.get_strides().size());
            for (size_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(a.get_strides()[i], b.get_strides()[i]);
            }
            // Test constructing the array using empty_like with a different type, which preserves the memory layout
            b = empty_like(a, dt2);
            EXPECT_EQ(ndim, b.get_strides().size());
            for (size_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(2 * a.get_strides()[i], b.get_strides()[i]);
            }
            //cout << "perm " << axisperm[0] << " " << axisperm[1] << " " << axisperm[2] << "\n";
            //cout << "strides " << a.get_strides(0) << " " << a.get_strides(1) << " " << a.get_strides(2) << "\n";
        } while(next_permutation(&axisperm[0], &axisperm[0] + ndim));
    }
}

#if 0
TEST(Array, CharArrayConstructor) {
    nd::array a;
    char values[8] = {1,2,3,4,5,6,7,8};

    // Constructor assignment
    a = values;
    EXPECT_EQ(1, a.get_ndim());
    EXPECT_EQ(8, a.get_shape()[0]);
    EXPECT_EQ(ndt::make_dtype<char>(), a.get_dtype());
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

#ifdef DYND_INIT_LIST
TEST(Array, InitializerLists) {
    nd::array a = {1, 2, 3, 4, 5};
    EXPECT_EQ(5, a.get_num_elements());
    EXPECT_EQ(ndt::make_dtype<int>(), a.get_dtype());
    EXPECT_EQ(1, a.get_ndim());
    EXPECT_EQ(5, a.get_shape()[0]);
    EXPECT_EQ((int)sizeof(int), a.get_strides()[0]);
    const int *ptr_i = (const int *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);

    nd::array b = {{1., 2., 3.}, {4., 5., 6.25}};
    EXPECT_EQ(6, b.get_num_elements());
    EXPECT_EQ(ndt::make_dtype<double>(), b.get_dtype());
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
    EXPECT_EQ(ndt::make_dtype<long long>(), a.get_dtype());
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
#endif // DYND_INIT_LIST

#endif

TEST(Array, InitFromNestedCArray) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    nd::array a = i0;
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<int>())), a.get_dtype());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(2, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_shape()[1]);
    EXPECT_EQ(3*(int)sizeof(int), a.get_strides()[0]);
    EXPECT_EQ((int)sizeof(int), a.get_strides()[1]);
    const int *ptr_i = (const int *)a.get_ndo()->m_data_pointer;
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);
    EXPECT_EQ(6, ptr_i[5]);

    float i1[2][2][3] = {{{1,2,3}, {1.5f, 2.5f, 3.5f}}, {{-10, 0, -3.1f}, {9,8,7}}};
    a = i1;
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<float>()))), a.get_dtype());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(2, a.get_shape()[0]);
    EXPECT_EQ(2, a.get_shape()[1]);
    EXPECT_EQ(3, a.get_shape()[2]);
    EXPECT_EQ(6*(int)sizeof(float), a.get_strides()[0]);
    EXPECT_EQ(3*(int)sizeof(float), a.get_strides()[1]);
    EXPECT_EQ((int)sizeof(float), a.get_strides()[2]);
    const float *ptr_f = (float *)a.get_ndo()->m_data_pointer;
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

TEST(Array, Storage) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    nd::array a = i0;

    nd::array b = a.storage();
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(ndt::make_dtype<int>())), a.get_dtype());
    EXPECT_EQ(make_strided_dim_type(make_strided_dim_type(make_fixedbytes_type(4, 4))), b.get_dtype());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());
}
