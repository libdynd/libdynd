//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../test_memory.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Array : public Memory<T> {
};

TYPED_TEST_CASE_P(Array);

TEST(Array, NullConstructor) {
    nd::array a;

    // Default-constructed nd::array is NULL and will crash if access is attempted
    EXPECT_EQ(NULL, a.get_memblock().get());
}

TEST(Array, FromValueConstructor) {
    nd::array a;
    // Bool
    a = nd::array(true);
    EXPECT_EQ(ndt::make_type<dynd_bool>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array(dynd_bool(true));
    EXPECT_EQ(ndt::make_type<dynd_bool>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    // Signed int
    a = nd::array((int8_t)1);
    EXPECT_EQ(ndt::make_type<int8_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((int16_t)1);
    EXPECT_EQ(ndt::make_type<int16_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((int32_t)1);
    EXPECT_EQ(ndt::make_type<int32_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((int64_t)1);
    EXPECT_EQ(ndt::make_type<int64_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    // Unsigned int
    a = nd::array((uint8_t)1);
    EXPECT_EQ(ndt::make_type<uint8_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((uint16_t)1);
    EXPECT_EQ(ndt::make_type<uint16_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((uint32_t)1);
    EXPECT_EQ(ndt::make_type<uint32_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array((uint64_t)1);
    EXPECT_EQ(ndt::make_type<uint64_t>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    // Floating point
    a = nd::array(1.0f);
    EXPECT_EQ(ndt::make_type<float>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array(1.0);
    EXPECT_EQ(ndt::make_type<double>(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    // Complex
    a = nd::array(dynd_complex<float>(1,1));
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
    a = nd::array(dynd_complex<double>(1,1));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), a.get_type());
    EXPECT_EQ((uint32_t) nd::read_access_flag | nd::immutable_access_flag, a.get_access_flags());
}

TEST(Array, FromValueConstructorRW) {
    nd::array a;
    // Bool
    a = nd::array_rw(true);
    EXPECT_EQ(ndt::make_type<dynd_bool>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw(dynd_bool(true));
    EXPECT_EQ(ndt::make_type<dynd_bool>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    // Signed int
    a = nd::array_rw((int8_t)1);
    EXPECT_EQ(ndt::make_type<int8_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((int16_t)1);
    EXPECT_EQ(ndt::make_type<int16_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((int32_t)1);
    EXPECT_EQ(ndt::make_type<int32_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((int64_t)1);
    EXPECT_EQ(ndt::make_type<int64_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    // Unsigned int
    a = nd::array_rw((uint8_t)1);
    EXPECT_EQ(ndt::make_type<uint8_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((uint16_t)1);
    EXPECT_EQ(ndt::make_type<uint16_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((uint32_t)1);
    EXPECT_EQ(ndt::make_type<uint32_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw((uint64_t)1);
    EXPECT_EQ(ndt::make_type<uint64_t>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    // Floating point
    a = nd::array_rw(1.0f);
    EXPECT_EQ(ndt::make_type<float>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw(1.0);
    EXPECT_EQ(ndt::make_type<double>(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    // Complex
    a = nd::array_rw(dynd_complex<float>(1,1));
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    a = nd::array_rw(dynd_complex<double>(1,1));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), a.get_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
}

TYPED_TEST_P(Array, ScalarConstructor) {
    nd::array a;
    // Scalar nd::array
    a = nd::empty(TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<float>()), a.get_type());
    EXPECT_EQ(ndt::make_type<float>(), a.get_type().get_canonical_type());
    EXPECT_EQ(nd::readwrite_access_flags, a.get_access_flags());
    EXPECT_TRUE(a.is_scalar());
}

TYPED_TEST_P(Array, OneDimConstructor) {
    // One-dimensional strided nd::array with one element
    nd::array a = nd::empty(1, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(1, TestFixture::MakeType(ndt::make_type<float>())), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(1, ndt::make_type<float>()), a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(1, a.get_dim_size(0));
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // One-dimensional nd::array
    a = nd::empty(3, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, TestFixture::MakeType(ndt::make_type<float>())), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_type<float>()), a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[0]);
}

TYPED_TEST_P(Array, TwoDimConstructor) {
    // Two-dimensional nd::array with a size-one dimension
    nd::array a = nd::empty(3, 1, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(1, TestFixture::MakeType(ndt::make_type<float>()))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(1, ndt::make_type<float>())), a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(1, a.get_shape()[1]);
    EXPECT_EQ(1, a.get_dim_size(1));
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);

    // Two-dimensional nd::array with a size-one dimension
    a = nd::empty(1, 3, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(1, ndt::make_fixed_dim(3, TestFixture::MakeType(ndt::make_type<float>()))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(1, ndt::make_fixed_dim(3, ndt::make_type<float>())), a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(1, a.get_dim_size(0));
    EXPECT_EQ(3, a.get_shape()[1]);
    EXPECT_EQ(3, a.get_dim_size(1));
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);

    // Two-dimensional nd::array
    a = nd::empty(3, 5, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5, TestFixture::MakeType(ndt::make_type<float>()))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5, ndt::make_type<float>())), a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(2u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(5, a.get_dim_size(1));
    EXPECT_EQ(2u, a.get_strides().size());
    EXPECT_EQ(5 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);
}

TYPED_TEST_P(Array, ThreeDimConstructor) {
    // Three-dimensional nd::array with size-one dimension
    nd::array a = nd::empty(1, 5, 4, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(1, ndt::make_fixed_dim(5,
                ndt::make_fixed_dim(4, TestFixture::MakeType(ndt::make_type<float>())))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(1, ndt::make_fixed_dim(5, ndt::make_fixed_dim(4, ndt::make_type<float>()))),
                a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(1, a.get_shape()[0]);
    EXPECT_EQ(1, a.get_dim_size(0));
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(5, a.get_dim_size(1));
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(4, a.get_dim_size(2));
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);

    // Three-dimensional nd::array with size-one dimension
    a = nd::empty(3, 1, 4, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(1,
                ndt::make_fixed_dim(4, TestFixture::MakeType(ndt::make_type<float>())))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(1, ndt::make_fixed_dim(4, ndt::make_type<float>()))),
                a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(1, a.get_shape()[1]);
    EXPECT_EQ(1, a.get_dim_size(1));
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(4, a.get_dim_size(2));
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(0, a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);

    // Three-dimensional nd::array with size-one dimension
    a = nd::empty(3, 5, 1, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5,
                ndt::make_fixed_dim(1, TestFixture::MakeType(ndt::make_type<float>())))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5, ndt::make_fixed_dim(1, ndt::make_type<float>()))),
                a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(5, a.get_dim_size(1));
    EXPECT_EQ(1, a.get_shape()[2]);
    EXPECT_EQ(1, a.get_dim_size(2));
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(5 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(0, a.get_strides()[2]);

    // Three-dimensional nd::array
    a = nd::empty(3, 5, 4, TestFixture::MakeType(ndt::make_type<float>()));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5,
                ndt::make_fixed_dim(4, TestFixture::MakeType(ndt::make_type<float>())))), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim(5, ndt::make_fixed_dim(4, ndt::make_type<float>()))),
                a.get_type().get_canonical_type());
    EXPECT_FALSE(a.is_scalar());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(3, a.get_dim_size(0));
    EXPECT_EQ(5, a.get_shape()[1]);
    EXPECT_EQ(5, a.get_dim_size(1));
    EXPECT_EQ(4, a.get_shape()[2]);
    EXPECT_EQ(4, a.get_dim_size(2));
    EXPECT_EQ(3u, a.get_strides().size());
    EXPECT_EQ(5 * 4 * sizeof(float), (unsigned)a.get_strides()[0]);
    EXPECT_EQ(4 * sizeof(float), (unsigned)a.get_strides()[1]);
    EXPECT_EQ(sizeof(float), (unsigned)a.get_strides()[2]);
}

TEST(Array, IntScalarConstructor) {
    stringstream ss;

    nd::array a = 3;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<int>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3,\n      type=\"int32\")", ss.str());

    a = (int8_t)1;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<int8_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(1,\n      type=\"int8\")", ss.str());

    a = (int16_t)2;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<int16_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(2,\n      type=\"int16\")", ss.str());

    a = (int32_t)3;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<int32_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3,\n      type=\"int32\")", ss.str());

    a = (int64_t)4;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<int64_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(4,\n      type=\"int64\")", ss.str());
}

TEST(Array, UIntScalarConstructor) {
    stringstream ss;

    nd::array a = (uint8_t)5;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<uint8_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(5,\n      type=\"uint8\")", ss.str());

    a = (uint16_t)6;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<uint16_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(6,\n      type=\"uint16\")", ss.str());

    a = (uint32_t)7;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<uint32_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(7,\n      type=\"uint32\")", ss.str());

    a = (uint64_t)8;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<uint64_t>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(8,\n      type=\"uint64\")", ss.str());
}

TEST(Array, FloatScalarConstructor) {
    stringstream ss;

    nd::array a = 3.25f;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<float>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3.25,\n      type=\"float32\")", ss.str());

    a = 3.5;
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<double>(), a.get_type());
    ss.str(""); ss << a;
    EXPECT_EQ("array(3.5,\n      type=\"float64\")", ss.str());

    a = dynd_complex<float>(3.14f, 1.0f);
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(), a.get_type());

    a = dynd_complex<double>(3.14, 1.0);
    EXPECT_TRUE(a.is_scalar());
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), a.get_type());
}

TEST(Array, StdVectorConstructor) {
    nd::array a;
    std::vector<float> v;

    // Empty vector
    a = v;
    EXPECT_EQ(ndt::make_fixed_dim(0, ndt::make_type<float>()), a.get_type());
    EXPECT_EQ(1, a.get_type().get_ndim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(0, a.get_shape()[0]);
    EXPECT_EQ(0, a.get_dim_size(0));
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // Size-10 vector
    for (int i = 0; i < 10; ++i) {
        v.push_back(i/0.5f);
    }
    a = v;
    EXPECT_EQ(ndt::make_fixed_dim(10, ndt::make_type<float>()), a.get_type());
    EXPECT_EQ(1, a.get_type().get_ndim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(10, a.get_shape()[0]);
    EXPECT_EQ(10, a.get_dim_size(0));
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
    EXPECT_EQ(ndt::type("0 * string"), a.get_type());
    EXPECT_EQ(1, a.get_type().get_ndim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(0, a.get_shape()[0]);
    EXPECT_EQ(0, a.get_dim_size(0));
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ(0, a.get_strides()[0]);

    // Size-5 vector
    v.push_back("this");
    v.push_back("is a test of");
    v.push_back("string");
    v.push_back("vectors");
    v.push_back("testing testing testing testing testing testing testing testing testing");
    a = v;
    EXPECT_EQ(ndt::type("5 * string"), a.get_type());
    EXPECT_EQ(1, a.get_type().get_ndim());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(5, a.get_shape()[0]);
    EXPECT_EQ(5, a.get_dim_size(0));
    EXPECT_EQ(1u, a.get_strides().size());
    EXPECT_EQ((intptr_t)a.get_type().at(0).get_data_size(), a.get_strides()[0]);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(v[i], a(i).as<string>());
    }
}

TYPED_TEST_P(Array, AsScalar) {
    nd::array a;

    a = nd::empty(TestFixture::MakeType(ndt::make_type<float>()));
    a.val_assign(3.14f);
    EXPECT_EQ(3.14f, a.as<float>());
    EXPECT_EQ(3.14f, a.as<double>());
    EXPECT_THROW(a.as<int64_t>(), runtime_error);
    EXPECT_EQ(3, a.as<int64_t>(assign_error_overflow));
    EXPECT_THROW(a.as<dynd_bool>(), runtime_error);
    EXPECT_THROW(a.as<dynd_bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<dynd_bool>(assign_error_nocheck));
    EXPECT_THROW(a.as<bool>(), runtime_error);
    EXPECT_THROW(a.as<bool>(assign_error_overflow), runtime_error);
    EXPECT_EQ(true, a.as<bool>(assign_error_nocheck));

    a = nd::empty(TestFixture::MakeType(ndt::make_type<double>()));
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
    for (intptr_t ndim = 1; ndim <= 6; ++ndim) {
        // Go through all the permutations on ndim elements
        // to check every memory layout
        intptr_t num_elements = 1;
        for (intptr_t i = 0; i < ndim; ++i) {
            shape[i] = i + 2;
            axisperm[i] = int(i);
            num_elements *= shape[ i];
        }
        do {
            // Test constructing the array using the perm
            a = nd::make_strided_array(dt, ndim, shape, nd::read_access_flag|nd::write_access_flag, axisperm);
            EXPECT_EQ(ndim, (intptr_t)a.get_strides().size());
            intptr_t s = dt.get_data_size();
            for (intptr_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(s, a.get_strides()[axisperm[i]]);
                s *= shape[axisperm[i]];
            }
            // Test constructing the array using empty_like, which preserves the memory layout
            b = empty_like(a);
            EXPECT_EQ(ndim, (intptr_t)b.get_strides().size());
            for (intptr_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(a.get_strides()[i], b.get_strides()[i]);
            }
            // Test constructing the array using empty_like with a different type, which preserves the memory layout
            b = empty_like(a, dt2);
            EXPECT_EQ(ndim, (intptr_t)b.get_strides().size());
            for (intptr_t i = 0; i < ndim; ++i) {
                EXPECT_EQ(2 * a.get_strides()[i], b.get_strides()[i]);
            }
            //cout << "perm " << axisperm[0] << " " << axisperm[1] << " " << axisperm[2] << "\n";
            //cout << "strides " << a.get_strides(0) << " " << a.get_strides(1) << " " << a.get_strides(2) << "\n";
        } while(next_permutation(&axisperm[0], &axisperm[0] + ndim));
    }
}

TEST(Array, InitFromInitializerLists) {
    nd::array a = {1, 2, 3, 4, 5};
    EXPECT_EQ(ndt::make_type<int>(), a.get_dtype());
    ASSERT_EQ(1, a.get_ndim());
    ASSERT_EQ(5, a.get_shape()[0]);
    ASSERT_EQ(5, a.get_dim_size(0));
    EXPECT_EQ((int)sizeof(int), a.get_strides()[0]);
    const int *ptr_i = (const int *)a.get_readonly_originptr();
    EXPECT_EQ(1, ptr_i[0]);
    EXPECT_EQ(2, ptr_i[1]);
    EXPECT_EQ(3, ptr_i[2]);
    EXPECT_EQ(4, ptr_i[3]);
    EXPECT_EQ(5, ptr_i[4]);

#ifndef DYND_NESTED_INIT_LIST_BUG
    nd::array b = {{1., 2., 3.}, {4., 5., 6.25}};
    EXPECT_EQ(ndt::make_type<double>(), b.get_dtype());
    ASSERT_EQ(2, b.get_ndim());
    ASSERT_EQ(2, b.get_shape()[0]);
    ASSERT_EQ(2, b.get_dim_size(0));
    ASSERT_EQ(3, b.get_shape()[1]);
    ASSERT_EQ(3, b.get_dim_size(1));
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
    EXPECT_EQ(ndt::make_type<long long>(), a.get_dtype());
    ASSERT_EQ(3, a.get_ndim());
    ASSERT_EQ(2, a.get_shape()[0]);
    ASSERT_EQ(2, a.get_dim_size(0));
    ASSERT_EQ(2, a.get_shape()[1]);
    ASSERT_EQ(2, a.get_dim_size(1));
    ASSERT_EQ(2, a.get_shape()[2]);
    ASSERT_EQ(2, a.get_dim_size(2));
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
#endif // DYND_NESTED_INIT_LIST_BUG
}

TEST(Array, InitFromNestedCArray) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    nd::array a = i0;
    EXPECT_EQ(ndt::type("2 * 3 * int"), a.get_type());
    EXPECT_JSON_EQ_ARR("[[1,2,3], [4,5,6]]", a);

    float i1[2][2][3] = {{{1,2,3}, {1.5f, 2.5f, 3.5f}}, {{-10, 0, -3.1f}, {9,8,7}}};
    a = i1;
    EXPECT_EQ(ndt::type("2 * 2 * 3 * float32"), a.get_type());
    EXPECT_JSON_EQ_ARR("[[[1,2,3], [1.5,2.5,3.5]], [[-10,0,-3.1], [9,8,7]]]", a);
}

TEST(Array, Storage) {
    int i0[2][3] = {{1,2,3}, {4,5,6}};
    nd::array a = i0;

    nd::array b = a.storage();
    EXPECT_EQ(ndt::make_fixed_dim(2, ndt::make_fixed_dim(3, ndt::make_type<int>())), a.get_type());
    EXPECT_EQ(ndt::make_fixed_dim(2, ndt::make_fixed_dim(3, ndt::make_fixedbytes(4, 4))), b.get_type());
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_shape(), b.get_shape());
    EXPECT_EQ(a.get_strides(), b.get_strides());
}

TEST(Array, SimplePrint) {
  int vals[3] = {1, 2, 3};
  nd::array a = vals;
  stringstream ss;
  ss << a;
  EXPECT_EQ("array([1, 2, 3],\n      type=\"3 * int32\")", ss.str());
}

TEST(Array, SimplePrintEmpty) {
  std::vector<float> v;
  nd::array a = v;
  stringstream ss;
  ss << a;
  EXPECT_EQ("array([],\n      type=\"0 * float32\")", ss.str());
}

TEST(Array, PrintBoolScalar) {
  nd::array a(true);
  stringstream ss;
  ss << a;
  EXPECT_EQ("array(True,\n      type=\"bool\")", ss.str());
}

TEST(Array, PrintBoolVector) {
  nd::array a = nd::empty(3, "bool");
  a.vals() = true;
  stringstream ss;
  ss << a;
  EXPECT_EQ("array([True, True, True],\n      type=\"3 * bool\")", ss.str());
}

REGISTER_TYPED_TEST_CASE_P(Array, ScalarConstructor, OneDimConstructor, TwoDimConstructor, ThreeDimConstructor, AsScalar);

INSTANTIATE_TYPED_TEST_CASE_P(Default, Array, DefaultMemory);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Array, CUDAMemory);
#endif // DYND_CUDA
