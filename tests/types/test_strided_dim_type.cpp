
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/typed_data_assign.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/array.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

TEST(StridedArrayDType, Basic) {
    ndt::type d = ndt::make_strided_dim(ndt::make_type<int32_t>());

    EXPECT_EQ(1, d.get_ndim());
    EXPECT_EQ(1, d.get_strided_ndim());
    EXPECT_EQ(ndt::make_type<int32_t>(), d.p("element_type").as<ndt::type>());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(StridedArrayDType, ReplaceScalarTypes) {
    ndt::type dafloat, dadouble, daint32;
    dafloat = ndt::make_strided_dim(ndt::make_type<float>());
    dadouble = ndt::make_strided_dim(ndt::make_type<double>());

    EXPECT_EQ(ndt::make_strided_dim(ndt::make_convert<float, double>()),
            dadouble.with_replaced_scalar_types(ndt::make_type<float>()));

    // Two dimensional array
    dafloat = ndt::make_strided_dim(dafloat);
    dadouble = ndt::make_strided_dim(dadouble);

    EXPECT_EQ(ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_convert<double, float>())),
            dafloat.with_replaced_scalar_types(ndt::make_type<double>()));
}

TEST(StridedArrayDType, DTypeAt) {
    ndt::type dfloat = ndt::make_type<float>();
    ndt::type darr1 = ndt::make_strided_dim(dfloat);
    ndt::type darr2 = ndt::make_strided_dim(darr1);
    ndt::type dtest;

    // indexing into a type with a slice produces another
    // strided array, so the type is unchanged.
    EXPECT_EQ(darr1, darr1.at(1 <= irange() < 3));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() < 3));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() < 3, irange() < 2));

    // Even if it's just one element, a slice still produces another array
    EXPECT_EQ(darr1, darr1.at(1 <= irange() <= 1));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() <= 1));
    EXPECT_EQ(darr2, darr2.at(1 <= irange() <= 1, 2 <= irange() <= 2));

    // Indexing with an integer collapses a dimension
    EXPECT_EQ(dfloat, darr1.at(1));
    EXPECT_EQ(darr1, darr2.at(1));
    EXPECT_EQ(darr1, darr2.at(1 <= irange() <= 1, 1));
    EXPECT_EQ(dfloat, darr2.at(2, 1));

    // Should get an exception with too many indices
    EXPECT_THROW(dfloat.at(1), too_many_indices);
    EXPECT_THROW(darr1.at(1, 2), too_many_indices);
    EXPECT_THROW(darr2.at(1, 2, 3), too_many_indices);
}

TEST(StridedArrayDType, IsExpression) {
    ndt::type dfloat = ndt::make_type<float>();
    ndt::type darr1 = ndt::make_strided_dim(dfloat);
    ndt::type darr2 = ndt::make_strided_dim(darr1);

    EXPECT_FALSE(darr1.is_expression());
    EXPECT_FALSE(darr2.is_expression());

    dfloat = ndt::make_convert(ndt::make_type<double>(), dfloat);
    darr1 = ndt::make_strided_dim(dfloat);
    darr2 = ndt::make_strided_dim(darr1);

    EXPECT_TRUE(darr1.is_expression());
    EXPECT_TRUE(darr2.is_expression());
}

TEST(StridedArrayDType, AssignKernel) {
    nd::array a, b;
    unary_ckernel_builder k;
    int vals_int[] = {3,5,7};

    // Assignment scalar -> strided array
    a = nd::array_rw(vals_int);
    b = 9.0;
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
    k.reset();

    // Assignment strided array -> strided array
    a = nd::empty<float[3]>();
    a.vals() = 0;
    b = vals_int;
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Assignment strided array -> scalar
    a = 9.0;
    b = vals_int;
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    EXPECT_THROW(make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(),
                                        b.get_type(), b.get_arrmeta(),
                                        kernel_request_single,
                                        &eval::default_eval_context),
                 broadcast_error);
}

TEST(StridedArrayDType, IsTypeSubarray) {
    EXPECT_TRUE(ndt::type("strided * int32").is_type_subarray(ndt::type("strided * int32")));
    EXPECT_TRUE(ndt::type("strided * strided * int32").is_type_subarray(ndt::type("strided * int32")));
    EXPECT_TRUE(ndt::type("strided * int32").is_type_subarray(ndt::make_type<int32_t>()));
    EXPECT_FALSE(ndt::make_type<int32_t>().is_type_subarray(ndt::type("strided * int32")));
    EXPECT_FALSE(ndt::type("strided * int32").is_type_subarray(ndt::type("strided * strided * int32")));
    EXPECT_FALSE(ndt::type("strided * int32").is_type_subarray(ndt::type("3 * int32")));
    EXPECT_FALSE(ndt::type("strided * int32").is_type_subarray(ndt::type("var * int32")));
    EXPECT_FALSE(ndt::type("3 * int32").is_type_subarray(ndt::type("strided * int32")));
    EXPECT_FALSE(ndt::type("var * int32").is_type_subarray(ndt::type("strided * int32")));
}
