//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

TEST(VarArrayDType, Basic) {
    ndt::type d = ndt::make_var_dim(ndt::make_type<int32_t>());

    EXPECT_EQ(ndt::make_type<int32_t>(), d.p("element_type").as<ndt::type>());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(VarArrayDType, Shape) {
    ndt::type dfloat = ndt::make_type<float>();
    ndt::type darr1 = ndt::make_strided_dim(dfloat);
    ndt::type darr2 = ndt::make_var_dim(darr1);
    ndt::type darr3 = ndt::make_strided_dim(darr2);

    intptr_t shape[3] = {3, -1, 2};
    nd::array a = nd::dtyped_empty(3, shape, dfloat);
    EXPECT_EQ(darr3, a.get_type());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(-1, a.get_shape()[1]);
    EXPECT_EQ(2, a.get_shape()[2]);
}

TEST(VarArrayDType, DTypeSubscriptSimpleSingle) {
    nd::array n = parse_json("var * int32", "[2,4,6,8]");

    // Indexing collapses the leading dimension to just the int
    EXPECT_EQ(ndt::make_type<int>(), n(0).get_type());

    EXPECT_EQ(2, n(0).as<int>());
    EXPECT_EQ(4, n(1).as<int>());
    EXPECT_EQ(6, n(2).as<int>());
    EXPECT_EQ(8, n(3).as<int>());
    EXPECT_EQ(2, n(-4).as<int>());
    EXPECT_EQ(4, n(-3).as<int>());
    EXPECT_EQ(6, n(-2).as<int>());
    EXPECT_EQ(8, n(-1).as<int>());

    EXPECT_THROW(n(4), index_out_of_bounds);
    EXPECT_THROW(n(-5), index_out_of_bounds);
}

TEST(VarArrayDType, DTypeSubscriptSimpleSlice) {
    nd::array n = parse_json("var * int32", "[2,4,6,8]");

    // Slicing collapses the leading dimension to a strided array
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int>()), n(irange()).get_type());
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int>()), n(irange().by(-1)).get_type());
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int>()), n(1 <= irange() < 3).get_type());
    // But, indexing with a zero-sized index does not collapse from var to strided
    EXPECT_EQ(ndt::make_var_dim(ndt::make_type<int>()), n.at_array(0, NULL).get_type());

    EXPECT_EQ(2, n(1 <= irange() < 3).get_shape()[0]);
    EXPECT_EQ(4, n(1 <= irange() < 3)(0).as<int>());
    EXPECT_EQ(6, n(1 <= irange() < 3)(1).as<int>());

    EXPECT_EQ(4, n(irange().by(-1)).get_shape()[0]);
    EXPECT_EQ(8, n(irange().by(-1))(0).as<int>());
    EXPECT_EQ(6, n(irange().by(-1))(1).as<int>());
    EXPECT_EQ(4, n(irange().by(-1))(2).as<int>());
    EXPECT_EQ(2, n(irange().by(-1))(3).as<int>());

    EXPECT_EQ(4, n(irange()).get_shape()[0]);
    EXPECT_EQ(2, n(0).as<int>());
    EXPECT_EQ(4, n(1).as<int>());
    EXPECT_EQ(6, n(2).as<int>());
    EXPECT_EQ(8, n(3).as<int>());

    EXPECT_EQ(2, n(2 <= irange() < 4).get_shape()[0]);
    EXPECT_EQ(6, n(2 <= irange() < 4)(0).as<int>());
    EXPECT_EQ(8, n(2 <= irange() < 4)(1).as<int>());
}

TEST(VarArrayDType, DTypeSubscriptNested) {
    nd::array n = parse_json("var * var * int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");

    // Indexing with a zero-sized index does not convert the leading dim from var to strided
    EXPECT_EQ(ndt::type("var * var * int32"), n.at_array(0, NULL).get_type());
    // Indexing with a single index does not convert the next dim from var to strided
    EXPECT_EQ(ndt::type("var * int32"), n(0).get_type());
    EXPECT_EQ(ndt::type("int32"), n(0,0).get_type());

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n(0, irange()).get_shape()[0]);
    EXPECT_EQ(5, n(1, irange()).get_shape()[0]);
    EXPECT_EQ(0, n(2, irange()).get_shape()[0]);
    EXPECT_EQ(3, n(3, irange()).get_shape()[0]);

    // Check the individual values with positive indexes
    EXPECT_EQ(2, n(0,0).as<int>());
    EXPECT_EQ(4, n(0,1).as<int>());
    EXPECT_EQ(6, n(0,2).as<int>());
    EXPECT_EQ(8, n(0,3).as<int>());
    EXPECT_EQ(1, n(1,0).as<int>());
    EXPECT_EQ(3, n(1,1).as<int>());
    EXPECT_EQ(5, n(1,2).as<int>());
    EXPECT_EQ(7, n(1,3).as<int>());
    EXPECT_EQ(9, n(1,4).as<int>());
    EXPECT_EQ(-1, n(3,0).as<int>());
    EXPECT_EQ(-2, n(3,1).as<int>());
    EXPECT_EQ(-3, n(3,2).as<int>());

    // Check the individual values with negative indexes
    EXPECT_EQ(2, n(-4,-4).as<int>());
    EXPECT_EQ(4, n(-4,-3).as<int>());
    EXPECT_EQ(6, n(-4,-2).as<int>());
    EXPECT_EQ(8, n(-4,-1).as<int>());
    EXPECT_EQ(1, n(-3,-5).as<int>());
    EXPECT_EQ(3, n(-3,-4).as<int>());
    EXPECT_EQ(5, n(-3,-3).as<int>());
    EXPECT_EQ(7, n(-3,-2).as<int>());
    EXPECT_EQ(9, n(-3,-1).as<int>());
    EXPECT_EQ(-1, n(-1,-3).as<int>());
    EXPECT_EQ(-2, n(-1,-2).as<int>());
    EXPECT_EQ(-3, n(-1,-1).as<int>());

    // Out of bounds accesses
    EXPECT_THROW(n(0, 4), index_out_of_bounds);
    EXPECT_THROW(n(0, -5), index_out_of_bounds);
    EXPECT_THROW(n(1, 5), index_out_of_bounds);
    EXPECT_THROW(n(1, -6), index_out_of_bounds);
    EXPECT_THROW(n(2, 0), index_out_of_bounds);
    EXPECT_THROW(n(2, -1), index_out_of_bounds);
    EXPECT_THROW(n(3, 3), index_out_of_bounds);
    EXPECT_THROW(n(3, -4), index_out_of_bounds);
}

TEST(VarArrayDType, DTypeSubscriptFixedVarNested) {
    nd::array n = parse_json("4 * var * int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");

    EXPECT_EQ(ndt::type("4 * var * int32"), n.get_type());
    EXPECT_EQ(ndt::type("var * int32"), n(0).get_type());
    EXPECT_EQ(ndt::type("var * int32"), n.get_type().at(0));
    EXPECT_EQ(ndt::type("var * int32"), n.get_type().at_single(0));

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n(0, irange()).get_shape()[0]);
    EXPECT_EQ(5, n(1, irange()).get_shape()[0]);
    EXPECT_EQ(0, n(2, irange()).get_shape()[0]);
    EXPECT_EQ(3, n(3, irange()).get_shape()[0]);
}


TEST(VarArrayDType, DTypeSubscriptStridedVarNested) {
    nd::array n = parse_json("var * var * int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");
    // By indexing with a no-op slice, switch the var dim to strided
    n = n(irange());

    EXPECT_EQ(ndt::type("strided * var * int32"), n.get_type());
    EXPECT_EQ(ndt::type("var * int32"), n(0).get_type());

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n(0, irange()).get_shape()[0]);
    EXPECT_EQ(5, n(1, irange()).get_shape()[0]);
    EXPECT_EQ(0, n(2, irange()).get_shape()[0]);
    EXPECT_EQ(3, n(3, irange()).get_shape()[0]);
}

TEST(VarArrayDType, DTypeSubscriptFixedVarStruct) {
    nd::array n = parse_json("2 * var * {first_name: string, last_name: string, "
                    "gender: string[1], pictured: bool,}",
                    "[[{\"first_name\":\"Frank\",\"last_name\":\"Abrams\",\"gender\":\"M\",\"pictured\":true}],"
                    "[{\"first_name\":\"Melissa\",\"last_name\":\"Philips\",\"gender\":\"F\",\"pictured\":false}]]");

    nd::array nlastname = n(irange(), irange(), 1);
    EXPECT_EQ(ndt::type("strided * var * string"), nlastname.get_type());
    EXPECT_EQ("Abrams", nlastname(0,0).as<string>());
    EXPECT_EQ("Philips", nlastname(1,0).as<string>());

    nd::array ngender = n.p("gender");
    EXPECT_EQ(ndt::type("strided * var * string[1]"), ngender.get_type());
    EXPECT_EQ("M", ngender(0,0).as<string>());
    EXPECT_EQ("F", ngender(1,0).as<string>());
}

TEST(VarArrayDType, AccessCStructOfVar) {
    // A slightly complicated case of property access/indexing
    nd::array n = parse_json("var * {a: int32, b: var * int32}",
                    "[{\"a\":10, \"b\":[1,2,3,4,5]},"
                    " {\"a\":20, \"b\":[7,8,9]}]");

    EXPECT_EQ(ndt::type("var * {a: int32, b: var * int32}"), n.get_type());

    // In the property access, the first dimension will simplify to strided,
    // but the second shouldn't
    nd::array n2 = n.p("b");
    EXPECT_EQ(ndt::type("strided * var * int32"), n2.get_type());
    ASSERT_EQ(5, n2(0, irange()).get_shape()[0]);
    ASSERT_EQ(3, n2(1, irange()).get_shape()[0]);

    EXPECT_EQ(1, n2(0, 0).as<int>());
    EXPECT_EQ(2, n2(0, 1).as<int>());
    EXPECT_EQ(3, n2(0, 2).as<int>());
    EXPECT_EQ(4, n2(0, 3).as<int>());
    EXPECT_EQ(5, n2(0, 4).as<int>());
    EXPECT_EQ(7, n2(1, 0).as<int>());
    EXPECT_EQ(8, n2(1, 1).as<int>());
    EXPECT_EQ(9, n2(1, 2).as<int>());
}


TEST(VarArrayDType, AssignKernel) {
    nd::array a, b;
    unary_ckernel_builder k;

    // Assignment scalar -> uninitialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    b = 9.0;
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(1, a(irange()).get_shape()[0]);
    EXPECT_EQ(9, a(0).as<int>());
    k.reset();

    // Assignment scalar -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[3, 5, 7]");
    b = 9.0;
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
    k.reset();

    // Assignment initialized var array -> uninitialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    b = parse_json("var * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Assignment initialized var array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("var * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Broadcasting assignment initialized var array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("var * int32", "[9]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(9, a(0).as<int>());
    EXPECT_EQ(9, a(1).as<int>());
    EXPECT_EQ(9, a(2).as<int>());
    k.reset();

    // No-op assignment uninitialized var array -> uinitialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    b = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    // No error, a is still uninitialized
    k.reset();

    // Error assignment var array -> scalar
    a = 9.0;
    b = parse_json("var * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    EXPECT_THROW(make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(),
                                        b.get_type(), b.get_arrmeta(),
                                        kernel_request_single,
                                        &eval::default_eval_context),
                 broadcast_error);

    // Error assignment initialized var array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("var * int32", "[9, 2]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();
}

TEST(VarArrayDType, AssignVarStridedKernel) {
    nd::array a, b;
    unary_ckernel_builder k;
    int vals_int[] = {3,5,7};

    // Assignment strided array -> uninitialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Assignment strided array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0]");
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Error assignment strided array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0, 0]");
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Assignment initialized var array -> strided array
    a = nd::empty<int[3]>();
    a.vals() = 0;
    b = parse_json("var * int32", "[3, 5, 7]");
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Error assignment initialized var array -> strided array
    a = nd::empty<int[3]>();
    a.vals() = 0;
    b = parse_json("var * int32", "[3, 5, 7, 9]");
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Error assignment uninitialized var array -> strided array
    a = nd::empty<int[3]>();
    a.vals() = 0;
    b = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    EXPECT_EQ(strided_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    runtime_error);
    k.reset();
}

TEST(VarArrayDType, AssignVarFixedKernel) {
    nd::array a, b;
    unary_ckernel_builder k;

    // Assignment fixed array -> uninitialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    b = parse_json("3 * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Assignment fixed array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("3 * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    ASSERT_EQ(3, a(irange()).get_shape()[0]);
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Error assignment fixed array -> initialized var array
    a = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    parse_json(a, "[0, 0, 0, 0]");
    b = parse_json("3 * int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Assignment initialized var array -> cfixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = parse_json("var * int32", "[3, 5, 7]");
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a(0).as<int>());
    EXPECT_EQ(5, a(1).as<int>());
    EXPECT_EQ(7, a(2).as<int>());
    k.reset();

    // Error assignment initialized var array -> cfixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = parse_json("var * int32", "[3, 5, 7, 9]");
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Error assignment uninitialized var array -> cfixed array
    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<int>()));
    a.vals() = 0;
    b = nd::empty(ndt::make_var_dim(ndt::make_type<int>()));
    EXPECT_EQ(cfixed_dim_type_id, a.get_type().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_type().get_type_id());
    make_assignment_kernel(&k, 0, a.get_type(), a.get_arrmeta(), b.get_type(),
                           b.get_arrmeta(), kernel_request_single,
                           &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    runtime_error);
    k.reset();
}

TEST(VarDimDType, IsTypeSubarray) {
  EXPECT_TRUE(
      ndt::type("var * int32").is_type_subarray(ndt::type("var * int32")));
  EXPECT_TRUE(
      ndt::type("3 * var * int32").is_type_subarray(ndt::type("var * int32")));
  EXPECT_TRUE(
      ndt::type("var * var * int32").is_type_subarray(ndt::type("int32")));
  EXPECT_TRUE(
      ndt::type("var * int32").is_type_subarray(ndt::make_type<int32_t>()));
  EXPECT_FALSE(
      ndt::make_type<int32_t>().is_type_subarray(ndt::type("var * int32")));
  EXPECT_FALSE(ndt::type("var * int32")
                   .is_type_subarray(ndt::type("var * var * int32")));
  EXPECT_FALSE(
      ndt::type("var * int32").is_type_subarray(ndt::type("strided * int32")));
  EXPECT_FALSE(
      ndt::type("var * int32").is_type_subarray(ndt::type("3 * int32")));
  EXPECT_FALSE(
      ndt::type("strided * int32").is_type_subarray(ndt::type("var * int32")));
  EXPECT_FALSE(
      ndt::type("3 * int32").is_type_subarray(ndt::type("var * int32")));
}

TEST(VarArrayDType, AssignAfterCreated) {
  nd::array a;

  a = nd::empty("var * int32");
  // First assign an array, allocating var storage
  int vals[3] = {1, 3, 5};
  a.vals() = vals;
  EXPECT_JSON_EQ_ARR("[1, 3, 5]", a);
  // Now do a broadcasting assignment into that
  a.vals() = 7;
  EXPECT_JSON_EQ_ARR("[7, 7, 7]", a);
  // Finally do another array assignment into it
  a.vals() = vals;
  EXPECT_JSON_EQ_ARR("[1, 3, 5]", a);
}
