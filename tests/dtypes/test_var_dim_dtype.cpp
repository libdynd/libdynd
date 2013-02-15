//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/tuple_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

TEST(VarArrayDType, Shape) {
    dtype dfloat = make_dtype<float>();
    dtype darr1 = make_strided_dim_dtype(dfloat);
    dtype darr2 = make_var_dim_dtype(darr1);
    dtype darr3 = make_strided_dim_dtype(darr2);

    intptr_t shape[3] = {3, -1, 2};
    ndobject a = make_strided_ndobject(dfloat, 3, shape);
    EXPECT_EQ(darr3, a.get_dtype());
    EXPECT_EQ(3u, a.get_shape().size());
    EXPECT_EQ(3, a.get_shape()[0]);
    EXPECT_EQ(-1, a.get_shape()[1]);
    EXPECT_EQ(2, a.get_shape()[2]);
}

TEST(VarArrayDType, DTypeSubscriptSimpleSingle) {
    ndobject n = parse_json("VarDim, int32", "[2,4,6,8]");

    // Indexing collapses the leading dimension to just the int
    EXPECT_EQ(make_dtype<int>(), n.at(0).get_dtype());

    EXPECT_EQ(2, n.at(0).as<int>());
    EXPECT_EQ(4, n.at(1).as<int>());
    EXPECT_EQ(6, n.at(2).as<int>());
    EXPECT_EQ(8, n.at(3).as<int>());
    EXPECT_EQ(2, n.at(-4).as<int>());
    EXPECT_EQ(4, n.at(-3).as<int>());
    EXPECT_EQ(6, n.at(-2).as<int>());
    EXPECT_EQ(8, n.at(-1).as<int>());

    EXPECT_THROW(n.at(4), index_out_of_bounds);
    EXPECT_THROW(n.at(-5), index_out_of_bounds);
}

TEST(VarArrayDType, DTypeSubscriptSimpleSlice) {
    ndobject n = parse_json("VarDim, int32", "[2,4,6,8]");

    // Slicing collapses the leading dimension to a strided array
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), n.at(irange()).get_dtype());
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), n.at(irange() / -1).get_dtype());
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), n.at(1 <= irange() < 3).get_dtype());
    // In particular, indexing with a zero-sized index converts from var to strided
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int>()), n.at_array(0, NULL).get_dtype());

    EXPECT_EQ(2, n.at(1 <= irange() < 3).get_shape()[0]);
    EXPECT_EQ(4, n.at(1 <= irange() < 3).at(0).as<int>());
    EXPECT_EQ(6, n.at(1 <= irange() < 3).at(1).as<int>());

    EXPECT_EQ(4, n.at(irange() / -1).get_shape()[0]);
    EXPECT_EQ(8, n.at(irange() / -1).at(0).as<int>());
    EXPECT_EQ(6, n.at(irange() / -1).at(1).as<int>());
    EXPECT_EQ(4, n.at(irange() / -1).at(2).as<int>());
    EXPECT_EQ(2, n.at(irange() / -1).at(3).as<int>());

    EXPECT_EQ(4, n.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(2, n.at_array(0, NULL).at(0).as<int>());
    EXPECT_EQ(4, n.at_array(0, NULL).at(1).as<int>());
    EXPECT_EQ(6, n.at_array(0, NULL).at(2).as<int>());
    EXPECT_EQ(8, n.at_array(0, NULL).at(3).as<int>());

    EXPECT_THROW(n.at(2 <= irange() <= 4), irange_out_of_bounds);
}

TEST(VarArrayDType, DTypeSubscriptNested) {
    ndobject n = parse_json("VarDim, VarDim, int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");

    // Indexing with a zero-sized index converts the leading dim from var to strided
    EXPECT_EQ(dtype("M, VarDim, int32"), n.at_array(0, NULL).get_dtype());
    // Indexing with a single index converts the next dim from var to strided
    EXPECT_EQ(dtype("M, int32"), n.at(0).get_dtype());
    EXPECT_EQ(dtype("int32"), n.at(0,0).get_dtype());

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n.at(0).get_shape()[0]);
    EXPECT_EQ(5, n.at(1).get_shape()[0]);
    EXPECT_EQ(0, n.at(2).get_shape()[0]);
    EXPECT_EQ(3, n.at(3).get_shape()[0]);

    // Check the individual values with positive indexes
    EXPECT_EQ(2, n.at(0,0).as<int>());
    EXPECT_EQ(4, n.at(0,1).as<int>());
    EXPECT_EQ(6, n.at(0,2).as<int>());
    EXPECT_EQ(8, n.at(0,3).as<int>());
    EXPECT_EQ(1, n.at(1,0).as<int>());
    EXPECT_EQ(3, n.at(1,1).as<int>());
    EXPECT_EQ(5, n.at(1,2).as<int>());
    EXPECT_EQ(7, n.at(1,3).as<int>());
    EXPECT_EQ(9, n.at(1,4).as<int>());
    EXPECT_EQ(-1, n.at(3,0).as<int>());
    EXPECT_EQ(-2, n.at(3,1).as<int>());
    EXPECT_EQ(-3, n.at(3,2).as<int>());

    // Check the individual values with negative indexes
    EXPECT_EQ(2, n.at(-4,-4).as<int>());
    EXPECT_EQ(4, n.at(-4,-3).as<int>());
    EXPECT_EQ(6, n.at(-4,-2).as<int>());
    EXPECT_EQ(8, n.at(-4,-1).as<int>());
    EXPECT_EQ(1, n.at(-3,-5).as<int>());
    EXPECT_EQ(3, n.at(-3,-4).as<int>());
    EXPECT_EQ(5, n.at(-3,-3).as<int>());
    EXPECT_EQ(7, n.at(-3,-2).as<int>());
    EXPECT_EQ(9, n.at(-3,-1).as<int>());
    EXPECT_EQ(-1, n.at(-1,-3).as<int>());
    EXPECT_EQ(-2, n.at(-1,-2).as<int>());
    EXPECT_EQ(-3, n.at(-1,-1).as<int>());

    // Out of bounds accesses
    EXPECT_THROW(n.at(0, 4), index_out_of_bounds);
    EXPECT_THROW(n.at(0, -5), index_out_of_bounds);
    EXPECT_THROW(n.at(1, 5), index_out_of_bounds);
    EXPECT_THROW(n.at(1, -6), index_out_of_bounds);
    EXPECT_THROW(n.at(2, 0), index_out_of_bounds);
    EXPECT_THROW(n.at(2, -1), index_out_of_bounds);
    EXPECT_THROW(n.at(3, 3), index_out_of_bounds);
    EXPECT_THROW(n.at(3, -4), index_out_of_bounds);
}

TEST(VarArrayDType, DTypeSubscriptFixedVarNested) {
    ndobject n = parse_json("4, VarDim, int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");

    EXPECT_EQ(dtype("4, VarDim, int32"), n.get_dtype());
    EXPECT_EQ(dtype("M, int32"), n.at(0).get_dtype());
    EXPECT_EQ(dtype("M, int32"), n.get_dtype().at(0));
    EXPECT_EQ(dtype("VarDim, int32"), n.get_dtype().at_single(0));

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n.at(0).get_shape()[0]);
    EXPECT_EQ(5, n.at(1).get_shape()[0]);
    EXPECT_EQ(0, n.at(2).get_shape()[0]);
    EXPECT_EQ(3, n.at(3).get_shape()[0]);
}


TEST(VarArrayDType, DTypeSubscriptStridedVarNested) {
    ndobject n = parse_json("VarDim, VarDim, int32",
                    "[[2,4,6,8], [1,3,5,7,9], [], [-1,-2,-3]]");
    // By indexing with an empty index, switch the var dim to strided
    n = n.at_array(0, NULL);

    EXPECT_EQ(dtype("M, VarDim, int32"), n.get_dtype());
    EXPECT_EQ(dtype("M, int32"), n.at(0).get_dtype());

    // Validate the shapes after one level of indexing
    EXPECT_EQ(4, n.at(0).get_shape()[0]);
    EXPECT_EQ(5, n.at(1).get_shape()[0]);
    EXPECT_EQ(0, n.at(2).get_shape()[0]);
    EXPECT_EQ(3, n.at(3).get_shape()[0]);
}

TEST(VarArrayDType, DTypeSubscriptFixedVarStruct) {
    ndobject n = parse_json("2, VarDim, {first_name: string; last_name: string; "
                    "gender: string(1); pictured: bool;}",
                    "[[{\"first_name\":\"Frank\",\"last_name\":\"Abrams\",\"gender\":\"M\",\"pictured\":true}],"
                    "[{\"first_name\":\"Melissa\",\"last_name\":\"Philips\",\"gender\":\"F\",\"pictured\":false}]]");

    ndobject nlastname = n.at(irange(), irange(), 1);
    EXPECT_EQ(dtype("M, VarDim, string"), nlastname.get_dtype());
    EXPECT_EQ("Abrams", nlastname.at(0,0).as<string>());
    EXPECT_EQ("Philips", nlastname.at(1,0).as<string>());

    ndobject ngender = n.p("gender");
    EXPECT_EQ(dtype("M, VarDim, string(1)"), ngender.get_dtype());
    EXPECT_EQ("M", ngender.at(0,0).as<string>());
    EXPECT_EQ("F", ngender.at(1,0).as<string>());
}

TEST(VarArrayDType, AccessFixedStructOfVar) {
    // A slightly complicated case of property access/indexing
    ndobject n = parse_json("VarDim, {a: int32; b: VarDim, int32}",
                    "[{\"a\":10, \"b\":[1,2,3,4,5]},"
                    " {\"a\":20, \"b\":[7,8,9]}]");

    EXPECT_EQ(dtype("VarDim, {a: int32; b: VarDim, int32}"), n.get_dtype());

    // In the property access, the first dimension will simplify to strided,
    // but the second shouldn't
    ndobject n2 = n.p("b");
    EXPECT_EQ(dtype("M, VarDim, int32"), n2.get_dtype());
    EXPECT_EQ(5, n2.at(0).get_shape()[0]);
    EXPECT_EQ(3, n2.at(1).get_shape()[0]);

    EXPECT_EQ(1, n2.at(0, 0).as<int>());
    EXPECT_EQ(2, n2.at(0, 1).as<int>());
    EXPECT_EQ(3, n2.at(0, 2).as<int>());
    EXPECT_EQ(4, n2.at(0, 3).as<int>());
    EXPECT_EQ(5, n2.at(0, 4).as<int>());
    EXPECT_EQ(7, n2.at(1, 0).as<int>());
    EXPECT_EQ(8, n2.at(1, 1).as<int>());
    EXPECT_EQ(9, n2.at(1, 2).as<int>());
}


TEST(VarArrayDType, AssignKernel) {
    ndobject a, b;
    assignment_kernel k;

    // Assignment scalar -> uninitialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    b = 9.0;
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(9, a.at(0).as<int>());
    k.reset();

    // Assignment scalar -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[3, 5, 7]");
    b = 9.0;
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(9, a.at(0).as<int>());
    EXPECT_EQ(9, a.at(1).as<int>());
    EXPECT_EQ(9, a.at(2).as<int>());
    k.reset();

    // Assignment initialized var array -> uninitialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    b = parse_json("VarDim, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Assignment initialized var array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("VarDim, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Broadcasting assignment initialized var array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("VarDim, int32", "[9]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(9, a.at(0).as<int>());
    EXPECT_EQ(9, a.at(1).as<int>());
    EXPECT_EQ(9, a.at(2).as<int>());
    k.reset();

    // No-op assignment uninitialized var array -> uinitialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    b = ndobject(make_var_dim_dtype(make_dtype<int>()));
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    // No error, a is still uninitialized
    k.reset();

    // Error assignment var array -> scalar
    a = 9.0;
    b = parse_json("VarDim, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    EXPECT_THROW(make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context),
                broadcast_error);

    // Error assignment initialized var array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("VarDim, int32", "[9, 2]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();
}

TEST(VarArrayDType, AssignVarStridedKernel) {
    ndobject a, b;
    assignment_kernel k;
    int vals_int[] = {3,5,7};

    // Assignment strided array -> uninitialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Assignment strided array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0]");
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Error assignment strided array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0, 0]");
    b = vals_int;
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(strided_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Assignment initialized var array -> strided array
    a = make_strided_ndobject(3, make_dtype<int>());
    a.vals() = 0;
    b = parse_json("VarDim, int32", "[3, 5, 7]");
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Error assignment initialized var array -> strided array
    a = make_strided_ndobject(3, make_dtype<int>());
    a.vals() = 0;
    b = parse_json("VarDim, int32", "[3, 5, 7, 9]");
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Error assignment uninitialized var array -> strided array
    a = make_strided_ndobject(3, make_dtype<int>());
    a.vals() = 0;
    b = ndobject(make_var_dim_dtype(make_dtype<int>()));
    EXPECT_EQ(strided_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    runtime_error);
    k.reset();
}

TEST(VarArrayDType, AssignVarFixedKernel) {
    ndobject a, b;
    assignment_kernel k;

    // Assignment fixed array -> uninitialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Assignment fixed array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0]");
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at_array(0, NULL).get_shape()[0]);
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Error assignment fixed array -> initialized var array
    a = ndobject(make_var_dim_dtype(make_dtype<int>()));
    parse_json(a, "[0, 0, 0, 0]");
    b = parse_json("3, int32", "[3, 5, 7]");
    EXPECT_EQ(var_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(fixed_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Assignment initialized var array -> fixed array
    a = ndobject(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = parse_json("VarDim, int32", "[3, 5, 7]");
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    k(a.get_readwrite_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(3, a.at(0).as<int>());
    EXPECT_EQ(5, a.at(1).as<int>());
    EXPECT_EQ(7, a.at(2).as<int>());
    k.reset();

    // Error assignment initialized var array -> fixed array
    a = ndobject(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = parse_json("VarDim, int32", "[3, 5, 7, 9]");
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    broadcast_error);
    k.reset();

    // Error assignment uninitialized var array -> strided array
    a = ndobject(make_fixed_dim_dtype(3, make_dtype<int>()));
    a.vals() = 0;
    b = ndobject(make_var_dim_dtype(make_dtype<int>()));
    EXPECT_EQ(fixed_dim_type_id, a.get_dtype().get_type_id());
    EXPECT_EQ(var_dim_type_id, b.get_dtype().get_type_id());
    make_assignment_kernel(&k, 0, a.get_dtype(), a.get_ndo_meta(),
                    b.get_dtype(), b.get_ndo_meta(), assign_error_default, &eval::default_eval_context);
    EXPECT_THROW(k(a.get_readwrite_originptr(),
                        b.get_readonly_originptr()),
                    runtime_error);
    k.reset();
}
