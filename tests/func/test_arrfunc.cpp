//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/arrfunc_old_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/take_arrfunc.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(ArrFunc, Assignment)
{
  // Create an arrfunc for converting string to int
  nd::arrfunc af = make_arrfunc_from_assignment(
      ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::make_type<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::make_fixedstring(16), af.get_type()->get_arg_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(
      af.get(), af.get_type(), &ckb, 0, af.get_type()->get_return_type(), NULL,
      af.get_type()->get_arg_types_raw(), src_arrmeta, kernel_request_single,
      &eval::default_eval_context, nd::array(), nd::array());
  int int_out = 0;
  char str_in[16] = "3251";
  const char *str_in_ptr = str_in;
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(reinterpret_cast<char *>(&int_out), const_cast<char **>(&str_in_ptr),
        ckb.get());
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(
      af.get(), af.get_type(), &ckb, 0, af.get_type()->get_return_type(), NULL,
      af.get_type()->get_arg_types_raw(), src_arrmeta, kernel_request_strided,
      &eval::default_eval_context, nd::array(), nd::array());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  const char *strs_in_ptr = strs_in[0];
  expr_strided_t ustro = ckb.get()->get_function<expr_strided_t>();
  intptr_t strs_in_stride = sizeof(strs_in[0]);
  ustro(reinterpret_cast<char *>(&ints_out), sizeof(int),
        const_cast<char **>(&strs_in_ptr), &strs_in_stride, 3, ckb.get());
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

TEST(ArrFunc, Assignment_CallInterface)
{
  // Test with the unary operation prototype
  nd::arrfunc af = make_arrfunc_from_assignment(
      ndt::make_type<int>(), ndt::make_string(), assign_error_default);

  // Call it through the call() interface
  nd::array b = af("12345678");
  EXPECT_EQ(ndt::make_type<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);

  // Test with the expr operation prototype
  af = make_arrfunc_from_assignment(ndt::make_type<int>(), ndt::make_string(),
                                    assign_error_default);

  // Call it through the call() interface
  b = af("12345678");
  EXPECT_EQ(ndt::make_type<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);
}

TEST(ArrFunc, Property)
{
  // Create an arrfunc for getting the year from a date
  nd::arrfunc af = make_arrfunc_from_property(ndt::make_date(), "year");
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::make_type<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::make_date(), af.get_type()->get_arg_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(
      af.get(), af.get_type(), &ckb, 0, af.get_type()->get_return_type(), NULL,
      af.get_type()->get_arg_types_raw(), src_arrmeta, kernel_request_single,
      &eval::default_eval_context, nd::array(), nd::array());
  int int_out = 0;
  int date_in = date_ymd::to_days(2013, 12, 30);
  const char *date_in_ptr = reinterpret_cast<const char *>(&date_in);
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(reinterpret_cast<char *>(&int_out), const_cast<char **>(&date_in_ptr),
        ckb.get());
  EXPECT_EQ(2013, int_out);
}

TEST(ArrFunc, AssignmentAsExpr)
{
  // Create an arrfunc for converting string to int
  nd::arrfunc af = make_arrfunc_from_assignment(
      ndt::make_type<int>(), ndt::make_fixedstring(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::make_type<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::make_fixedstring(16), af.get_type()->get_arg_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(
      af.get(), af.get_type(), &ckb, 0, af.get_type()->get_return_type(), NULL,
      af.get_type()->get_arg_types_raw(), src_arrmeta, kernel_request_single,
      &eval::default_eval_context, nd::array(), nd::array());
  int int_out = 0;
  char str_in[16] = "3251";
  char *str_in_ptr = str_in;
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(reinterpret_cast<char *>(&int_out), &str_in_ptr, ckb.get());
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(
      af.get(), af.get_type(), &ckb, 0, af.get_type()->get_return_type(), NULL,
      af.get_type()->get_arg_types_raw(), src_arrmeta, kernel_request_strided,
      &eval::default_eval_context, nd::array(), nd::array());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  char *strs_in_ptr = strs_in[0];
  intptr_t strs_in_stride = 16;
  expr_strided_t ustro = ckb.get()->get_function<expr_strided_t>();
  ustro(reinterpret_cast<char *>(&ints_out), sizeof(int), &strs_in_ptr,
        &strs_in_stride, 3, ckb.get());
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

/*
// TODO Reenable once there's a convenient way to make the binary arrfunc
TEST(ArrFunc, Expr) {
    arrfunc_type_data af;
    // Create an arrfunc for adding two ints
    ndt::type add_ints_type = (nd::array((int)0) +
nd::array((int)0)).get_type();
    make_arrfunc_from_assignment(
                    ndt::make_type<int>(), add_ints_type,
                    expr_operation_funcproto, assign_error_default, af);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (arrfunc_proto_t)af.ckernel_funcproto);
    ASSERT_EQ(2, af.get_narg());
    ASSERT_EQ(ndt::make_type<int>(), af.get_return_type());
    ASSERT_EQ(ndt::make_type<int>(), af.get_arg_type(0));
    ASSERT_EQ(ndt::make_type<int>(), af.get_arg_type(1));

    const char *src_arrmeta[2] = {NULL, NULL};

    // Instantiate a single ckernel
    ckernel_builder ckb;
    af.instantiate(&af, &ckb, 0, af.get_return_type(), NULL,
                        af.get_arg_types(), src_arrmeta,
                        kernel_request_single, &eval::default_eval_context);
    int int_out = 0;
    int int_in1 = 1, int_in2 = 3;
    char *int_in_ptr[2] = {reinterpret_cast<char *>(&int_in1),
                        reinterpret_cast<char *>(&int_in2)};
    expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
    usngo(reinterpret_cast<char *>(&int_out), int_in_ptr, ckb.get());
    EXPECT_EQ(4, int_out);

    // Instantiate a strided ckernel
    ckb.reset();
    af.instantiate(&af, &ckb, 0, af.get_return_type(), NULL,
                        af.get_arg_types(), src_arrmeta,
                        kernel_request_strided, &eval::default_eval_context);
    int ints_out[3] = {0, 0, 0};
    int ints_in1[3] = {1,2,3}, ints_in2[3] = {5,-210,1234};
    char *ints_in_ptr[2] = {reinterpret_cast<char *>(&ints_in1),
                        reinterpret_cast<char *>(&ints_in2)};
    intptr_t ints_in_strides[2] = {sizeof(int), sizeof(int)};
    expr_strided_t ustro = ckb.get()->get_function<expr_strided_t>();
    ustro(reinterpret_cast<char *>(ints_out), sizeof(int),
                    ints_in_ptr, ints_in_strides, 3, ckb.get());
    EXPECT_EQ(6, ints_out[0]);
    EXPECT_EQ(-208, ints_out[1]);
    EXPECT_EQ(1237, ints_out[2]);
}
*/
