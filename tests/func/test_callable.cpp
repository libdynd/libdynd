//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/take.hpp>
#include <dynd/gfunc/call_gcallable.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(Callable, SingleStridedConstructor)
{
  nd::callable f(ndt::type("(int32) -> int32"),
                 [](ckernel_prefix *DYND_UNUSED(self), char * dst,
                    char * const * src) { *reinterpret_cast<int32 *>(dst) = *reinterpret_cast<int32 *>(src[0]) + 5; },
                 0);

  EXPECT_EQ(8, f(3));
}

TEST(Callable, Assignment)
{
  // Create an callable for converting string to int
  nd::callable af =
      make_callable_from_assignment(ndt::type::make<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::type::make<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::fixed_string_type::make(16), af.get_type()->get_pos_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_single, &eval::default_eval_context, 0, NULL,
                        std::map<std::string, ndt::type>());
  int int_out = 0;
  char str_in[16] = "3251";
  const char *str_in_ptr = str_in;
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(ckb.get(), reinterpret_cast<char *>(&int_out), const_cast<char **>(&str_in_ptr));
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_strided, &eval::default_eval_context, 0, NULL,
                        std::map<std::string, ndt::type>());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  const char *strs_in_ptr = strs_in[0];
  expr_strided_t ustro = ckb.get()->get_function<expr_strided_t>();
  intptr_t strs_in_stride = sizeof(strs_in[0]);
  ustro(ckb.get(), reinterpret_cast<char *>(&ints_out), sizeof(int), const_cast<char **>(&strs_in_ptr), &strs_in_stride,
        3);
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

static double func(int x, double y)
{
  return 2.0 * x + y;
}

TEST(Callable, Construction)
{
  nd::callable af0 = nd::functional::apply(&func);
  EXPECT_EQ(4.5, af0(1, 2.5).as<double>());

  nd::callable af1 = nd::functional::apply(&func, "y");
  EXPECT_EQ(4.5, af1(1, kwds("y", 2.5)).as<double>());

  nd::callable af2 = nd::functional::apply([](int x, int y) { return x - y; });
  EXPECT_EQ(-4, af2(3, 7).as<int>());

  nd::callable af3 = nd::functional::apply([](int x, int y) { return x - y; }, "y");
  EXPECT_EQ(-4, af3(3, kwds("y", 7)).as<int>());
}

TEST(Callable, CallOperator)
{
  nd::callable af = nd::functional::apply(&func);
  // Calling with positional arguments
  EXPECT_EQ(4.5, af(1, 2.5).as<double>());
  EXPECT_EQ(7.5, af(2, 3.5).as<double>());
  // Wrong number of positional argumetns
  EXPECT_THROW(af(2), invalid_argument);
  EXPECT_THROW(af(2, 3.5, 7), invalid_argument);
  // Extra keyword argument
  EXPECT_THROW(af(2, 3.5, kwds("x", 10)), invalid_argument);

  af = nd::functional::apply(&func, "x");
  // Calling with positional and keyword arguments
  EXPECT_EQ(4.5, af(1, kwds("x", 2.5)).as<double>());
  EXPECT_EQ(7.5, af(2, kwds("x", 3.5)).as<double>());
  // Wrong number of positional/keyword arguments
  EXPECT_THROW(af(2), invalid_argument);
  EXPECT_THROW(af(2, 3.5), invalid_argument);
  EXPECT_THROW(af(2, 3.5, 7), invalid_argument);
  // Extra/wrong keyword argument
  EXPECT_THROW(af(2, kwds("y", 3.5)), invalid_argument);
  EXPECT_THROW(af(2, kwds("x", 10, "y", 20)), invalid_argument);
  EXPECT_THROW(af(2, 3.5, kwds("x", 10, "y", 20)), invalid_argument);

  af = nd::functional::apply([]() { return 10; });
  // Calling with no arguments
  EXPECT_EQ(10, af().as<int>());
  // Calling with empty keyword arguments
  EXPECT_EQ(10, af(kwds()).as<int>());
  // Wrong number of positional/keyword arguments
  EXPECT_THROW(af(2), invalid_argument);
  EXPECT_THROW(af(kwds("y", 3.5)), invalid_argument);
}

TEST(Callable, DynamicCall)
{
  nd::callable af;

  nd::array values[3] = {7, 2.5, 5};
  const char *names[3] = {"x", "y", "z"};

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; });
  EXPECT_EQ(26.5, af(3, values).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "z");
  EXPECT_EQ(26.5, af(2, values, kwds(1, names + 2, values + 2)).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "y", "z");
  EXPECT_EQ(26.5, af(1, values, kwds(2, names + 1, values + 1)).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "x", "y", "z");
  EXPECT_EQ(26.5, af(kwds(3, names, values)).as<double>());
}

TEST(Callable, DecomposedDynamicCall)
{
  nd::callable af;

  ndt::type ret_tp;
  nd::array values[3] = {7, 2.5, 5};
  ndt::type types[3] = {values[0].get_type(), values[1].get_type(), values[2].get_type()};
  const char *const arrmetas[3] = {values[0].get_arrmeta(), values[1].get_arrmeta(), values[2].get_arrmeta()};
  char *const datas[3] = {values[0].get_ndo()->data.ptr, values[1].get_ndo()->data.ptr, values[2].get_ndo()->data.ptr};
  //  const char *names[3] = {"x", "y", "z"};

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; });
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5, (*af.get())(ret_tp, 3, types, arrmetas, datas, 0, NULL, map<std::string, ndt::type>()).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "z");
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5, (*af.get())(ret_tp, 2, types, arrmetas, datas, 1, values + 2, map<std::string, ndt::type>()).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "y", "z");
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5, (*af.get())(ret_tp, 1, types, arrmetas, datas, 2, values + 1, map<std::string, ndt::type>()).as<double>());

  //  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "x", "y", "z");
  //  ret_tp = af.get_ret_type();
  //  EXPECT_EQ(26.5, (*af.get())(ret_tp, 0, NULL, NULL, NULL, 3, values, map<string, ndt::type>()).as<double>());
}

TEST(Callable, KeywordParsing)
{
  nd::callable af0 = nd::functional::apply([](int x, int y) { return x + y; }, "y");
  EXPECT_EQ(5, af0(1, kwds("y", 4)).as<int>());
  EXPECT_THROW(af0(1, kwds("z", 4)).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0(1, kwds("Y", 4)).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0(1, kwds("y", 2.5)).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0(1, kwds("y", 4, "y", 2.5)).as<int>(), std::invalid_argument);
}

/*
TEST(Callable, Option)
{
  struct callable {
    int operator()(int x, int y) { return x + y; }

    static void
    resolve_option_vals(const callable_type_data *DYND_UNUSED(self),
                        const callable_type *DYND_UNUSED(self_tp),
                        intptr_t DYND_UNUSED(nsrc),
                        const ndt::type *DYND_UNUSED(src_tp), nd::array &kwds,
                        const std::map<std::string, ndt::type>
&DYND_UNUSED(tp_vars))
    {
      nd::array x = kwds.p("x");
      if (x.is_missing()) {
        x.vals() = 4;
      }
    }
  };

  nd::callable af = nd::functional::apply(callable(), "x");
  EXPECT_EQ(5, af(1, kwds("x", 4)).as<int>());

  af.set_as_option(&callable::resolve_option_vals, "x");
  EXPECT_EQ(6, af(1, kwds("x", 5)).as<int>());
  EXPECT_EQ(5, af(1).as<int>());
}
*/

TEST(Callable, Assignment_CallInterface)
{
  // Test with the unary operation prototype
  nd::callable af =
      make_callable_from_assignment(ndt::type::make<int>(), ndt::string_type::make(), assign_error_default);

  // Call it through the call() interface
  nd::array b = af("12345678");
  EXPECT_EQ(ndt::type::make<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);

  // Test with the expr operation prototype
  af = make_callable_from_assignment(ndt::type::make<int>(), ndt::string_type::make(), assign_error_default);

  // Call it through the call() interface
  b = af("12345678");
  EXPECT_EQ(ndt::type::make<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);
}

TEST(Callable, Property)
{
  // Create an callable for getting the year from a date
  nd::callable af = make_callable_from_property(ndt::date_type::make(), "year");
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::type::make<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::date_type::make(), af.get_type()->get_pos_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_single, &eval::default_eval_context, 0, NULL,
                        std::map<std::string, ndt::type>());
  int int_out = 0;
  int date_in = date_ymd::to_days(2013, 12, 30);
  const char *date_in_ptr = reinterpret_cast<const char *>(&date_in);
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(ckb.get(), reinterpret_cast<char *>(&int_out), const_cast<char **>(&date_in_ptr));
  EXPECT_EQ(2013, int_out);
}

TEST(Callable, AssignmentAsExpr)
{
  // Create an callable for converting string to int
  nd::callable af =
      make_callable_from_assignment(ndt::type::make<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::type::make<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::fixed_string_type::make(16), af.get_type()->get_pos_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  ckernel_builder<kernel_request_host> ckb;
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_single, &eval::default_eval_context, 0, NULL,
                        std::map<std::string, ndt::type>());
  int int_out = 0;
  char str_in[16] = "3251";
  char *str_in_ptr = str_in;
  expr_single_t usngo = ckb.get()->get_function<expr_single_t>();
  usngo(ckb.get(), reinterpret_cast<char *>(&int_out), &str_in_ptr);
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_strided, &eval::default_eval_context, 0, NULL,
                        std::map<std::string, ndt::type>());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  char *strs_in_ptr = strs_in[0];
  intptr_t strs_in_stride = 16;
  expr_strided_t ustro = ckb.get()->get_function<expr_strided_t>();
  ustro(ckb.get(), reinterpret_cast<char *>(&ints_out), sizeof(int), &strs_in_ptr, &strs_in_stride, 3);
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

/*
TEST(Callable, LLVM)
{
  nd::callable f = nd::functional::apply([](int x, int y) { return x + y; });
  // nd::callable f = nd::add::children[int32_type_id][int32_type_id];

  std::cout << f.get_single().ir << std::endl;
  std::exit(-1);
}
*/

/*
// TODO Reenable once there's a convenient way to make the binary callable
TEST(Callable, Expr) {
    callable_type_data af;
    // Create an callable for adding two ints
    ndt::type add_ints_type = (nd::array((int)0) +
nd::array((int)0)).get_type();
    make_callable_from_assignment(
                    ndt::type::make<int>(), add_ints_type,
                    expr_operation_funcproto, assign_error_default, af);
    // Validate that its types, etc are set right
    ASSERT_EQ(expr_operation_funcproto, (callable_proto_t)af.ckernel_funcproto);
    ASSERT_EQ(2, af.get_narg());
    ASSERT_EQ(ndt::type::make<int>(), af.get_return_type());
    ASSERT_EQ(ndt::type::make<int>(), af.get_arg_type(0));
    ASSERT_EQ(ndt::type::make<int>(), af.get_arg_type(1));

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
