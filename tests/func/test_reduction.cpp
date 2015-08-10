//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/json_parser.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Reduction, BuiltinSum_Kernel)
{
  ckernel_builder<kernel_request_host> ckb;
  expr_single_t fn;
  char *src = NULL;

  // int32
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, int32_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  int32_t s32 = 0, a32[3] = {1, -2, 12};
  src = (char *)&a32[0];
  fn(ckb.get(), (char *)&s32, &src);
  EXPECT_EQ(1, s32);
  src = (char *)&a32[1];
  fn(ckb.get(), (char *)&s32, &src);
  EXPECT_EQ(-1, s32);
  src = (char *)&a32[2];
  fn(ckb.get(), (char *)&s32, &src);
  EXPECT_EQ(11, s32);

  // int64
  ckb.reset();
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, int64_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  int64_t s64 = 0, a64[3] = {1, -20000000000LL, 12};
  src = (char *)&a64[0];
  fn(ckb.get(), (char *)&s64, &src);
  EXPECT_EQ(1, s64);
  src = (char *)&a64[1];
  fn(ckb.get(), (char *)&s64, &src);
  EXPECT_EQ(-19999999999LL, s64);
  src = (char *)&a64[2];
  fn(ckb.get(), (char *)&s64, &src);
  EXPECT_EQ(-19999999987LL, s64);

  // float32
  ckb.reset();
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, float32_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  float sf32 = 0, af32[3] = {1.25f, -2.5f, 12.125f};
  src = (char *)&af32[0];
  fn(ckb.get(), (char *)&sf32, &src);
  EXPECT_EQ(1.25f, sf32);
  src = (char *)&af32[1];
  fn(ckb.get(), (char *)&sf32, &src);
  EXPECT_EQ(-1.25f, sf32);
  src = (char *)&af32[2];
  fn(ckb.get(), (char *)&sf32, &src);
  EXPECT_EQ(10.875f, sf32);

  // float64
  ckb.reset();
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, float64_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  double sf64 = 0, af64[3] = {1.25, -2.5, 12.125};
  src = (char *)&af64[0];
  fn(ckb.get(), (char *)&sf64, &src);
  EXPECT_EQ(1.25, sf64);
  src = (char *)&af64[1];
  fn(ckb.get(), (char *)&sf64, &src);
  EXPECT_EQ(-1.25, sf64);
  src = (char *)&af64[2];
  fn(ckb.get(), (char *)&sf64, &src);
  EXPECT_EQ(10.875, sf64);

  // complex[float32]
  ckb.reset();
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, complex_float32_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  dynd::complex<float> scf32 = 0,
                       acf32[3] = {dynd::complex<float>(1.25f, -2.125f),
                                   dynd::complex<float>(-2.5f, 1.0f),
                                   dynd::complex<float>(12.125f, 12345.f)};
  src = (char *)&acf32[0];
  fn(ckb.get(), (char *)&scf32, &src);
  EXPECT_EQ(dynd::complex<float>(1.25f, -2.125f), scf32);
  src = (char *)&acf32[1];
  fn(ckb.get(), (char *)&scf32, &src);
  EXPECT_EQ(dynd::complex<float>(-1.25f, -1.125f), scf32);
  src = (char *)&acf32[2];
  fn(ckb.get(), (char *)&scf32, &src);
  EXPECT_EQ(dynd::complex<float>(10.875f, 12343.875f), scf32);

  // complex[float64]
  ckb.reset();
  kernels::make_builtin_sum_reduction_ckernel(&ckb, 0, complex_float64_type_id,
                                              kernel_request_single);
  fn = ckb.get()->get_function<expr_single_t>();
  dynd::complex<double> scf64 = 0,
                        acf64[3] = {dynd::complex<double>(1.25, -2.125),
                                    dynd::complex<double>(-2.5, 1.0),
                                    dynd::complex<double>(12.125, 12345.)};
  src = (char *)&acf64[0];
  fn(ckb.get(), (char *)&scf64, &src);
  EXPECT_EQ(dynd::complex<float>(1.25, -2.125), scf64);
  src = (char *)&acf64[1];
  fn(ckb.get(), (char *)&scf64, &src);
  EXPECT_EQ(dynd::complex<double>(-1.25, -1.125), scf64);
  src = (char *)&acf64[2];
  fn(ckb.get(), (char *)&scf64, &src);
  EXPECT_EQ(dynd::complex<double>(10.875, 12343.875), scf64);
}

TEST(Reduction, BuiltinSum_Lift0D_NoIdentity)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a zero-dimensional reduction callable (basically a no-op)
  bool reduction_dimflags[1] = {false};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("float32"), nd::callable(), false, 0,
      reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a = 1.25f;
  ASSERT_EQ(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(ndt::type::make<float>());
  ASSERT_EQ(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(1.25f, b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift0D_WithIdentity)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a zero-dimensional reduction callable (basically a no-op)
  // Use 100.f as the "identity" to confirm it's really being used
  bool reduction_dimflags[1] = {false};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("float32"), nd::callable(), false, 0,
      reduction_dimflags, true, true, false, nd::array(100.f));

  // Set up some data for the test reduction
  nd::array a = 1.25f;
  ASSERT_EQ(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(ndt::type::make<float>());
  ASSERT_EQ(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(100.f + 1.25f, b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift1D_NoIdentity)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a one-dimensional strided float32 reduction callable
  bool reduction_dimflags[1] = {true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * float32"), nd::callable(), false, 1,
      reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  float vals0[5] = {1.5, -22., 3.75, 1.125, -3.375};
  nd::array a = vals0;
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(ndt::type::make<float>());
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(vals0[0] + vals0[1] + vals0[2] + vals0[3] + vals0[4],
            b.as<float>());

  // Instantiate it again with some different data
  ckb.reset();
  float vals1[1] = {3.75f};
  a = vals1;
  src_tp[0] = a.get_type();
  src_arrmeta[0] = a.get_arrmeta();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(vals1[0], b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift1D_WithIdentity)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a one-dimensional strided float32 reduction callable
  // Use 100.f as the "identity" to confirm it's really being used
  bool reduction_dimflags[1] = {true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * float32"), nd::callable(), false, 1,
      reduction_dimflags, true, true, false, nd::array(100.f));

  // Set up some data for the test reduction
  float vals0[5] = {1.5, -22., 3.75, 1.125, -3.375};
  nd::array a = vals0;
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(ndt::type::make<float>());
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(100.f + vals0[0] + vals0[1] + vals0[2] + vals0[3] + vals0[4],
            b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {true, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      false, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(ndt::type::make<float>());
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(1.5f + 2.f + 7.f - 2.25f + 7.f + 2.125f, b.as<float>());

  // Instantiate it again with some different data
  ckb.reset();
  a = parse_json("1 * 2 * float32", "[[1.5, -2]]");
  src_tp[0] = a.get_type();
  src_arrmeta[0] = a.get_arrmeta();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(1.5f - 2.f, b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce_KeepDim)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {true, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      true, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(1, 1, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(1.5f + 2.f + 7.f - 2.25f + 7.f + 2.125f, b(0, 0).as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {false, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      false, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(2, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(2, b.get_shape()[0]);
  EXPECT_EQ(1.5f + 2.f + 7.f, b(0).as<float>());
  EXPECT_EQ(-2.25f + 7 + 2.125f, b(1).as<float>());

  // Instantiate it again with some different data
  ckb.reset();
  a = parse_json("1 * 2 * float32", "[[1.5, -2]]");
  b = nd::empty(1, "float32");
  src_tp[0] = a.get_type();
  src_arrmeta[0] = a.get_arrmeta();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(1, b.get_shape()[0]);
  EXPECT_EQ(1.5f - 2.f, b(0).as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce_KeepDim)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {false, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      true, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(2, 1, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(2, b.get_shape()[0]);
  EXPECT_EQ(1.5f + 2.f + 7.f, b(0, 0).as<float>());
  EXPECT_EQ(-2.25f + 7 + 2.125f, b(1, 0).as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {true, false};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      false, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(3, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(3, b.get_shape()[0]);
  EXPECT_EQ(1.5f - 2.25f, b(0).as<float>());
  EXPECT_EQ(2.f + 7.f, b(1).as<float>());
  EXPECT_EQ(7.f + 2.125f, b(2).as<float>());

  // Instantiate it again with some different data
  ckb.reset();
  a = parse_json("1 * 2 * float32", "[[1.5, -2]]");
  b = nd::empty(2, "float32");
  src_tp[0] = a.get_type();
  src_arrmeta[0] = a.get_arrmeta();
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(2, b.get_shape()[0]);
  EXPECT_EQ(1.5f, b(0).as<float>());
  EXPECT_EQ(-2.f, b(1).as<float>());
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast_KeepDim)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a two-dimensional strided float32 reduction callable
  bool reduction_dimflags[2] = {true, false};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * float32"), nd::callable(),
      true, 2, reduction_dimflags, true, true, false, nd::array());

  // Set up some data for the test reduction
  nd::array a =
      parse_json("2 * 3 * float32", "[[1.5, 2, 7], [-2.25, 7, 2.125]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(1, 3, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(3, b.get_shape()[1]);
  EXPECT_EQ(1.5f - 2.25f, b(0, 0).as<float>());
  EXPECT_EQ(2.f + 7.f, b(0, 1).as<float>());
  EXPECT_EQ(7.f + 2.125f, b(0, 2).as<float>());
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceReduceReduce)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a three-dimensional strided float32 reduction callable
  bool reduction_dimflags[3] = {true, true, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * Fixed * float32"),
      nd::callable(), false, 3, reduction_dimflags, true, true, false,
      nd::array());

  // Set up some data for the test reduction
  nd::array a = parse_json("2 * 3 * 2 * float32", "[[[1.5, -2.375], [2, 1.25], "
                                                  "[7, -0.5]], [[-2.25, 1], "
                                                  "[7, 0], [2.125, 0.25]]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty("float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  EXPECT_EQ(1.5f - 2.375f + 2.f + 1.25f + 7.f - 0.5f - 2.25f + 1.f + 7.f +
                2.125f + 0.25f,
            b.as<float>());
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_BroadcastReduceReduce)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a three-dimensional strided float32 reduction callable
  bool reduction_dimflags[3] = {false, true, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * Fixed * float32"),
      nd::callable(), false, 3, reduction_dimflags, true, true, false,
      nd::array());

  // Set up some data for the test reduction
  nd::array a = parse_json("2 * 3 * 2 * float32", "[[[1.5, -2.375], [2, 1.25], "
                                                  "[7, -0.5]], [[-2.25, 1], "
                                                  "[7, 0], [2.125, 0.25]]]");
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(2, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(2, b.get_shape()[0]);
  EXPECT_EQ(1.5f - 2.375f + 2.f + 1.25f + 7.f - 0.5f, b(0).as<float>());
  EXPECT_EQ(-2.25f + 1.f + 7.f + 2.125f + 0.25f, b(1).as<float>());
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceBroadcastReduce)
{
  // Start with a float32 reduction callable
  nd::callable reduction_kernel =
      kernels::make_builtin_sum_reduction_callable(float32_type_id);

  // Lift it to a three-dimensional strided float32 reduction callable
  bool reduction_dimflags[3] = {true, false, true};
  nd::callable af = nd::functional::reduction(
      reduction_kernel, ndt::type("Fixed * Fixed * Fixed * float32"),
      nd::callable(), false, 3, reduction_dimflags, true, true, false,
      nd::array());

  // Set up some data for the test reduction
  nd::array a = parse_json("2 * 3 * 2 * float32", "[[[1.5, -2.375], [2, 1.25], "
                                                  "[7, -0.5]], [[-2.25, 1], "
                                                  "[7, 0], [2.125, 0.25]]]");
  a = a(irange(), irange(), irange());
  EXPECT_TYPE_MATCH(af.get_type()->get_pos_type(0), a.get_type());
  nd::array b = nd::empty(3, "float32");
  EXPECT_TYPE_MATCH(af.get_type()->get_return_type(), b.get_type());

  // Instantiate the lifted ckernel
  ckernel_builder<kernel_request_host> ckb;
  const ndt::type src_tp[1] = {a.get_type()};
  const char *src_arrmeta[1] = {a.get_arrmeta()};
  af.get()->instantiate(af.get()->static_data, 0, NULL, &ckb, 0, b.get_type(),
                        b.get_arrmeta(), 1, src_tp, src_arrmeta,
                        kernel_request_single, &eval::default_eval_context,
                        nd::array(), std::map<std::string, ndt::type>());
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();

  // Call it on the data
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(ckb.get(), b.get_readwrite_originptr(), &src);
  ASSERT_EQ(3, b.get_shape()[0]);
  EXPECT_EQ(1.5f - 2.375f - 2.25f + 1.f, b(0).as<float>());
  EXPECT_EQ(2.f + 1.25f + 7.f, b(1).as<float>());
  EXPECT_EQ(7.f - 0.5f + 2.125f + 0.25f, b(2).as<float>());
}
