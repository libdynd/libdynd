//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/take.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(Callable, SingleStridedConstructor)
{
  nd::callable f(ndt::type("(int32) -> int32"), [](ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src) {
    *reinterpret_cast<int32 *>(dst) = *reinterpret_cast<int32 *>(src[0]) + 5;
  }, 0);

  EXPECT_ARRAY_EQ(8, f(3));
}

TEST(Callable, Assignment)
{
  // Create an callable for converting string to int
  nd::callable af =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::make_type<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::fixed_string_type::make(16), af.get_type()->get_pos_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  nd::kernel_builder ckb;
  af.get()->instantiate(af.get()->static_data(), NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_single, 0, NULL, std::map<std::string, ndt::type>());
  int int_out = 0;
  char str_in[16] = "3251";
  const char *str_in_ptr = str_in;
  kernel_single_t usngo = ckb.get()->get_function<kernel_single_t>();
  usngo(ckb.get(), reinterpret_cast<char *>(&int_out), const_cast<char **>(&str_in_ptr));
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(af.get()->static_data(), NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_strided, 0, NULL, std::map<std::string, ndt::type>());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  const char *strs_in_ptr = strs_in[0];
  kernel_strided_t ustro = ckb.get()->get_function<kernel_strided_t>();
  intptr_t strs_in_stride = sizeof(strs_in[0]);
  ustro(ckb.get(), reinterpret_cast<char *>(&ints_out), sizeof(int), const_cast<char **>(&strs_in_ptr), &strs_in_stride,
        3);
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

TEST(Callable, Construction)
{
  nd::callable f0([](int x, double y) { return 2.0 * x + y; });
  EXPECT_ARRAY_EQ(4.5, f0(1, 2.5));

  nd::callable f1([](int x, double y) { return 2.0 * x + y; }, "y");
  EXPECT_ARRAY_EQ(4.5, f1({1}, {{"y", 2.5}}));

  nd::callable f2([](int x, int y) { return x - y; });
  EXPECT_ARRAY_EQ(-4, f2(3, 7).as<int>());

  nd::callable f3([](int x, int y) { return x - y; }, "y");
  EXPECT_ARRAY_EQ(-4, f3({3}, {{"y", 7}}).as<int>());
}

TEST(Callable, CallOperator)
{
  nd::callable f([](int x, double y) { return 2.0 * x + y; });
  // Calling with positional arguments
  EXPECT_EQ(4.5, f(1, 2.5).as<double>());
  EXPECT_EQ(7.5, f(2, 3.5).as<double>());
  // Wrong number of positional argumetns
  EXPECT_THROW(f(2), invalid_argument);
  EXPECT_THROW(f(2, 3.5, 7), invalid_argument);
  // Extra keyword argument
  EXPECT_THROW(f({2, 3.5}, {{"x", 10}}), invalid_argument);

  f = nd::functional::apply([](int x, double y) { return 2.0 * x + y; }, "y");
  // Calling with positional and keyword arguments
  EXPECT_EQ(4.5, f({1}, {{"y", 2.5}}).as<double>());
  EXPECT_EQ(7.5, f({2}, {{"y", 3.5}}).as<double>());
  // Calling with positional arguments
  EXPECT_EQ(7.5, f(2, 3.5).as<double>());
  // Wrong number of positional/keyword arguments
  EXPECT_THROW(f(2), invalid_argument);
  EXPECT_THROW(f(2, 3.5, 7), invalid_argument);
  // Extra/wrong keyword argument
  EXPECT_THROW(f({2}, {{"x", 3.5}}), invalid_argument);
  EXPECT_THROW(f({2}, {{"x", 10}, {"y", 20}}), invalid_argument);
  EXPECT_THROW(f({2, 3.5}, {{"x", 10}, {"y", 20}}), invalid_argument);

  f = nd::functional::apply([]() { return 10; });
  // Calling with no arguments
  EXPECT_EQ(10, f().as<int>());
  // Calling with no arguments
  EXPECT_EQ(10, f({}, {}).as<int>());
  // Wrong number of positional/keyword arguments
  EXPECT_THROW(f(2), invalid_argument);
  EXPECT_THROW(f({}, {{"y", 3.5}}), invalid_argument);
}

TEST(Callable, DynamicCall)
{
  nd::array args[3] = {7, 2.5, 5};
  pair<const char *, nd::array> kwds[3] = {{"x", args[0]}, {"y", args[1]}, {"z", args[2]}};

  nd::callable af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; });
  EXPECT_EQ(26.5, af.call(3, args, 0, nullptr).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "z");
  EXPECT_EQ(26.5, af.call(2, args, 1, kwds + 2).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "y", "z");
  EXPECT_EQ(26.5, af.call(1, args, 2, kwds + 1).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "x", "y", "z");
  EXPECT_EQ(26.5, af.call(0, nullptr, 3, kwds).as<double>());
}

TEST(Callable, DecomposedDynamicCall)
{
  nd::callable af;

  ndt::type ret_tp;
  nd::array values[3] = {7, 2.5, 5};
  ndt::type types[3] = {values[0].get_type(), values[1].get_type(), values[2].get_type()};
  const char *const arrmetas[3] = {values[0].get()->metadata(), values[1].get()->metadata(),
                                   values[2].get()->metadata()};
  char *const datas[3] = {values[0].get()->data, values[1].get()->data, values[2].get()->data};
  //  const char *names[3] = {"x", "y", "z"};

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; });
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5, af->call(ret_tp, 3, types, arrmetas, datas, 0, NULL, map<std::string, ndt::type>()).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "z");
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5,
            af->call(ret_tp, 2, types, arrmetas, datas, 1, values + 2, map<std::string, ndt::type>()).as<double>());

  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "y", "z");
  ret_tp = af.get_ret_type();
  EXPECT_EQ(26.5,
            af->call(ret_tp, 1, types, arrmetas, datas, 2, values + 1, map<std::string, ndt::type>()).as<double>());

  //  af = nd::functional::apply([](int x, double y, int z) { return 2 * x - y + 3 * z; }, "x", "y", "z");
  //  ret_tp = af.get_ret_type();
  //  EXPECT_EQ(26.5, af->call(ret_tp, 0, NULL, NULL, NULL, 3, values, map<string, ndt::type>()).as<double>());
}

TEST(Callable, KeywordParsing)
{
  nd::callable af0 = nd::functional::apply([](int x, int y) { return x + y; }, "y");
  EXPECT_EQ(5, af0({1}, {{"y", 4}}).as<int>());
  EXPECT_THROW(af0({1}, {{"z", 4}}).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0({1}, {{"Y", 4}}).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0({1}, {{"y", 2.5}}).as<int>(), std::invalid_argument);
  EXPECT_THROW(af0({1}, {{"y", 4}, {"y", 2.5}}).as<int>(), std::invalid_argument);
}

TEST(Callable, Assignment_CallInterface)
{
  // Test with the unary operation prototype
  nd::callable af =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::make_type<ndt::string_type>(), assign_error_default);

  // Call it through the call() interface
  nd::array b = af("12345678");
  EXPECT_EQ(ndt::make_type<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);

  // Test with the expr operation prototype
  af = make_callable_from_assignment(ndt::make_type<int>(), ndt::make_type<ndt::string_type>(), assign_error_default);

  // Call it through the call() interface
  b = af("12345678");
  EXPECT_EQ(ndt::make_type<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);
}

TEST(Callable, AssignmentAsExpr)
{
  // Create an callable for converting string to int
  nd::callable af =
      make_callable_from_assignment(ndt::make_type<int>(), ndt::fixed_string_type::make(16), assign_error_default);
  // Validate that its types, etc are set right
  ASSERT_EQ(1, af.get_type()->get_narg());
  ASSERT_EQ(ndt::make_type<int>(), af.get_type()->get_return_type());
  ASSERT_EQ(ndt::fixed_string_type::make(16), af.get_type()->get_pos_type(0));

  const char *src_arrmeta[1] = {NULL};

  // Instantiate a single ckernel
  nd::kernel_builder ckb;
  af.get()->instantiate(af.get()->static_data(), NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_single, 0, NULL, std::map<std::string, ndt::type>());
  int int_out = 0;
  char str_in[16] = "3251";
  char *str_in_ptr = str_in;
  kernel_single_t usngo = ckb.get()->get_function<kernel_single_t>();
  usngo(ckb.get(), reinterpret_cast<char *>(&int_out), &str_in_ptr);
  EXPECT_EQ(3251, int_out);

  // Instantiate a strided ckernel
  ckb.reset();
  af.get()->instantiate(af.get()->static_data(), NULL, &ckb, 0, af.get_type()->get_return_type(), NULL,
                        af.get_type()->get_npos(), af.get_type()->get_pos_types_raw(), src_arrmeta,
                        kernel_request_strided, 0, NULL, std::map<std::string, ndt::type>());
  int ints_out[3] = {0, 0, 0};
  char strs_in[3][16] = {"123", "4567", "891029"};
  char *strs_in_ptr = strs_in[0];
  intptr_t strs_in_stride = 16;
  kernel_strided_t ustro = ckb.get()->get_function<kernel_strided_t>();
  ustro(ckb.get(), reinterpret_cast<char *>(&ints_out), sizeof(int), &strs_in_ptr, &strs_in_stride, 3);
  EXPECT_EQ(123, ints_out[0]);
  EXPECT_EQ(4567, ints_out[1]);
  EXPECT_EQ(891029, ints_out[2]);
}

/*
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/SourceMgr.h>
*/

/*
TEST(Callable, LLVM)
{
  nd::callable f = nd::functional::apply([](int x, int y) { return x + y; });
  // nd::callable f = nd::add::children[int32_type_id][int32_type_id];

  std::cout << f->ir << std::endl;

//  llvm::SMDiagnostic error;
//  llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(f->ir))->getMemBufferRef(), error,
llvm::getGlobalContext());
}
*/
