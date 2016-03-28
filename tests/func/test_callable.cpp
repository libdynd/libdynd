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
#include <dynd/callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/index.hpp>
#include <dynd/array.hpp>

#include <dynd/arithmetic.hpp>
#include <dynd/callables/call_stack.hpp>

using namespace std;
using namespace dynd;

/*
TEST(Callable, Resolve)
{
  ndt::type dst_tp = ndt::type("Any");
  size_t nsrc = 2;
  ndt::type src_tp[2] = {ndt::type("10 * int32"), ndt::type("10 * int32")};

  nd::call_stack s;
  s.push_back(nd::add, dst_tp, nsrc, src_tp);
  nd::add->resolve(s, 0, nullptr, std::map<std::string, ndt::type>());

  for (auto it : s) {
    std::cout << "func = " << it.func << std::endl;
    std::cout << "dst_tp = " << it.dst_tp << std::endl;
    std::cout << "nsrc = " << it.nsrc << std::endl;
    for (size_t i = 0; i < it.nsrc; ++i) {
      std::cout << "src_tp[" << i << "] = " << it.src_tp[i] << std::endl;
    }
    std::cout << std::endl;
  }
  std::exit(-1);

  //  s.push_back(callable(this, true), dst_tp, nsrc, src_tp);
  //  resolve(s, nkwd, kwds, tp_vars);
}
*/

/*
TEST(Callable, SingleStridedConstructor)
{
  nd::callable f(ndt::type("(int32) -> int32"), [](nd::kernel_prefix *DYND_UNUSED(self), char *dst, char *const *src) {
    *reinterpret_cast<int32 *>(dst) = *reinterpret_cast<int32 *>(src[0]) + 5;
  }, 0);

  EXPECT_ARRAY_EQ(8, f(3));
}
*/

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

TEST(Callable, FromKernel)
{
  struct kernel : nd::base_strided_kernel<kernel, 1> {
    void single(char *res, char *const *args) { *reinterpret_cast<int *>(res) = *reinterpret_cast<int *>(args[0]) + 7; }
  };

  nd::callable f = nd::make_callable<kernel>(ndt::make_type<int(int)>());
  EXPECT_EQ(ndt::make_type<int(int)>(), f.get_array_type());
  EXPECT_ARRAY_EQ(9, f(2));
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
  nd::callable af = nd::assign.specialize(ndt::make_type<int>(), {ndt::make_type<ndt::string_type>()});

  // Call it through the call() interface
  nd::array b = af("12345678");
  EXPECT_EQ(ndt::make_type<int>(), b.get_type());
  EXPECT_EQ(12345678, b.as<int>());

  // Call it with some incompatible arguments
  EXPECT_THROW(af(12345), invalid_argument);
  EXPECT_THROW(af(false), invalid_argument);
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
  // nd::callable f = nd::add::children[int32_id][int32_id];

  std::cout << f->ir << std::endl;

//  llvm::SMDiagnostic error;
//  llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(f->ir))->getMemBufferRef(), error,
llvm::getGlobalContext());
}
*/
