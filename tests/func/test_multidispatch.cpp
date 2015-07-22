//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

int func0(int, float, double) { return 0; }
int func1(int, double, double) { return 1; }
int func2(int, float, float) { return 2; }
int func3(float, int, int) { return 3; }
int func4(int16_t, float, double) { return 4; }
int func5(int16_t, float, float) { return 5; }

double manip0(int x, int y) { return x + y; };
double manip1(double x, double y) { return x - y; };

// TODO: Reenable tests involving float16

TEST(MultiDispatchCallable, Ambiguous)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));

  EXPECT_THROW(nd::functional::old_multidispatch(funcs.size(), &funcs[0]),
               invalid_argument);

  funcs.push_back(nd::functional::apply(&func5));
}

TEST(MultiDispatchCallable, ExactSignatures)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));
  funcs.push_back(nd::functional::apply(&func5));

  nd::callable af = nd::functional::old_multidispatch(funcs.size(), &funcs[0]);

  EXPECT_EQ(0, af(1, 1.f, 1.0).as<int>());
  EXPECT_EQ(1, af(1, 1.0, 1.0).as<int>());
  EXPECT_EQ(2, af(1, 1.f, 1.f).as<int>());
  EXPECT_EQ(3, af(1.f, 1, 1).as<int>());
  EXPECT_EQ(4, af((int16_t)1, 1.f, 1.0).as<int>());
  EXPECT_EQ(5, af((int16_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchCallable, PromoteToSignature)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));
  funcs.push_back(nd::functional::apply(&func5));

  nd::callable af = nd::functional::old_multidispatch(funcs.size(), &funcs[0]);

  //  EXPECT_EQ(0, af(1, float16(1.f), 1.0).as<int>());
  EXPECT_EQ(1, af(1, 1.0, 1.f).as<int>());
  //  EXPECT_EQ(2, af(1, 1.f, float16(1.f)).as<int>());
  EXPECT_EQ(3, af(1.f, 1, (int16_t)1).as<int>());
  EXPECT_EQ(4, af((int8_t)1, 1.f, 1.0).as<int>());
  EXPECT_EQ(5, af((int8_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchCallable, Values)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&manip0));
  funcs.push_back(nd::functional::apply(&manip1));
  nd::callable af = nd::functional::elwise(
      nd::functional::old_multidispatch(funcs.size(), &funcs[0]));
  nd::array a, b, c;

  // Exactly match (int, int) -> real
  a = parse_json("3 * int", "[1, 3, 5]");
  b = parse_json("3 * int", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);

  // Exactly match (real, real) -> real
  a = parse_json("3 * real", "[1, 3, 5]");
  b = parse_json("3 * real", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[-1, -2, 4]", c);

  // Promote to (int, int) -> real
  a = parse_json("3 * int16", "[1, 3, 5]");
  b = parse_json("3 * int8", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);

  // Promote to (real, real) -> real
  /*
    a = parse_json("3 * int16", "[1, 3, 5]");
    b = parse_json("3 * float16", "[2, 5, 1]");
    c = af(a, b);
    EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
    EXPECT_JSON_EQ_ARR("[-1, -2, 4]", c);
  */
}

/**
TODO: This test broken when the order of resolve_option_values and
      resolve_dst_type changed. It should be fixed when we sort out
multidispatch.

TEST(MultiDispatchCallable, Dims)
{
  vector<nd::callable> funcs;
  // Instead of making a multidispatch callable, then lifting it,
  // we lift multiple callables, then make a multidispatch callable from them.
  funcs.push_back(nd::functional::elwise(nd::functional::apply(&manip0)));
  funcs.push_back(nd::functional::elwise(nd::functional::apply(&manip1)));
  nd::callable af = nd::functional::multidispatch(funcs.size(), &funcs[0]);
  nd::array a, b, c;

  // Exactly match (int, int) -> real
  a = parse_json("3 * int", "[1, 3, 5]");
  b = parse_json("3 * int", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);
}
*/

template <typename T>
T tester(T x, T y)
{
  return x + y;
}

/*
TEST(MultidispatchCallable, Untitled)
{
  nd::callable af = nd::functional::multidispatch<2>(
      ndt::type("(Any, Any) -> Any"),
      {nd::functional::apply(&tester<int>),
       nd::functional::apply(&tester<double>),
       nd::functional::apply(&tester<int>)});

  std::cout << af << std::endl;
  std::cout << af(2.0, 3.5) << std::endl;
  std::cout << af(1, 2) << std::endl;

  std::exit(-1);
}
*/

/*

TEST(Multidispatch, CArray)
{
  nd::callable children[DYND_TYPE_ID_MAX];
  children[float64_type_id] = nd::functional::apply(&func<double>);

  nd::callable func = nd::functional::multidispatch(ndt::type("(Any) -> Any"),
                                                   children, nd::callable());
  func(3.0);
}

TEST(Multidispatch, Vector)
{
  vector<nd::callabe> children(DYND_TYPE_ID_MAX);
  children[float64_type_id] = nd::functional::apply(&func<double>);

  nd::callable func = nd::functional::multidispatch<1>(ndt::type("(Any) ->
Any"),
                                                      std::move(children),
nd::callable());
  children.clear();

  std::cout << func(3.0) << std::endl;
  std::cout << func << std::endl;

  std::exit(-1);
}
*/

template <typename A0, typename A1>
typename std::common_type<A0, A1>::type func(A0 x, A1 y)
{
  return x + y;
}

TEST(Multidispatch, Map)
{
  //  std::array<type_id_t, 2> key;

  map<array<type_id_t, 2>, nd::callable> children;
  children[{{float64_type_id, float32_type_id}}] =
      nd::functional::apply(&func<double, float>);
  children[{{int32_type_id, int32_type_id}}] =
      nd::functional::apply(&func<int32, int32>);

  nd::callable func = nd::functional::multidispatch<2>(
      ndt::type("(Any, Any) -> Any"), children, nd::callable());

  //  std::cout << "made" << std::endl;

  EXPECT_EQ(5.5, func(2.0, 3.5f));
  EXPECT_EQ(3, func(1, 2));
}

/*
TEST(Multidispatch, Map)
{
  std::cout << has_key_type<std::map<int, int>>::value << std::endl;
  std::cout << has_key_type<std::vector<int>>::value << std::endl;
  std::exit(-1);
}
*/

#ifdef DYND_CUDA

/*
template <typename T>
struct callable0 {
  DYND_CUDA_HOST_DEVICE T operator()(T x, T y) const { return x + y; }
};

TEST(MultidispatchFunc, CudaHostDevice)
{
  nd::callable af = nd::functional::multidispatch(
      ndt::type("(M[R], M[R]) -> M[R]"),
      {nd::functional::apply<callable0<int>>(),
       nd::functional::apply<kernel_request_cuda_device, callable0<int>>()});

  std::cout << af(1, 2) << std::endl;
  std::cout << af(nd::array(1).to_cuda_device(), nd::array(2).to_cuda_device())
            << std::endl;

  nd::callable af1 = nd::functional::elwise(af);
  std::cout << af1 << std::endl;

  nd::array a = nd::random::uniform(kwds("dst_tp", ndt::type("10 * int32")));
  nd::array b = nd::random::uniform(kwds("dst_tp", ndt::type("10 * int32")));

  std::cout << af1(a, b) << std::endl;
  std::cout << af1(a.to_cuda_device(), b.to_cuda_device()) << std::endl;

//  std::exit(-1);
}
*/

#endif