//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply_arrfunc.hpp>

using namespace std;
using namespace dynd;

int func0(int x, int y)
{
	return 2 * (x - y);
}

double func1(double x, int y)
{
  return x + 2.6 * y;
}

float func2(const float (&x)[3])
{
    return x[0] + x[1] + x[2];
}

unsigned int func3()
{
  return 12U;
}

double func4(const double (&x)[3], const double (&y)[3])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

long func5(const long (&x)[2][3])
{
    return x[0][0] + x[0][1] + x[1][2];
}

TEST(Apply, Function)
{
	nd::arrfunc af;

	af = nd::make_apply_arrfunc<decltype(&func0), &func0>();
	EXPECT_EQ(4, af(5, 3).as<int>());
	af = nd::make_apply_arrfunc(func0);
	EXPECT_EQ(4, af(5, 3).as<int>());

	af = nd::make_apply_arrfunc<decltype(&func1), &func1>();
	EXPECT_EQ(53.15, af(3.75, 19).as<double>());
	af = nd::make_apply_arrfunc(func1);
	EXPECT_EQ(53.15, af(3.75, 19).as<double>());

  af = nd::make_apply_arrfunc<decltype(&func2), &func2>();
  EXPECT_FLOAT_EQ(13.2, af(nd::array({3.9f, -7.0f, 16.3f}).view(ndt::make_type<float[3]>())).as<float>());
  af = nd::make_apply_arrfunc(func2);
  EXPECT_FLOAT_EQ(13.2, af(nd::array({3.9f, -7.0f, 16.3f}).view(ndt::make_type<float[3]>())).as<float>());  

  af = nd::make_apply_arrfunc<decltype(&func3), &func3>();
  EXPECT_EQ(12U, af().as<unsigned int>());
  af = nd::make_apply_arrfunc(func3);
  EXPECT_EQ(12U, af().as<unsigned int>());

  af = nd::make_apply_arrfunc<decltype(&func4), &func4>();
  EXPECT_DOUBLE_EQ(166.765, af(nd::array({9.14, -2.7, 15.32}).view(ndt::make_type<double[3]>()),
    nd::array({0.0, 0.65, 11.0}).view(ndt::make_type<double[3]>())).as<double>());
  af = nd::make_apply_arrfunc(func4);
  EXPECT_DOUBLE_EQ(166.765, af(nd::array({9.14, -2.7, 15.32}).view(ndt::make_type<double[3]>()),
    nd::array({0.0, 0.65, 11.0}).view(ndt::make_type<double[3]>())).as<double>());

  af = nd::make_apply_arrfunc<decltype(&func5), &func5>();
  EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L, -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
  af = nd::make_apply_arrfunc(&func5);
  EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L, -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
}

TEST(Apply, FunctionWithKeywords)
{
  nd::arrfunc af;

  af = nd::make_apply_arrfunc<decltype(&func0), &func0>("y");
  EXPECT_EQ(4, af(5, kwds("y", 3)).as<int>());
  af = nd::make_apply_arrfunc(func0, "y");

  af = nd::make_apply_arrfunc<decltype(&func1), &func1>("y");
  EXPECT_EQ(53.15, af(3.75, kwds("y", 19)).as<double>());
}

struct callable0
{
  int operator ()(int x, int y) const
  {
	 return func0(x, y);
  }
};

struct callable1
{
  double operator ()(double x, int y) const
  {
	return func1(x, y);
  }
};

struct callable2
{
  int m_y;

  callable2(int y) : m_y(y) {
  }

  int operator ()(int x) const
  {
   return func0(x, m_y);
  }
};

TEST(Apply, Callable)
{
  nd::arrfunc af;

  af = nd::make_apply_arrfunc<callable0>();
  EXPECT_EQ(4, af(5, 3).as<int>());
  af = nd::make_apply_arrfunc(callable0());
  EXPECT_EQ(4, af(5, 3).as<int>());

  af = nd::make_apply_arrfunc<callable1>();
  EXPECT_EQ(53.15, af(3.75, 19).as<double>());
  af = nd::make_apply_arrfunc(callable1());
  EXPECT_EQ(53.15, af(3.75, 19).as<double>());
}

TEST(Apply, CallableWithKeywords)
{
  nd::arrfunc af;

  af = nd::make_apply_arrfunc<callable2, int>("y");
  EXPECT_EQ(4, af(5, kwds("y", 3)).as<int>());
}

TEST(Apply, MemberFunction)
{

}