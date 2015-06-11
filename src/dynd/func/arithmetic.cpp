//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

struct nd::plus nd::plus;

nd::array nd::operator+(const array &a0) { return plus(a0); }

struct nd::minus nd::minus;

nd::array nd::operator-(const array &a0) { return minus(a0); }

struct nd::add nd::add;

nd::array nd::operator+(const array &a0, const array &a1)
{
  return add(a0, a1);
}

struct nd::subtract nd::subtract;

nd::array nd::operator-(const array &a0, const array &a1)
{
  return subtract(a0, a1);
}

struct nd::multiply nd::multiply;

nd::array nd::operator*(const array &a0, const array &a1)
{
  return multiply(a0, a1);
}

struct nd::divide nd::divide;

nd::array nd::operator/(const array &a0, const array &a1)
{
  return divide(a0, a1);
}