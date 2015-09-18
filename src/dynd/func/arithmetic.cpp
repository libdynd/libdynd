//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

DYND_API struct nd::plus nd::plus;

nd::array nd::operator+(const array &a0)
{
  return plus(a0);
}

DYND_API struct nd::minus nd::minus;

nd::array nd::operator-(const array &a0)
{
  return minus(a0);
}

DYND_API struct nd::add nd::add;

nd::array nd::operator+(const array &a0, const array &a1)
{
  return add(a0, a1);
}

DYND_API struct nd::subtract nd::subtract;

nd::array nd::operator-(const array &a0, const array &a1)
{
  return subtract(a0, a1);
}

DYND_API struct nd::multiply nd::multiply;

nd::array nd::operator*(const array &a0, const array &a1)
{
  return multiply(a0, a1);
}

DYND_API struct nd::divide nd::divide;

nd::array nd::operator/(const array &a0, const array &a1)
{
  return divide(a0, a1);
}

/*
struct nd::compound_add nd::compound_add;

nd::array &nd::array::operator+=(const array &rhs)
{
  compound_add(rhs, kwds("dst", *this));
  return *this;
}
*/

DYND_API struct nd::compound_add nd::compound_add;

DYND_API struct nd::compound_div nd::compound_div;

nd::array &nd::array::operator/=(const array &rhs)
{
  compound_div(rhs, kwds("dst", *this));
  return *this;
}
