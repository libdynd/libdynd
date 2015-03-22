//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/kernels/arithmetic.hpp>

using namespace dynd;

nd::arrfunc nd::plus::make()
{
  return functional::elwise(as_arrfunc<multidispatch_plus_ck>());
}

struct nd::plus nd::plus;

nd::array nd::operator+(const nd::array &a0) { return nd::plus(a0); }

nd::arrfunc nd::minus::make()
{
  return functional::elwise(as_arrfunc<multidispatch_minus_ck>());
}

struct nd::minus nd::minus;

nd::array nd::operator-(const nd::array &a0) { return nd::minus(a0); }

nd::arrfunc nd::add::make()
{
  return functional::elwise(as_arrfunc<virtual_add_ck>());
}

struct nd::add nd::add;

nd::array nd::operator+(const nd::array &a0, const nd::array &a1)
{
  return nd::add(a0, a1);
}

nd::arrfunc nd::subtract::make()
{
  return functional::elwise(as_arrfunc<virtual_subtract_ck>());
}

struct nd::subtract nd::subtract;

nd::array nd::operator-(const nd::array &a0, const nd::array &a1)
{
  return nd::subtract(a0, a1);
}

nd::arrfunc nd::multiply::make()
{
  return functional::elwise(as_arrfunc<virtual_multiply_ck>());
}

struct nd::multiply nd::multiply;

nd::array nd::operator*(const nd::array &a0, const nd::array &a1)
{
  return nd::multiply(a0, a1);
}

nd::arrfunc nd::divide::make()
{
  return functional::elwise(as_arrfunc<virtual_divide_ck>());
}

struct nd::divide nd::divide;

nd::array nd::operator/(const nd::array &a0, const nd::array &a1)
{
  return nd::divide(a0, a1);
}