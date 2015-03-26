//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/multidispatch.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/kernels/arithmetic.hpp>

using namespace dynd;

nd::arrfunc nd::plus::make()
{
  std::vector<arrfunc> children = as_arrfuncs<plus_ck, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any) -> Any"), children));
}

struct nd::plus nd::plus;

nd::array nd::operator+(const nd::array &a0) { return nd::plus(a0); }

nd::arrfunc nd::minus::make()
{
  std::vector<arrfunc> children = as_arrfuncs<minus_ck, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any) -> Any"), children));
}

struct nd::minus nd::minus;

nd::array nd::operator-(const nd::array &a0) { return nd::minus(a0); }

nd::arrfunc nd::add::make()
{
  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"),
      as_arrfuncs<add_ck, arithmetic_type_ids, arithmetic_type_ids>()));
}

struct nd::add nd::add;

nd::array nd::operator+(const nd::array &a0, const nd::array &a1)
{
  return nd::add(a0, a1);
}

nd::arrfunc nd::subtract::make()
{
  std::vector<arrfunc> children =
      as_arrfuncs<subtract_ck, arithmetic_type_ids, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"), children));
}

struct nd::subtract nd::subtract;

nd::array nd::operator-(const nd::array &a0, const nd::array &a1)
{
  return nd::subtract(a0, a1);
}

nd::arrfunc nd::multiply::make()
{
  std::vector<arrfunc> children =
      as_arrfuncs<multiply_ck, arithmetic_type_ids, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"), children));
}

struct nd::multiply nd::multiply;

nd::array nd::operator*(const nd::array &a0, const nd::array &a1)
{
  return nd::multiply(a0, a1);
}

nd::arrfunc nd::divide::make()
{
  std::vector<arrfunc> children =
      as_arrfuncs<divide_ck, arithmetic_type_ids, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"), children));
}

struct nd::divide nd::divide;

nd::array nd::operator/(const nd::array &a0, const nd::array &a1)
{
  return nd::divide(a0, a1);
}