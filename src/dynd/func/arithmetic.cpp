//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/multidispatch.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/kernels/arithmetic.hpp>

using namespace dynd;

nd::arrfunc nd::plus::children[DYND_TYPE_ID_MAX + 1];
nd::arrfunc nd::plus::default_child;

nd::arrfunc nd::plus::make()
{
  arrfunc::make_all<plus_kernel, arithmetic_type_ids>(children);

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any) -> Any"), DYND_TYPE_ID_MAX + 1, children, default_child,
      false));
}

struct nd::plus nd::plus;

nd::array nd::operator+(const nd::array &a0) { return nd::plus(a0); }

nd::arrfunc nd::minus::children[DYND_TYPE_ID_MAX + 1];
nd::arrfunc nd::minus::default_child;

nd::arrfunc nd::minus::make()
{
  arrfunc::make_all<minus_kernel, arithmetic_type_ids>(children);

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any) -> Any"), DYND_TYPE_ID_MAX + 1, children, default_child,
      false));
}

struct nd::minus nd::minus;

nd::array nd::operator-(const nd::array &a0) { return nd::minus(a0); }

nd::arrfunc nd::add::children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];
nd::arrfunc nd::add::default_child;

nd::arrfunc nd::add::make()
{
  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"),
      arrfunc::make_all<add_ck, arithmetic_type_ids, arithmetic_type_ids>()));
}

struct nd::add nd::add;

nd::array nd::operator+(const nd::array &a0, const nd::array &a1)
{
  return nd::add(a0, a1);
}

nd::arrfunc nd::subtract::make()
{
  std::vector<arrfunc> children = arrfunc::make_all<
      subtract_ck, arithmetic_type_ids, arithmetic_type_ids>();

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
  std::vector<arrfunc> children = arrfunc::make_all<
      multiply_ck, arithmetic_type_ids, arithmetic_type_ids>();

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
      arrfunc::make_all<divide_ck, arithmetic_type_ids, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"), children));
}

struct nd::divide nd::divide;

nd::array nd::operator/(const nd::array &a0, const nd::array &a1)
{
  return nd::divide(a0, a1);
}