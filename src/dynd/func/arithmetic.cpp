//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/multidispatch.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/call.hpp>
#include <dynd/kernels/arithmetic.hpp>
#include <array>

using namespace dynd;

namespace {

template <template <type_id_t...> class T, type_id_t...>
struct arithmetic_arrfunc_factory;

template <template <type_id_t...> class T, type_id_t I0>
struct arithmetic_arrfunc_factory<T, I0> {
  static nd::arrfunc make() { return nd::arrfunc::make<T<I0>>(0); }
};

} // anonymous namespace

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

namespace {

template <type_id_t I0>
using minus_arrfunc_factory = arithmetic_arrfunc_factory<nd::minus_kernel, I0>;

} // anonymous namespace

nd::arrfunc nd::minus::children[DYND_TYPE_ID_MAX + 1];
nd::arrfunc nd::minus::default_child;

nd::arrfunc nd::minus::make()
{
  arrfunc self = functional::call(ndt::type("(Any) -> Any"), nd::minus);

  // ...
  for (const std::pair<const type_id_t, arrfunc> &pair :
       arrfunc::make_all<minus_kernel, arithmetic_type_ids>()) {
    children[pair.first] = pair.second;
  }

  // ...
  for (type_id_t i0 : dim_type_ids::vals()) {
    ndt::type child_tp = ndt::type("(Fixed * Any) -> Fixed * Any");
    children[i0] = functional::elwise(child_tp, self);
  }

  return functional::multidispatch_by_type_id(self.get_array_type(),
                                              DYND_TYPE_ID_MAX + 1, children,
                                              default_child, false);
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