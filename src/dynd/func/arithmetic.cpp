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

using namespace std;
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
  const arrfunc self = functional::call(ndt::type("(Any) -> Any"), nd::minus);

  for (const pair<const type_id_t, arrfunc> &pair :
       arrfunc::make_all<minus_kernel, arithmetic_type_ids>()) {
    children[pair.first] = pair.second;
  }

  for (type_id_t i0 : dim_type_ids::vals()) {
    const ndt::type child_tp =
        ndt::arrfunc_type::make({ndt::type(i0)}, ndt::type("Any"));
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

ndt::type make_kind(type_id_t type_id)
{
  if (type_id == fixed_dim_type_id) {
    return ndt::type("Fixed * Any");
  } else if (type_id == var_dim_type_id) {
    return ndt::type("var * Any");
  }

  return ndt::type(type_id);
}

// kernel_type, type_id0, type_id1, 

nd::arrfunc nd::add::make()
{
  arrfunc self = functional::call(ndt::type("(Any, Any) -> Any"), nd::add);

  for (const pair<pair<type_id_t, type_id_t>, arrfunc> &pair :
       arrfunc::make_all<add_ck, arithmetic_type_ids, arithmetic_type_ids>()) {
    children[pair.first.first][pair.first.second] = pair.second;
  }

  for (type_id_t i0 : arithmetic_type_ids::vals()) {
    for (type_id_t i1 : dim_type_ids::vals()) {
      children[i0][i1] = functional::elwise(
          ndt::arrfunc_type::make({ndt::type(i0), ndt::type(i1)},
                                  ndt::type("Any")),
          self);
    }
  }

  for (type_id_t i0 : dim_type_ids::vals()) {
    for (type_id_t i1 : arithmetic_type_ids::vals()) {
      ndt::type pos[2] = {make_kind(i0), make_kind(i1)};

      children[i0][i1] =
          functional::elwise(make_arrfunc(2, pos, ndt::type("Any")), self);
    }
  }

  for (type_id_t i0 : dim_type_ids::vals()) {
    for (type_id_t i1 : dim_type_ids::vals()) {
      ndt::type pos[2] = {make_kind(i0), make_kind(i1)};

      children[i0][i1] =
          functional::elwise(make_arrfunc(2, pos, ndt::type("Any")), self);
    }
  }

  return functional::multidispatch_by_type_id(self.get_array_type(), children,
                                              default_child);
}

struct nd::add nd::add;

nd::array nd::operator+(const nd::array &a0, const nd::array &a1)
{
  return nd::add(a0, a1);
}

nd::arrfunc nd::subtract::make()
{
  std::vector<arrfunc> children = arrfunc::old_make_all<
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
  std::vector<arrfunc> children = arrfunc::old_make_all<
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
  std::vector<arrfunc> children = arrfunc::old_make_all<
      divide_ck, arithmetic_type_ids, arithmetic_type_ids>();

  return functional::elwise(functional::multidispatch_by_type_id(
      ndt::type("(Any, Any) -> Any"), children));
}

struct nd::divide nd::divide;

nd::array nd::operator/(const nd::array &a0, const nd::array &a1)
{
  return nd::divide(a0, a1);
}