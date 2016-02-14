//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/comparison.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/compare_kernels.hpp>

using namespace std;
using namespace dynd;

template <typename SelfType, template <type_id_t...> class KernelType>
map<array<type_id_t, 2>, nd::callable> nd::comparison_operator<SelfType, KernelType>::make_children()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id, uint64_id,
                           float32_id, float64_id> numeric_ids;

  auto children = callable::make_all<KernelType, numeric_ids, numeric_ids>();

  callable self = functional::call<SelfType>(ndt::type("(Any, Any) -> Any"));

  for (type_id_t i0 : i2a<numeric_ids>()) {
    for (type_id_t i1 : i2a<dim_ids>()) {
      const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
      children[{{i0, i1}}] = functional::elwise(child_tp, self);
    }
  }

  for (type_id_t i : i2a<numeric_ids>()) {
    children[{{option_id, i}}] = functional::forward_na<0>(self);
    children[{{i, option_id}}] = functional::forward_na<1>(self);
  }
  children[{{option_id, option_id}}] = callable::make<option_comparison_kernel<SelfType, true, true>>();

  for (type_id_t dim_tp_id : i2a<dim_ids>()) {
    children[{{dim_tp_id, option_id}}] = functional::elwise(self);
    children[{{option_id, dim_tp_id}}] = functional::elwise(self);
  }

  for (type_id_t i0 : i2a<dim_ids>()) {
    typedef join<numeric_ids, dim_ids>::type type_ids;
    for (type_id_t i1 : i2a<type_ids>()) {
      const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
      children[{{i0, i1}}] = functional::elwise(child_tp, self);
    }
  }

  children[{{string_id, string_id}}] = callable::make<KernelType<string_id, string_id>>();

  return children;
}

template <typename SelfType, template <type_id_t...> class KernelType>
nd::callable nd::comparison_operator<SelfType, KernelType>::make()
{
  auto children = SelfType::make_children();
  return functional::dispatch(ndt::type("(Any, Any) -> Any"),
                              [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                         const ndt::type *src_tp) mutable -> callable & {
                                callable &child = children[{{src_tp[0].get_id(), src_tp[1].get_id()}}];
                                if (child.is_null()) {
                                  throw std::runtime_error("no child found");
                                }

                                return child;
                              });
}

DYND_API struct nd::less nd::less;

nd::array nd::operator<(const array &a0, const array &a1) { return less(a0, a1); }

DYND_API struct nd::less_equal nd::less_equal;

nd::array nd::operator<=(const array &a0, const array &a1) { return less_equal(a0, a1); }

map<array<type_id_t, 2>, nd::callable> nd::equal::make_children()
{
  std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
  children[{{complex_float32_id, complex_float32_id}}] =
      callable::make<equal_kernel<complex_float32_id, complex_float32_id>>(0);
  children[{{complex_float64_id, complex_float64_id}}] =
      callable::make<equal_kernel<complex_float64_id, complex_float64_id>>(0);
  children[{{tuple_id, tuple_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>(0);
  children[{{struct_id, struct_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>(0);
  children[{{type_id, type_id}}] = callable::make<equal_kernel<type_id, type_id>>(0);

  return children;
}

DYND_API struct nd::equal nd::equal;

nd::array nd::operator==(const array &a0, const array &a1) { return equal(a0, a1); }

map<array<type_id_t, 2>, nd::callable> nd::not_equal::make_children()
{
  std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
  children[{{complex_float32_id, complex_float32_id}}] =
      callable::make<not_equal_kernel<complex_float32_id, complex_float32_id>>(0);
  children[{{complex_float64_id, complex_float64_id}}] =
      callable::make<not_equal_kernel<complex_float64_id, complex_float64_id>>(0);
  children[{{tuple_id, tuple_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>(0);
  children[{{struct_id, struct_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>(0);
  children[{{type_id, type_id}}] = callable::make<not_equal_kernel<type_id, type_id>>(0);

  return children;
}

DYND_API struct nd::not_equal nd::not_equal;

nd::array nd::operator!=(const array &a0, const array &a1) { return not_equal(a0, a1); }

DYND_API struct nd::greater_equal nd::greater_equal;

nd::array nd::operator>=(const array &a0, const array &a1) { return greater_equal(a0, a1); }

DYND_API struct nd::greater nd::greater;

nd::array nd::operator>(const array &a0, const array &a1) { return greater(a0, a1); }

nd::callable nd::total_order::make()
{
  std::map<std::array<type_id_t, 2>, callable> children;
  children[{{fixed_string_id, fixed_string_id}}] =
      callable::make<total_order_kernel<fixed_string_id, fixed_string_id>>();
  children[{{string_id, string_id}}] = callable::make<total_order_kernel<string_id, string_id>>();
  children[{{int32_id, int32_id}}] = callable::make<total_order_kernel<int32_id, int32_id>>();
  children[{{bool_id, bool_id}}] = callable::make<total_order_kernel<bool_id, bool_id>>();

  return functional::dispatch(ndt::type("(Any, Any) -> Any"),
                              [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                         const ndt::type *src_tp) mutable -> callable & {
                                callable &child = children[{{src_tp[0].get_id(), src_tp[1].get_id()}}];
                                if (child.is_null()) {
                                  throw std::runtime_error("no child found");
                                }

                                return child;
                              });
}

DYND_API struct nd::total_order nd::total_order;
