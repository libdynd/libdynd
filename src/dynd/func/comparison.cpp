//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/comparison.hpp>

using namespace std;
using namespace dynd;

DYND_DEFAULT_DECLFUNC_GET(nd::less)

DYND_API struct nd::less nd::less;

nd::array nd::operator<(const array &a0, const array &a1) { return less(a0, a1); }

DYND_DEFAULT_DECLFUNC_GET(nd::less_equal)

DYND_API struct nd::less_equal nd::less_equal;

nd::array nd::operator<=(const array &a0, const array &a1) { return less_equal(a0, a1); }

DYND_DEFAULT_DECLFUNC_GET(nd::equal)

DYND_API struct nd::equal nd::equal;

nd::array nd::operator==(const array &a0, const array &a1) { return equal(a0, a1); }

DYND_DEFAULT_DECLFUNC_GET(nd::not_equal)

DYND_API struct nd::not_equal nd::not_equal;

nd::array nd::operator!=(const array &a0, const array &a1) { return not_equal(a0, a1); }

DYND_DEFAULT_DECLFUNC_GET(nd::greater_equal)

DYND_API struct nd::greater_equal nd::greater_equal;

nd::array nd::operator>=(const array &a0, const array &a1) { return greater_equal(a0, a1); }

DYND_DEFAULT_DECLFUNC_GET(nd::greater)

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

DYND_DEFAULT_DECLFUNC_GET(nd::total_order)

DYND_API struct nd::total_order nd::total_order;
