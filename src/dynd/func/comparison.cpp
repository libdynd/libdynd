//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/comparison.hpp>
#include <dynd/callables/total_order_callable.hpp>

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
  dispatcher<callable> dispatcher;
  dispatcher.insert(
      {{fixed_string_id, fixed_string_id}, make_callable<total_order_callable<fixed_string_id, fixed_string_id>>()});
  dispatcher.insert({{string_id, string_id}, make_callable<total_order_callable<string_id, string_id>>()});
  dispatcher.insert({{int32_id, int32_id}, make_callable<total_order_callable<int32_id, int32_id>>()});
  dispatcher.insert({{bool_id, bool_id}, make_callable<total_order_callable<bool_id, bool_id>>()});

  return make_callable<comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
}

DYND_DEFAULT_DECLFUNC_GET(nd::total_order)

DYND_API struct nd::total_order nd::total_order;
