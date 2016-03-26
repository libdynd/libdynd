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

DYND_API nd::callable nd::greater = make_callable<comparison_dispatch_callable>(
    ndt::type("(Any, Any) -> Any"), make_comparison_children<nd::greater, nd::greater_callable>());

nd::array nd::operator>(const array &a0, const array &a1) { return greater(a0, a1); }

DYND_API nd::callable nd::total_order = nd::make_callable<nd::comparison_dispatch_callable>(
    ndt::type("(Any, Any) -> Any"),
    dispatcher<callable>{{{fixed_string_id, fixed_string_id},
                          nd::make_callable<nd::total_order_callable<fixed_string_id, fixed_string_id>>()},
                         {{string_id, string_id}, nd::make_callable<nd::total_order_callable<string_id, string_id>>()},
                         {{int32_id, int32_id}, nd::make_callable<nd::total_order_callable<int32_id, int32_id>>()},
                         {{bool_id, bool_id}, nd::make_callable<nd::total_order_callable<bool_id, bool_id>>()}});
