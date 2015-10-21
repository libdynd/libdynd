//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/comparison.hpp>

using namespace std;
using namespace dynd;

DYND_API struct nd::less nd::less;

nd::array nd::operator<(const array &a0, const array &a1)
{
  return less(a0, a1);
}

DYND_API struct nd::less_equal nd::less_equal;

nd::array nd::operator<=(const array &a0, const array &a1)
{
  return less_equal(a0, a1);
}

DYND_API struct nd::equal nd::equal;

nd::array nd::operator==(const array &a0, const array &a1)
{
  return equal(a0, a1);
}

DYND_API struct nd::not_equal nd::not_equal;

nd::array nd::operator!=(const array &a0, const array &a1)
{
  return not_equal(a0, a1);
}

DYND_API struct nd::greater_equal nd::greater_equal;

nd::array nd::operator>=(const array &a0, const array &a1)
{
  return greater_equal(a0, a1);
}

DYND_API struct nd::greater nd::greater;

nd::array nd::operator>(const array &a0, const array &a1)
{
  return greater(a0, a1);
}

nd::callable nd::total_order::children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

DYND_API struct nd::total_order nd::total_order;
