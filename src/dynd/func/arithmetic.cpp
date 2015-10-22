//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

#define DYND_DEF_UNARY_OP_AND_CALLABLE(OP, NAME)                                                                       \
  DYND_API struct nd::NAME nd::NAME;                                                                                   \
  nd::array nd::operator OP(const array &a0)                                                                           \
  {                                                                                                                    \
    return nd::NAME(a0);                                                                                               \
  }

DYND_DEF_UNARY_OP_AND_CALLABLE(+, plus)
DYND_DEF_UNARY_OP_AND_CALLABLE(-, minus)
DYND_DEF_UNARY_OP_AND_CALLABLE(!, logical_not)
DYND_DEF_UNARY_OP_AND_CALLABLE(~, bitwise_not)

#undef DYND_DEF_UNARY_OP_AND_CALLABLE

#define DYND_DEF_BINARY_OP_WITH_CALLABLE(OP, NAME)                                                                     \
  DYND_API struct nd::NAME nd::NAME;                                                                                   \
  nd::array nd::operator OP(const array &a0, const array &a1)                                                          \
  {                                                                                                                    \
    return nd::NAME(a0, a1);                                                                                           \
  }

DYND_DEF_BINARY_OP_WITH_CALLABLE(+, add)
DYND_DEF_BINARY_OP_WITH_CALLABLE(-, subtract)
DYND_DEF_BINARY_OP_WITH_CALLABLE(*, multiply)
DYND_DEF_BINARY_OP_WITH_CALLABLE(/, divide)
DYND_DEF_BINARY_OP_WITH_CALLABLE(&&, logical_and)
DYND_DEF_BINARY_OP_WITH_CALLABLE(||, logical_or)

#undef DYND_DEF_BINARY_OP_WITH_CALLABLE

#define DYND_DEF_COMPOUND_OP_WITH_CALLABLE(OP, NAME)                                                                   \
  DYND_API struct nd::NAME nd::NAME;                                                                                   \
  nd::array &nd::array::operator OP(const array &rhs)                                                                  \
  {                                                                                                                    \
    nd::NAME(rhs, kwds("dst", *this));                                                                                 \
    return *this;                                                                                                      \
  }

DYND_DEF_COMPOUND_OP_WITH_CALLABLE(+=, compound_add)
DYND_DEF_COMPOUND_OP_WITH_CALLABLE(/=, compound_div)

#undef DYND_DEF_COMPOUND_OP_WITH_CALLABLE
