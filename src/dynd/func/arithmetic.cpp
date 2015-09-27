//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arithmetic.hpp>

using namespace std;
using namespace dynd;

#define DYND_DefUnaryOpWithCallable(OP, NAME)                        \
DYND_API struct nd:: NAME nd:: NAME;                                 \
nd::array nd::operator OP(const array &a0) { return nd:: NAME(a0); } \

DYND_DefUnaryOpWithCallable(+, plus)
DYND_DefUnaryOpWithCallable(-, minus)

#undef DYND_DefUnaryOpWithCallable

#define DYND_DefBinaryOpWithCallable(OP, NAME)                                            \
DYND_API struct nd:: NAME nd:: NAME;                                                      \
nd::array nd::operator OP(const array &a0, const array &a1) { return nd:: NAME(a0, a1); } \

DYND_DefBinaryOpWithCallable(+, add)
DYND_DefBinaryOpWithCallable(-, subtract)
DYND_DefBinaryOpWithCallable(*, multiply)
DYND_DefBinaryOpWithCallable(/, divide)

#undef DYND_DefBinaryOpWithCallable

#define DYND_DefCompoundOpWithArrfunc(OP, NAME)       \
DYND_API struct nd:: NAME nd:: NAME;                  \
nd::array &nd::array::operator OP(const array &rhs)   \
{                                                     \
  nd:: NAME(rhs, kwds("dst", *this));                 \
  return *this;                                       \
}                                                     \

DYND_DefCompoundOpWithArrfunc(+=, compound_add)
DYND_DefCompoundOpWithArrfunc(/=, compound_div)

#undef DYND_DefCompoundOpWithArrfunc
