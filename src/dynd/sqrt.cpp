//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/sqrt_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

template <typename>
using isdef_sqrt = std::true_type;

DYND_API nd::callable nd::sqrt = make_unary_arithmetic<nd::sqrt_callable, isdef_sqrt, type_sequence<float, double>>();
