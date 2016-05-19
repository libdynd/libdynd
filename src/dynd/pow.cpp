//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/pow_callable.hpp>

// This should eventually be made to work for integer
// and complex inputs as well.

// The expression SFINAE check fails to detect that pow can be used
// with float128 on OSX, so provide an empty conditional
// and only use real-valued inputs for this callable.

// We don't plan on upcasting integers or making float/int
// operations upcast to doubles in the future, so, for now,
// only provide this callable for float32 and float64, where
// std::pow already unambiguously does what we want for all
// given combinations of input types.

template <typename, typename>
using isdef_pow = std::true_type;

DYND_API nd::callable nd::pow = make_binary_arithmetic<nd::pow_callable, isdef_pow, type_sequence<float, double>>();
