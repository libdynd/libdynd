//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/pow_callable.hpp>

// This should eventually be made to work for complex inputs as well.
// The expression SFINAE check fails to detect that pow can be used
// with int128 and uint128 on OSX, so provide an empty conditional
// and only use real-valued inputs for this callable.
typedef join<integral_ids, float_ids>::type real_ids;
template <type_id_t, type_id_t>
using isdef_pow = std::true_type;

DYND_API nd::callable nd::pow = make_binary_arithmetic<nd::pow_callable, isdef_pow, real_ids>();
