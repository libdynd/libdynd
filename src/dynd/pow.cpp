//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/binary_arithmetic.hpp>
#include <dynd/callables/pow_callable.hpp>

// Use arithmetic_ids here since this should eventually be made to work
// for complex inputs as well. Those cases will currently be filtered
// out by the isdef_pow test for whether or not the needed operation
// is defined.
DYND_API nd::callable nd::pow = make_binary_arithmetic<nd::pow_callable, dynd::detail::isdef_pow, arithmetic_ids>();
