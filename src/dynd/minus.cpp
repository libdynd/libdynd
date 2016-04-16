//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/minus_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

DYND_API nd::callable nd::minus =
    make_unary_arithmetic<nd::minus_callable, dynd::detail::isdef_minus, arithmetic_ids>();
