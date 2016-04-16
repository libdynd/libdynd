//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/logical_not_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

DYND_API nd::callable nd::logical_not =
    make_unary_arithmetic<nd::logical_not_callable, dynd::detail::isdef_logical_not, arithmetic_ids>();
