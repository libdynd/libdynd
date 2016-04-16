//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/compound_add_callable.hpp>
#include <dynd/compound_arithmetic.hpp>

DYND_API nd::callable nd::compound_add = make_compound_arithmetic<nd::compound_add_callable, binop_ids>();
