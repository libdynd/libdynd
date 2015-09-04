//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd { namespace ndt {

/**
 * Uses an appropriate view<..., bytes<>> type
 * to put the type on top of unaligned storage.
 */
DYND_API ndt::type make_unaligned(const ndt::type& value_type);

/**
 * Reduces a type's alignment requirements to 1.
 */
template<typename T>
ndt::type make_unaligned()
{
    return make_unaligned(type::make<T>());
}

}} // namespace dynd::ndt
