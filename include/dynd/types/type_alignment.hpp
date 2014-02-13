//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_ALIGNMENT_HPP_
#define _DYND__TYPE_ALIGNMENT_HPP_

#include <dynd/type.hpp>

namespace dynd { namespace ndt {

/**
 * Uses an appropriate view<..., bytes<>> type
 * to put the type on top of unaligned storage.
 */
ndt::type make_unaligned(const ndt::type& value_type);

/**
 * Reduces a type's alignment requirements to 1.
 */
template<typename T>
ndt::type make_unaligned()
{
    return make_unaligned(ndt::make_type<T>());
}

}} // namespace dynd::ndt

#endif // _DYND__TYPE_ALIGNMENT_HPP_
