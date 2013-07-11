//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The realign dtype applies a more
// stringent alignment to a bytes dtype.
//
#ifndef _DYND__REALIGN_TYPE_HPP_
#define _DYND__REALIGN_TYPE_HPP_

#include <dynd/type.hpp>

namespace dynd { namespace ndt {

/**
 * Uses an appropriate view<..., bytes<>> type
 * to put the type on top of unaligned storage.
 */
ndt::type make_unaligned_type(const ndt::type& value_type);

/**
 * Reduces a type's alignment requirements to 1.
 */
template<typename T>
ndt::type make_unaligned_type()
{
    return make_unaligned_type(ndt::make_dtype<T>());
}

}} // namespace dynd::ndt

#endif // _DYND__REALIGN_TYPE_HPP_
