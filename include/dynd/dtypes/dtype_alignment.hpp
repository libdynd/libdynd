//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The realign dtype applies a more
// stringent alignment to a bytes dtype.
//
#ifndef _DND__REALIGN_DTYPE_HPP_
#define _DND__REALIGN_DTYPE_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

/**
 * Uses an appropriate view<..., bytes<>> dtype
 * to put the dtype on top of unaligned storage.
 */
dtype make_unaligned_dtype(const dtype& value_dtype);

/**
 * Reduces a dtype's alignment requirements to 1.
 */
template<typename T>
dtype make_unaligned_dtype()
{
    return make_unaligned_dtype(make_dtype<T>());
}

} // namespace dynd

#endif // _DND__REALIGN_DTYPE_HPP_
