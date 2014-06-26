//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/array_iter.hpp>

namespace dynd { namespace nd {

/**
 * Primitive function to construct an nd::array with each element initialized
 * to a random value. This is used only for testing right now, and it should be 
 * completely redone at some point. Variable dimensions are supported. Only a dtype
 * of double is currently supported 
 */
nd::array typed_rand(intptr_t ndim, const intptr_t *shape, const ndt::type &tp);

}} // namespace dynd::nd
