//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SERIALIZE_HPP_
#define _DYND__SERIALIZE_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/ndobject.hpp>

namespace dynd { namespace gfunc {

/**
 * Serializes the ndobject into a blosc-compressed bytes
 * and a metadata storage string.
 */
ndobject serialize(const ndobject& val);

/**
 * Deserializes the results of serialize() back into
 * an ndobject.
 */
ndobject deserialize(const ndobject& data);

}} // namespace dynd::gfunc

#endif // _DYND__SERIALIZE_HPP_
