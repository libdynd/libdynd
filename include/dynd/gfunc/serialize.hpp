//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SERIALIZE_HPP_
#define _DYND__SERIALIZE_HPP_

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/array.hpp>

namespace dynd { namespace gfunc {

/**
 * Serializes the ndobject into a blosc-compressed bytes
 * and a metadata storage string.
 */
nd::array serialize(const nd::array& val);

/**
 * Deserializes the results of serialize() back into
 * an ndobject.
 */
nd::array deserialize(const nd::array& data);

}} // namespace dynd::gfunc

#endif // _DYND__SERIALIZE_HPP_
