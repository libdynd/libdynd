//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>

namespace dynd {

/**
 * Formats the nd::array as JSON.
 *
 * \param a  The array to format as JSON.
 * \param struct_as_list  If true, formats struct objects as lists, otherwise
 *                        formats them as objects/dicts.
 */
DYND_API nd::array format_json(const nd::array &a, bool struct_as_list = false);

} // namespace dynd
