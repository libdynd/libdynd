//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_FORMATTER_HPP_
#define _DYND__JSON_FORMATTER_HPP_

#include <dynd/array.hpp>

namespace dynd {

/**
 * Formats the nd::array as JSON.
 *
 * \param a  The array to format as JSON.
 * \param struct_as_list  If true, formats struct objects as lists, otherwise
 *                        formats them as objects/dicts.
 */
nd::array format_json(const nd::array &a, bool struct_as_list = false);

} // namespace dynd

#endif // _DYND__JSON_FORMATTER_HPP_
