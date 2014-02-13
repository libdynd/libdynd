//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_FORMATTER_HPP_
#define _DYND__JSON_FORMATTER_HPP_

#include <dynd/array.hpp>

namespace dynd {

/**
 * Formats the ndobject as JSON.
 *
 * \param n  The object to format as JSON.
 */
nd::array format_json(const nd::array& n);

} // namespace dynd

#endif // _DYND__JSON_FORMATTER_HPP_
