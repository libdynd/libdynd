//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_FORMATTER_HPP_
#define _DYND__JSON_FORMATTER_HPP_

#include <dynd/ndobject.hpp>

namespace dynd {

/**
 * Formats the ndobject as JSON.
 *
 * \param n  The object to format as JSON.
 */
ndobject format_json(const ndobject& n);

} // namespace dynd

#endif // _DYND__JSON_FORMATTER_HPP_
