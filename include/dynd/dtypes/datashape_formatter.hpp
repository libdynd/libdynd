//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATASHAPE_FORMATTER_HPP_
#define _DYND__DATASHAPE_FORMATTER_HPP_

#include <dynd/ndobject.hpp>

namespace dynd {

/**
 * Formats the ndobject's dtype + metadata as a
 * Blaze datashape.
 *
 * \param n  The object whose datashape to produce
 */
std::string format_datashape(const ndobject& n);

} // namespace dynd

#endif // _DYND__DATASHAPE_FORMATTER_HPP_
