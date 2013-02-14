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
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
std::string format_datashape(const ndobject& n,
                const std::string& prefix = "type BlazeDataShape = ",
                bool multiline = true);

/**
 * Formats the given dtype + metadata as a Blaze
 * datashape, writing the output to the stream.
 *
 * \param o  The stream where to write the datashape.
 * \param dt  The data type.
 * \param metadata  The data type's metadata.
 * \param multiline  If true, split the datashape across multiple lines.
 */
void format_datashape(std::ostream& o, const dtype& dt, const char *metadata,
                bool multiline);


} // namespace dynd

#endif // _DYND__DATASHAPE_FORMATTER_HPP_
