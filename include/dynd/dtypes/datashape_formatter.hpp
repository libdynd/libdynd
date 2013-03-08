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
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the dtype as a
 * Blaze datashape.
 *
 * \param d  The dtype whose datashape to produce
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
std::string format_datashape(const dtype& d,
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the given dtype + metadata + data as a Blaze
 * datashape, writing the output to the stream. One can
 * provide just the dtype (NULL metadata/data) or just
 * the dtype/metadata (NULL data) as well as specifying
 * a full ndobject dtype/metadata/data.
 *
 * \param o  The stream where to write the datashape.
 * \param dt  The data type.
 * \param metadata  The data type's metadata. This may be NULL.
 * \param data  The data for a leading element corresponding to the dtype/metadata.
 *              This may be NULL.
 * \param multiline  If true, split the datashape across multiple lines.
 */
void format_datashape(std::ostream& o, const dtype& dt,
                const char *metadata, const char *data, bool multiline);


} // namespace dynd

#endif // _DYND__DATASHAPE_FORMATTER_HPP_
