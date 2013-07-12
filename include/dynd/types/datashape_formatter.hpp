//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATASHAPE_FORMATTER_HPP_
#define _DYND__DATASHAPE_FORMATTER_HPP_

#include <dynd/array.hpp>

namespace dynd {

/**
 * Formats the dynd array's type + metadata as a
 * Blaze datashape.
 *
 * \param n  The object whose datashape to produce
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
std::string format_datashape(const nd::array& n,
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the type as a blaze datashape.
 *
 * \param tp  The type whose datashape to produce
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
std::string format_datashape(const ndt::type& tp,
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the given type + metadata + data as a Blaze
 * datashape, writing the output to the stream. One can
 * provide just the type (NULL metadata/data) or just
 * the type/metadata (NULL data) as well as specifying
 * a full dynd array type/metadata/data.
 *
 * \param o  The stream where to write the datashape.
 * \param tp  The data type.
 * \param metadata  The data type's metadata. This may be NULL.
 * \param data  The data for a leading element corresponding to the type/metadata.
 *              This may be NULL.
 * \param multiline  If true, split the datashape across multiple lines.
 */
void format_datashape(std::ostream& o, const ndt::type& tp,
                const char *metadata, const char *data, bool multiline);


} // namespace dynd

#endif // _DYND__DATASHAPE_FORMATTER_HPP_
