//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>

namespace dynd {

/**
 * Formats the dynd array's type + arrmeta as a
 * Blaze datashape.
 *
 * \param n  The object whose datashape to produce
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
DYND_API std::string format_datashape(const nd::array& n,
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the type as a blaze datashape.
 *
 * \param tp  The type whose datashape to produce
 * \param prefix  Prepends the datashape with this string
 * \param multiline  If true, split the datashape across multiple lines.
 */
DYND_API std::string format_datashape(const ndt::type& tp,
                const std::string& prefix = "",
                bool multiline = true);

/**
 * Formats the given type + arrmeta + data as a Blaze
 * datashape, writing the output to the stream. One can
 * provide just the type (NULL arrmeta/data) or just
 * the type/arrmeta (NULL data) as well as specifying
 * a full dynd array type/arrmeta/data.
 *
 * \param o  The stream where to write the datashape.
 * \param tp  The data type.
 * \param arrmeta  The data type's arrmeta. This may be NULL.
 * \param data  The data for a leading element corresponding to the type/arrmeta.
 *              This may be NULL.
 * \param multiline  If true, split the datashape across multiple lines.
 */
DYND_API void format_datashape(std::ostream& o, const ndt::type& tp,
                const char *arrmeta, const char *data, bool multiline);


} // namespace dynd
