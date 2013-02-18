//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATASHAPE_PARSER_HPP_
#define _DYND__DATASHAPE_PARSER_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

/**
 * Parses a Blaze datashape, producing the canonical DyND dtype for
 * it. The datashape string should be provided as a begin/end pointer
 * pair.
 *
 * The string buffer should be encoded with UTF-8.
 *
 * \param datashape_begin  The start of the buffer containing the datashape.
 * \param datashape_end    The end of the buffer containing the datashape.
 */
dtype dtype_from_datashape(const char *datashape_begin, const char *datashape_end);

inline dtype dtype_from_datashape(const std::string& datashape)
{
    return dtype_from_datashape(datashape.data(), datashape.data() + datashape.size());
}

inline dtype dtype_from_datashape(const char *datashape)
{
    return dtype_from_datashape(datashape, datashape + strlen(datashape));
}

template<int N>
inline dtype dtype_from_datashape(const char (&datashape)[N])
{
    return dtype_from_datashape(datashape, datashape + N - 1);
}

} // namespace dynd

#endif // _DYND__DATASHAPE_PARSER_HPP_

