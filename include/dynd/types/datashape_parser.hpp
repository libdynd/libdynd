//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATASHAPE_PARSER_HPP_
#define _DYND__DATASHAPE_PARSER_HPP_

#include <dynd/type.hpp>

namespace dynd {

/**
 * Parses a blaze datashape, producing the canonical dynd type for
 * it. The datashape string should be provided as a begin/end pointer
 * pair.
 *
 * The string buffer should be encoded with UTF-8.
 *
 * \param datashape_begin  The start of the buffer containing the datashape.
 * \param datashape_end    The end of the buffer containing the datashape.
 */
ndt::type type_from_datashape(const char *datashape_begin, const char *datashape_end);

inline ndt::type type_from_datashape(const std::string& datashape)
{
    return type_from_datashape(datashape.data(), datashape.data() + datashape.size());
}

inline ndt::type type_from_datashape(const char *datashape)
{
    return type_from_datashape(datashape, datashape + strlen(datashape));
}

template<int N>
inline ndt::type type_from_datashape(const char (&datashape)[N])
{
    return type_from_datashape(datashape, datashape + N - 1);
}

namespace init {
void datashape_parser_init();
void datashape_parser_cleanup();
}

} // namespace dynd

#endif // _DYND__DATASHAPE_PARSER_HPP_

