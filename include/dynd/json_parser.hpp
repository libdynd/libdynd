//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_PARSER_HPP_
#define _DYND__JSON_PARSER_HPP_

#include <dynd/ndobject.hpp>

namespace dynd {

/**
 * This function parses the JSON, encoded as UTF-8, into an ndobject
 * of the specified dtype. This parser works directly from JSON to the
 * ndobject representation, interpreting the data as the requested type
 * on the fly.
 *
 * If additional shape information is required to construct
 * the ndobject, the parsing will happen in two passes - first
 * to deduce the shape, then to populate the ndobject.
 *
 * \param dt  The dtype to interpret the JSON data.
 * \param json_begin  The beginning of the UTF-8 buffer containing the JSON.
 * \param json_end  One past the end of the UTF-8 buffer containing the JSON.
 */
ndobject parse_json(const dtype& dt, const char *json_begin, const char *json_end);

/**
 * Parses the input json as the requested dtype. The input can be a string or a
 * bytes ndobject. If the input is bytes, the parser assumes it is UTF-8 data.
 */
ndobject parse_json(const dtype& dt, const ndobject& json);

} // namespace dynd

#endif // _DYND__JSON_PARSER_HPP_