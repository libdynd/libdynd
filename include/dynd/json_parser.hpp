//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_PARSER_HPP_
#define _DYND__JSON_PARSER_HPP_

#include <dynd/array.hpp>

namespace dynd {

/**
 * Validates UTF-8 encoded JSON, throwing an exception if it
 * is not valid.
 *
 * \param json_begin  The beginning of the UTF-8 buffer containing the JSON.
 * \param json_end  One past the end of the UTF-8 buffer containing the JSON.
 */
void validate_json(const char *json_begin, const char *json_end);

/**
 * This function parses the JSON, encoded as UTF-8, into an ndobject
 * of the specified dtype. This parser works directly from JSON to the
 * ndobject representation, interpreting the data as the requested type
 * on the fly.
 *
 * The dtype must have a fixed data size, so every dimension must be
 * either variable-sized or fixed-sized, not a free variable.
 *
 * \param dt  The dtype to interpret the JSON data.
 * \param json_begin  The beginning of the UTF-8 buffer containing the JSON.
 * \param json_end  One past the end of the UTF-8 buffer containing the JSON.
 */
nd::array parse_json(const ndt::type& dt, const char *json_begin, const char *json_end);

/**
 * Same as the version given a dtype, but parses the JSON into an uninitialized
 * dynd array.
 */
void parse_json(nd::array& out, const char *json_begin, const char *json_end);

/**
 * Parses the input json as the requested dtype. The input can be a string or a
 * bytes array. If the input is bytes, the parser assumes it is UTF-8 data.
 */
nd::array parse_json(const ndt::type& dt, const nd::array& json);

/**
 * Same as the version given a dtype, but parses the JSON into an uninitialized
 * dynd array.
 */
void parse_json(nd::array& out, const nd::array& json);

inline nd::array parse_json(const ndt::type& dt, const std::string& json) {
    return parse_json(dt, json.data(), json.data() + json.size());
}

inline void parse_json(nd::array& out, const std::string& json) {
    parse_json(out, json.data(), json.data() + json.size());
}

inline nd::array parse_json(const ndt::type& dt, const char *json) {
    return parse_json(dt, json, json + strlen(json));
}

inline void parse_json(nd::array& out, const char *json) {
    return parse_json(out, json, json + strlen(json));
}

/** Interface to the JSON parser for an input of two string literals */
template<int M, int N>
inline nd::array parse_json(const char (&dt)[M], const char (&json)[N]) {
    return parse_json(ndt::type(dt, dt+M-1), json, json+N-1);
}

} // namespace dynd

#endif // _DYND__JSON_PARSER_HPP_

