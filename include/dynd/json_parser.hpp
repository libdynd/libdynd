//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
 * This function parses the JSON, encoded as UTF-8, into an nd::array
 * of the specified type. This parser works directly from JSON to the
 * nd::array representation, interpreting the data as the requested type
 * on the fly.
 *
 * The type must have a fixed data size, so every dimension must be
 * either variable-sized or fixed-sized, not a free variable.
 *
 * \param tp  The type to interpret the JSON data.
 * \param json_begin  The beginning of the UTF-8 buffer containing the JSON.
 * \param json_end  One past the end of the UTF-8 buffer containing the JSON.
 * \param ectx  An evaluation context.
 */
nd::array parse_json(const ndt::type& tp, const char *json_begin, const char *json_end, const eval::eval_context *ectx);

/**
 * Same as the version given a type, but parses the JSON into an uninitialized
 * dynd array.
 */
void parse_json(nd::array& out, const char *json_begin, const char *json_end, const eval::eval_context *ectx);

/**
 * Parses the input json as the requested type. The input can be a string or a
 * bytes array. If the input is bytes, the parser assumes it is UTF-8 data.
 */
nd::array parse_json(const ndt::type &tp, const nd::array &json,
                     const eval::eval_context *ectx);

/**
 * Same as the version given a type, but parses the JSON into an uninitialized
 * dynd array.
 */
void parse_json(nd::array &out, const nd::array &json,
                const eval::eval_context *ectx);

inline nd::array parse_json(const ndt::type &tp, const std::string &json,
                            const eval::eval_context *ectx)
{
    return parse_json(tp, json.data(), json.data() + json.size(), ectx);
}

inline void parse_json(nd::array &out, const std::string &json,
                       const eval::eval_context *ectx)
{
    parse_json(out, json.data(), json.data() + json.size(), ectx);
}

inline nd::array
parse_json(const ndt::type &tp, const char *json,
           const eval::eval_context *ectx = &eval::default_eval_context)
{
    return parse_json(tp, json, json + strlen(json), ectx);
}

inline void
parse_json(nd::array &out, const char *json,
           const eval::eval_context *ectx = &eval::default_eval_context)
{
    return parse_json(out, json, json + strlen(json), ectx);
}

/** Interface to the JSON parser for an input of two string literals */
template <int M, int N>
inline nd::array
parse_json(const char (&dt)[M], const char (&json)[N],
           const eval::eval_context *ectx = &eval::default_eval_context)
{
    return parse_json(ndt::type(dt, dt+M-1), json, json+N-1, ectx);
}

} // namespace dynd

#endif // _DYND__JSON_PARSER_HPP_

