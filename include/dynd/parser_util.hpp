 //
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PARSER_UTIL_HPP_
#define _DYND__PARSER_UTIL_HPP_

#include <string>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd { namespace parse {

/**
 * A helper class to save/restore the state
 * of 'begin' during parsing.
 *
 * Example:
 *    bool parse_ABCs(const char *&begin, const char *end)
 *    {
 *        saved_begin_state sbs(begin);
 *        for (int c = 'A'; c <= 'Z' && begin < end; ++c) {
 *            if (*begin++ != c) {
 *                // the saved_begin_state will restore begin
 *                return sbs.fail();
 *            }
 *        }
 *        return sbs.succeed();
 *    }
 */
class saved_begin_state {
    const char *&m_begin;
    const char *m_saved_begin;
    bool m_succeeded;

    // Non-copyable
    saved_begin_state(const saved_begin_state&);
    saved_begin_state& operator=(const saved_begin_state&);
public:
    explicit saved_begin_state(const char *&begin)
        : m_begin(begin), m_saved_begin(begin), m_succeeded(false) {}

    ~saved_begin_state() {
        if (!m_succeeded) {
            // Restore begin if not success
            m_begin = m_saved_begin;
        }
    }

    inline bool succeed() {
        m_succeeded = true;
        return true;
    }

    inline bool fail() {
        return false;
    }

    inline const char *saved_begin() const {
        return m_saved_begin;
    }
};

/**
 * An error message thrown when a parse error is encountered.
 */
class parse_error : public std::invalid_argument {
    const char *m_position;
public:
    parse_error(const char *position, const std::string& message)
        : std::invalid_argument(message), m_position(position) {
    }
    virtual ~parse_error() throw () {
    }
    const char *get_position() const {
        return m_position;
    }
};

/**
 * Modifies `begin` to skip past any whitespace.
 *
 * Example:
 *     skip_whitespace(begin, end);
 */
inline void skip_whitespace(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    while (begin < end && isspace(*begin)) {
        ++begin;
    }
    rbegin = begin;
}

/**
 * Modifies `begin` to skip past any whitespace and comments starting with #.
 *
 * Example:
 *     skip_whitespace(begin, end);
 */
inline void skip_whitespace_and_pound_comments(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    while (begin < end && isspace(*begin)) {
        ++begin;
    }

    // Comments
    if (begin < end && *begin == '#') {
        const char *line_end = (const char *)memchr(begin, '\n', end - begin);
        if (line_end == NULL) {
            begin = end;
        } else {
            begin = line_end + 1;
            skip_whitespace_and_pound_comments(begin, end);
        }
    }

    rbegin = begin;
}

/**
 * Modifies `rbegin` to skip past any whitespace. Returns false
 * if no whitespace was found to skip.
 *
 * Example:
 *     if (!skip_required_whitespace(begin, end)) {
 *         // Do something if there was no whitespace
 *     }
 */
inline bool skip_required_whitespace(const char *&rbegin, const char *end)
{
    const char *begin = rbegin;
    if (begin < end && isspace(*begin)) {
        ++begin;
        while (begin < end && isspace(*begin)) {
            ++begin;
        }
        rbegin = begin;
        return true;
    } else {
        return false;
    }
}

/**
 * Skips whitespace, then matches the provided literal string token. On success,
 * returns true and modifies `rbegin` to point after the token. If the token is a
 * single character, use the other `parse_token` function which accepts
 * a char.
 *
 * Example:
 *     // Match the token "while"
 *     if (parse_token(begin, end, "while")) {
 *         // Handle while statement
 *     } else {
 *         // No while token found
 *     }
 */
template <int N>
inline bool parse_token(const char *&rbegin, const char *end,
                        const char (&token)[N])
{
    const char *begin = rbegin;
    skip_whitespace(begin, end);
    if (N - 1 <= end - begin &&
            memcmp(begin, token, N - 1) == 0) {
        rbegin = begin + N - 1;
        return true;
    } else {
        return false;
    }
}

/**
 * Skips whitespace, then matches the provided literal character token. On
 * success, returns true and modifies `rbegin` to point after the token.
 *
 * Example:
 *     // Match the token "*"
 *     if (parse_token(begin, end, '*')) {
 *         // Handle multiplication
 *     } else {
 *         // No * token found
 *     }
 */
inline bool parse_token(const char *&rbegin, const char *end, char token)
{
    const char *begin = rbegin;
    skip_whitespace(begin, end);
    if (1 <= end - begin && *begin == token) {
        rbegin = begin + 1;
        return true;
    } else {
        return false;
    }
}

/**
 * Without skipping whitespace, matches the provided literal string token. On
 * success, returns true and modifies `rbegin` to point after the token. If the
 * token is a single character, use the other `parse_token_no_ws` function which
 * accepts a char.
 *
 * Example:
 *     // Match the token "while"
 *     if (parse_token_no_ws(begin, end, "while")) {
 *         // Handle while statement
 *     } else {
 *         // No while token found
 *     }
 */
template <int N>
inline bool parse_token_no_ws(const char *&rbegin, const char *end,
                              const char (&token)[N])
{
    const char *begin = rbegin;
    if (N - 1 <= end - begin && memcmp(begin, token, N - 1) == 0) {
        rbegin = begin + N - 1;
        return true;
    } else {
        return false;
    }
}

/**
 * Without skipping whitespace, matches the provided literal character token. On
 * success, returns true and modifies `rbegin` to point after the token.
 *
 * Example:
 *     // Match the token "*"
 *     if (parse_token_no_ws(begin, end, '*')) {
 *         // Handle multiplication
 *     } else {
 *         // No * token found
 *     }
 */
inline bool parse_token_no_ws(const char *&rbegin, const char *end, char token)
{
    const char *begin = rbegin;
    if (1 <= end - begin && *begin == token) {
        rbegin = begin + 1;
        return true;
    } else {
        return false;
    }
}

/**
 * Without skipping whitespace, parses a name matching
 * the regex "[a-zA-Z_][a-zA-Z0-9_]*".
 * Returns true if there is a match, setting `out_strbegin`
 * and `out_strend` to the character range which was matched.
 *
 * Example:
 *     // Match a name
 *     const char *strbegin, *strend;
 *     if (parse_name_no_ws(begin, end, strbegin, strend)) {
 *         // Match buffer range [strbegin, strend) as needed
 *     } else {
 *         // Non-alphabetic character is next
 *     }
 */
bool parse_name_no_ws(const char *&rbegin, const char *end,
                      const char *&out_strbegin, const char *&out_strend);

/**
 * Without skipping whitespace, parses a name containing only
 * alphabetical characters, matching the regex "[a-zA-Z]+".
 * Returns true if there is a match, setting `out_strbegin`
 * and `out_strend` to the character range which was matched.
 *
 * Example:
 *     // Match an alphabetic bareword
 *     const char *strbegin, *strend;
 *     if (parse_alpha_name_no_ws(begin, end, strbegin, strend)) {
 *         // Match buffer range [strbegin, strend) as needed
 *     } else {
 *         // Non-alphabetic character is next
 *     }
 */
bool parse_alpha_name_no_ws(const char *&rbegin, const char *end,
                            const char *&out_strbegin, const char *&out_strend);

/**
 * Without skipping whitespace, parses a double-quoted string.
 * Returns the part matched inside the double quotes in the output
 * string range, also setting the output flag if any escapes were in
 * the string, requiring processing to remove the escapes.
 *
 * Example:
 *     // Match a double-quoted string
 *     const char *strbegin, *strend;
 *     bool escaped;
 *     if (parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
 *         if (!escaped) {
 *             // Use the matched string bytes directly
 *         } else {
 *             string result;
 *             unescape_string(strbegin, strend, result);
 *             // Use result
 *         }
 *     } else {
 *         // Was not a double-quoted string
 *     }
 */
bool parse_doublequote_string_no_ws(const char *&rbegin, const char *end,
                                    const char *&out_strbegin,
                                    const char *&out_strend, bool &out_escaped);

/**
 * Unescapes the string provided in the byte range into the
 * output string as UTF-8. Typically used with the
 * ``parse_doublequote_string_no_ws`` function.
 */
void unescape_string(const char *strbegin, const char *strend,
                     std::string &out);

/**
 * Without skipping whitespace, parses a range of bytes following
 * the JSON number grammar, returning its range of bytes.
 */
bool parse_json_number_no_ws(const char *&rbegin, const char *end,
                             const char *&out_nbegin, const char *&out_nend);

/**
 * Does an exact comparison of a byte range to a string literal.
 */
template <int N>
inline bool compare_range_to_literal(const char *begin, const char *end,
                        const char (&token)[N])
{
    return (end - begin) == N - 1 &&
        !memcmp(begin, token, N - 1);
}

/**
 * Without skipping whitespace, parses an unsigned integer.
 *
 * Example:
 *     // Match a two digit month
 *     const char *match_begin, *match_end;
 *     if (parse_unsigned_int_no_ws(begin, end, match_begin, match_end) {
 *         // Convert to int, process
 *     } else {
 *         // Couldn't match unsigned integer
 *     }
 */
inline bool parse_unsigned_int_no_ws(const char *&rbegin, const char *end,
                                     const char *&out_strbegin,
                                     const char *&out_strend)
{
    const char *begin = rbegin;
    if (begin < end) {
        if ('1' <= *begin && *begin <= '9') {
            ++begin;
            while (begin < end && ('0' <= *begin && *begin <= '9')) {
                ++begin;
            }
            out_strbegin = rbegin;
            out_strend = begin;
            rbegin = begin;
            return true;
        } else if (*begin == '0') {
            if (begin + 1 < end && ('0' <= *(begin + 1) && *(begin + 1) <= '9')) {
                // Don't match leading zeros
                return false;
            } else {
                out_strbegin = begin;
                out_strend = begin + 1;
                rbegin = begin + 1;
                return true;
            }
        } else {
            return false;
        }
    } else {
        return false;
    }
}

/**
 * Without skipping whitespace, parses an integer with exactly two digits.
 * A leading zero is accepted.
 *
 * Example:
 *     // Match a two digit month
 *     int month;
 *     if (parse_2digit_int_no_ws(begin, end, month) {
 *         // Validate and process month
 *     } else {
 *         // Couldn't match month as an integer
 *     }
 */
bool parse_2digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

/**
 * Without skipping whitespace, parses an integer with one or two digits.
 * A leading zero is accepted.
 *
 * Example:
 *     // Match a one or two digit day ("Sept 3, 1997", "Sept 03, 1997", or
 *     // "Sept 15, 1997")
 *     int day;
 *     if (parse_1or2digit_int_no_ws(begin, end, day) {
 *         // Validate and process day
 *     } else {
 *         // Couldn't match day
 *     }
 */
bool parse_1or2digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

/**
 * Without skipping whitespace, parses an integer with exactly four digits.
 * A leading zero is accepted.
 *
 * Example:
 *     // Match a four digit year
 *     int year;
 *     if (parse_4digit_int_no_ws(begin, end, year) {
 *         // Process year
 *     } else {
 *         // Couldn't match year
 *     }
 */
bool parse_4digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

/**
 * Without skipping whitespace, parses an integer with exactly six digits.
 * A leading zero is accepted.
 *
 * Example:
 *     // Match a six digit year
 *     int year;
 *     if (parse_6digit_int_no_ws(begin, end, year) {
 *         // Process year
 *     } else {
 *         // Couldn't match year
 *     }
 */
bool parse_6digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

/**
 * Converts a string containing only an unsigned integer (no leading or
 * trailing space, etc) into a uint64, setting the output over flow or
 * bad parse flags if there are problems.
 */
uint64_t checked_string_to_uint64(const char *begin, const char *end,
                                  bool &out_overflow, bool &out_badparse);

/**
 * Converts a string containing only an unsigned integer (no leading or
 * trailing space, etc) into a uint64, ignoring any problems.
 */
uint64_t unchecked_string_to_uint64(const char *begin, const char *end);

/**
 * Converts a string containing an integer (no leading or trailing space)
 * into an integer with the specified type id, using the specified error
 * mode to handle errors.
 */
void string_to_int(char *out_int, type_id_t tid, const char *begin,
                   const char *end, assign_error_mode errmode);

/**
 * A helper class for matching a bunch of names and getting an integer.
 * Arrays of this struct should be in alphabetical order.
 *
 * When used together with the `parse_ci_str_named_value_no_ws` function,
 * the strings must be all in lower case.
 *
 * Example:
 *     // Note alphabetical order
 *     const named_value weekday_table[] = {
 *         named_value("fri", 4),
 *         named_value("mon", 0),
 *         named_value("sat", 5),
 *         named_value("sun", 6),
 *         named_value("thu", 3),
 *         named_value("tue", 1),
 *         named_value("wed", 2),
 *     }
 */
struct named_value {
    const char *name;
    int value;
    DYND_CONSTEXPR named_value(const char *name_, int value_)
        : name(name_), value(value_) {}
};

/**
 * Without skipping whitespace, matches a case insensitive alphabetical
 * string using a sorted list of named_value structures to get the value.
 *
 * All the strings in the named_value table must be lower case, and in
 * alphabetically sorted order.
 *
 *
 * Example:
 *     if (parse_ci_alpha_str_named_value_no_ws(begin, end, weekday_table,
 *                                              weekday) {
 *         // Handle matched weekday
 *     } else {
 *         // No weekday matched
 *     }
 */
template <int N>
inline bool parse_ci_alpha_str_named_value_no_ws(const char *&rbegin, const char *end,
                                  named_value (&nvt)[N], int &out_value)
{
    // TODO: Could specialize two implementations based on the size of N,
    //       for small N do a linear search, big N do a binary search.

    const char *begin = rbegin;
    const char *strbegin, *strend;
    if (!parse_alpha_name_no_ws(begin, end, strbegin, strend)) {
        return false;
    }
    int strfirstchar = tolower(*strbegin);
    // Search through the named value table for a matching string
    for (int i = 0; i < N; ++i) {
        const char *name = nvt[i].name;
        // Compare the first character
        if (*name++ == strfirstchar) {
            const char *strptr = strbegin + 1;
            // Compare the rest of the characters
            while (*name != '\0' && strptr < strend &&
                   *name == tolower(*strptr)) {
                ++name; ++strptr;
            }
            if (*name == '\0' && strptr == strend) {
                out_value = nvt[i].value;
                rbegin = begin;
                return true;
            }
        }
    }

    return false;
}

}} // namespace dynd::parse

#endif // _DYND__PARSER_UTIL_HPP_
