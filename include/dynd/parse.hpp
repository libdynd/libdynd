//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/string_encodings.hpp>

#define DYND_BOOL_NA (2)
#define DYND_INT8_NA (std::numeric_limits<int8_t>::min())
#define DYND_INT16_NA (std::numeric_limits<int16_t>::min())
#define DYND_INT32_NA (std::numeric_limits<int32_t>::min())
#define DYND_INT64_NA (std::numeric_limits<int64_t>::min())
#define DYND_INT128_NA (std::numeric_limits<int128>::min())
#define DYND_FLOAT16_NA_AS_UINT (0x7e0au)
#define DYND_FLOAT32_NA_AS_UINT (0x7f8007a2U)
#define DYND_FLOAT64_NA_AS_UINT (0x7ff00000000007a2ULL)

namespace dynd {

struct nocheck_t {
};

static const nocheck_t nocheck = nocheck_t();

inline void raise_string_cast_error(const ndt::type &dst_tp, const ndt::type &string_tp, const char *arrmeta,
                                    const char *data)
{
  std::stringstream ss;
  ss << "cannot cast string ";
  string_tp.print_data(ss, arrmeta, data);
  ss << " to " << dst_tp;
  throw std::invalid_argument(ss.str());
}

inline void raise_string_cast_error(const ndt::type &dst_tp, const char *begin, const char *end)
{
  std::stringstream ss;
  ss << "cannot cast string ";
  ss.write(begin, end - begin);
  ss << " to " << dst_tp;
  throw std::invalid_argument(ss.str());
}

inline void raise_string_cast_overflow_error(const ndt::type &dst_tp, const ndt::type &string_tp, const char *arrmeta,
                                             const char *data)
{
  std::stringstream ss;
  ss << "overflow converting string ";
  string_tp.print_data(ss, arrmeta, data);
  ss << " to " << dst_tp;
  throw std::overflow_error(ss.str());
}

inline void raise_string_cast_overflow_error(const ndt::type &dst_tp, const char *begin, const char *end)
{
  std::stringstream ss;
  ss << "overflow converting string ";
  ss.write(begin, end - begin);
  ss << " to " << dst_tp;
  throw std::overflow_error(ss.str());
}

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
class DYND_API saved_begin_state {
  const char *&m_begin;
  const char *m_saved_begin;
  bool m_succeeded;

  // Non-copyable
  saved_begin_state(const saved_begin_state &);
  saved_begin_state &operator=(const saved_begin_state &);

public:
  explicit saved_begin_state(const char *&begin) : m_begin(begin), m_saved_begin(begin), m_succeeded(false) {}

  ~saved_begin_state()
  {
    if (!m_succeeded) {
      // Restore begin if not success
      m_begin = m_saved_begin;
    }
  }

  inline bool succeed()
  {
    m_succeeded = true;
    return true;
  }

  inline bool fail() { return false; }

  inline const char *saved_begin() const { return m_saved_begin; }
};

/**
 * An error message thrown when a parse error is encountered.
 * All methods must be inlined since this class is not exported as a part
 * of the dll on Windows.
 */
class parse_error : public std::invalid_argument {
  const char *m_position;

public:
  parse_error(const char *position, const std::string &message) : std::invalid_argument(message), m_position(position)
  {
  }
  virtual ~parse_error() throw() {}
  const char *get_position() const { return m_position; }
};

/**
 * Modifies `begin` to skip past any whitespace.
 *
 * Example:
 *     skip_whitespace(begin, end);
 */
inline void skip_whitespace(const char *&rbegin, const char *end)
{
  using namespace std;
  const char *begin = rbegin;
  while (begin < end && DYND_ISSPACE(*begin)) {
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
  using namespace std;
  const char *begin = rbegin;
  while (begin < end && DYND_ISSPACE(*begin)) {
    ++begin;
  }

  // Comments
  if (begin < end && *begin == '#') {
    const char *line_end = (const char *)memchr(begin, '\n', end - begin);
    if (line_end == NULL) {
      begin = end;
    }
    else {
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
  using namespace std;
  const char *begin = rbegin;
  if (begin < end && DYND_ISSPACE(*begin)) {
    ++begin;
    while (begin < end && DYND_ISSPACE(*begin)) {
      ++begin;
    }
    rbegin = begin;
    return true;
  }
  else {
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
inline bool parse_token(const char *&rbegin, const char *end, const char(&token)[N])
{
  const char *begin = rbegin;
  skip_whitespace(begin, end);
  if (N - 1 <= end - begin && memcmp(begin, token, N - 1) == 0) {
    rbegin = begin + N - 1;
    return true;
  }
  else {
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
  }
  else {
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
inline bool parse_token_no_ws(const char *&rbegin, const char *end, const char(&token)[N])
{
  const char *begin = rbegin;
  if (N - 1 <= end - begin && memcmp(begin, token, N - 1) == 0) {
    rbegin = begin + N - 1;
    return true;
  }
  else {
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
  }
  else {
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
bool parse_name_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin, const char *&out_strend);

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
bool parse_alpha_name_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin, const char *&out_strend);

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
bool parse_doublequote_string_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin,
                                    const char *&out_strend, bool &out_escaped);

/**
 * Unescapes the string provided in the byte range into the
 * output string as UTF-8. Typically used with the
 * ``parse_doublequote_string_no_ws`` function.
 */
DYND_API void unescape_string(const char *strbegin, const char *strend, std::string &out);

/**
 * Without skipping whitespace, parses a range of bytes following
 * the JSON number grammar, returning its range of bytes.
 */
DYND_API bool parse_json_number_no_ws(const char *&rbegin, const char *end, const char *&out_nbegin,
                                      const char *&out_nend);

/**
 * Does an exact comparison of a byte range to a string literal.
 */
template <int N>
inline bool compare_range_to_literal(const char *begin, const char *end, const char(&token)[N])
{
  return (end - begin) == N - 1 && !memcmp(begin, token, N - 1);
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
inline bool parse_unsigned_int_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin,
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
    }
    else if (*begin == '0') {
      if (begin + 1 < end && ('0' <= *(begin + 1) && *(begin + 1) <= '9')) {
        // Don't match leading zeros
        return false;
      }
      else {
        out_strbegin = begin;
        out_strend = begin + 1;
        rbegin = begin + 1;
        return true;
      }
    }
    else {
      return false;
    }
  }
  else {
    return false;
  }
}

/**
 * Without skipping whitespace, parses a signed integer.
 *
 * Example:
 *     // Match a two digit month
 *     const char *match_begin, *match_end;
 *     if (parse_int_no_ws(begin, end, match_begin, match_end) {
 *         // Convert to int, process
 *     } else {
 *         // Couldn't match unsigned integer
 *     }
 */
inline bool parse_int_no_ws(const char *&rbegin, const char *end, const char *&out_strbegin, const char *&out_strend)
{
  const char *begin = rbegin;
  const char *saved_begin = begin;
  parse_token(begin, end, '-');
  if (parse_unsigned_int_no_ws(begin, end, out_strbegin, out_strend)) {
    out_strbegin = saved_begin;
    rbegin = begin;
    return true;
  }
  return false;
}

float checked_float64_to_float32(double value, assign_error_mode errmode);

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
DYND_API bool parse_2digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

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
DYND_API bool parse_1or2digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

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
DYND_API bool parse_4digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

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
DYND_API bool parse_6digit_int_no_ws(const char *&rbegin, const char *end, int &out_val);

/**
 * Converts a string containing only an integer (no leading or
 * trailing space, etc) into an intptr_t, raising an exception if
 * there are problems.
 */
DYND_API intptr_t checked_string_to_intptr(const char *begin, const char *end);

/**
 * Converts a string containing only an integer (no leading or
 * trailing space, etc) into an int64, raising an exception if
 * there are problems.
 */
DYND_API int64_t checked_string_to_int64(const char *begin, const char *end);

/**
 * Converts a string containing only an unsigned integer (no leading or
 * trailing space, etc), ignoring any problems.
 */
template <typename T>
std::enable_if_t<is_unsigned<T>::value, T> parse(const char *begin, const char *end, nocheck_t DYND_UNUSED(nocheck))
{
  T result = 0;
  while (begin < end) {
    char c = *begin;
    if ('0' <= c && c <= '9') {
      result = (result * 10u) + static_cast<uint32_t>(c - '0');
    }
    else if (c == 'e' || c == 'E') {
      // Accept "1e5", "1e+5" integers with a positive exponent,
      // a subset of floating point syntax. Note that "1.2e1"
      // is not accepted as the value 12 by this code.
      ++begin;
      if (begin < end && *begin == '+') {
        ++begin;
      }
      if (begin < end) {
        int exponent = 0;
        // Accept any number of zeros followed by at most
        // two digits. Anything greater would overflow.
        while (begin < end && *begin == '0') {
          ++begin;
        }
        if (begin < end && '0' <= *begin && *begin <= '9') {
          exponent = *begin++ - '0';
        }
        if (begin < end && '0' <= *begin && *begin <= '9') {
          exponent = (10 * exponent) + (*begin++ - '0');
        }
        if (begin == end) {
          // Apply the exponent in a naive way
          for (int i = 0; i < exponent; ++i) {
            result = result * 10u;
          }
        }
      }
      break;
    }
    else {
      break;
    }
    ++begin;
  }
  return result;
}

template <typename T>
T parse(const std::string &s, nocheck_t nocheck)
{
  return parse<T>(s.data(), s.data() + s.size(), nocheck);
}

template <typename T>
T parse(const string &s, nocheck_t nocheck)
{
  return parse<T>(s.begin(), s.end(), nocheck);
}

/**
 * Converts a string containing (no leading or trailing space, etc) to a type T,
 * setting the output over flow or bad parse flags if there are problems.
 */
template <typename T>
T parse(const char *begin, const char *end)
{
  T result = 0, prev_result = 0;
  if (begin == end) {
    raise_string_cast_error(ndt::make_type<T>(), begin, end);
  }
  while (begin < end) {
    char c = *begin;
    if ('0' <= c && c <= '9') {
      result = (result * 10u) + static_cast<T>(c - '0');
      if (result < prev_result) {
        raise_string_cast_overflow_error(ndt::make_type<T>(), begin, end);
      }
    }
    else {
      if (c == '.') {
        // Accept ".", ".0" with trailing decimal zeros as well
        ++begin;
        while (begin < end && *begin == '0') {
          ++begin;
        }
        if (begin == end) {
          break;
        }
      }
      else if (c == 'e' || c == 'E') {
        // Accept "1e5", "1e+5" integers with a positive exponent,
        // a subset of floating point syntax. Note that "1.2e1"
        // is not accepted as the value 12 by this code.
        ++begin;
        if (begin < end && *begin == '+') {
          ++begin;
        }
        if (begin < end) {
          int exponent = 0;
          // Accept any number of zeros followed by at most
          // two digits. Anything greater would overflow.
          while (begin < end && *begin == '0') {
            ++begin;
          }
          if (begin < end && '0' <= *begin && *begin <= '9') {
            exponent = *begin++ - '0';
          }
          if (begin < end && '0' <= *begin && *begin <= '9') {
            exponent = (10 * exponent) + (*begin++ - '0');
          }
          if (begin == end) {
            prev_result = result;
            // Apply the exponent in a naive way, but with
            // overflow checking
            for (int i = 0; i < exponent; ++i) {
              result = result * 10u;
              if (result < prev_result) {
                raise_string_cast_overflow_error(ndt::make_type<T>(), begin, end);
              }
              prev_result = result;
            }
            return result;
          }
        }
      }
      raise_string_cast_error(ndt::make_type<T>(), begin, end);
      break;
    }
    ++begin;
    prev_result = result;
  }
  return result;
}

template <typename T>
T parse(const std::string &s)
{
  return parse<T>(s.data(), s.data() + s.size());
}

template <typename T>
T parse(const string &s)
{
  return parse<T>(s.begin(), s.end());
}

/**
 * Converts a string containing an boolean (no leading or trailing space)
 * into a bool, using the specified error mode to handle errors.
 * If ``option`` is true, writes to option[bool].
 *
 * \param out_bool  The address of the bool or option[bool].
 * \param begin  The start of the UTF8 string buffer.
 * \param end  The end of the UTF8 string buffer.
 * \param option  If true, treat it as option[int] instead of just int.
 * \param errmode  The error handling mode.
 */
DYND_API void string_to_bool(char *out_bool, const char *begin, const char *end, bool option,
                             assign_error_mode errmode);

/**
 * Returns true if the string provided matches an option[T] missing value token,
 * such as "", "NA", "NULL", "null", "None".
 */
DYND_API bool matches_option_type_na_token(const char *begin, const char *end);

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
struct DYND_API named_value {
  const char *name;
  int value;
  DYND_CONSTEXPR named_value(const char *name_, int value_) : name(name_), value(value_) {}
};

double checked_string_to_float64(const char *begin, const char *end, assign_error_mode error_mode);

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
inline bool parse_ci_alpha_str_named_value_no_ws(const char *&rbegin, const char *end, named_value(&nvt)[N],
                                                 int &out_value)
{
  using namespace std;
  // TODO: Could specialize two implementations based on the size of N,
  //       for small N do a linear search, big N do a binary search.

  const char *begin = rbegin;
  const char *strbegin, *strend;
  if (!parse_alpha_name_no_ws(begin, end, strbegin, strend)) {
    return false;
  }
  int strfirstchar = DYND_TOLOWER(*strbegin);
  // Search through the named value table for a matching string
  for (int i = 0; i < N; ++i) {
    const char *name = nvt[i].name;
    // Compare the first character
    if (*name++ == strfirstchar) {
      const char *strptr = strbegin + 1;
      // Compare the rest of the characters
      while (*name != '\0' && strptr < strend && *name == DYND_TOLOWER(*strptr)) {
        ++name;
        ++strptr;
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

/**
 * A utility for checking whether a value would overflow when converted to
 * the specified type.
 *
 * if (overflow_check<int8_t>(unsigned_value, is_negative)) {
 *   throw overflow_error(...);
 * }
 *
 * if (overflow_check<int8_t>(signed_value)) {
 *   throw overflow_error(...);
 * }
 */
template <class T>
struct overflow_check;
template <>
struct overflow_check<int8_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fULL) != 0 && !(negative && value == 0x80ULL);
  }
};
template <>
struct overflow_check<int16_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffULL) != 0 && !(negative && value == 0x8000ULL);
  }
};
template <>
struct overflow_check<int32_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
  }
};
template <>
struct overflow_check<int128> {
  inline static bool is_overflow(uint128 value, bool negative)
  {
    return (value.m_hi & ~0x7fffffffffffffffULL) != 0 &&
           !(negative && value.m_hi == 0x8000000000000000ULL && value.m_lo == 0ULL);
  }
};
template <>
struct overflow_check<long> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
#if INT_MAX == LONG_MAX
    return (value & ~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
#else
    return (value & ~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
#endif
  }
#if INT_MAX == LONG_MAX
  inline static bool is_overflow(int64_t value) { return (value < -0x80000000LL) || (value > 0x7fffffffLL); }
#else
  inline static bool is_overflow(int64_t DYND_UNUSED(value)) { return false; }
#endif
};
template <>
struct overflow_check<long long> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
  }
  inline static bool is_overflow(int64_t DYND_UNUSED(value)) { return false; }
};
template <>
struct overflow_check<uint8_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffULL) != 0; }
};
template <>
struct overflow_check<uint16_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffffULL) != 0; }
};
template <>
struct overflow_check<uint32_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffffffffULL) != 0; }
};
template <>
struct overflow_check<uint64_t> {
  inline static bool is_overflow(uint64_t DYND_UNUSED(value)) { return false; }
};

DYND_API void parse_int64(int64_t &res, const char *begin, const char *end);

DYND_API void parse_uint64(uint64_t &res, const char *begin, const char *end);

DYND_API int parse_double(double &res, const char *begin, const char *end);

inline static double make_double_nan(bool negative)
{
  union {
    uint64_t i;
    double d;
  } nan;
  nan.i = negative ? 0xfff8000000000000ULL : 0x7ff8000000000000ULL;
  return nan.d;
}

template <class T>
void assign_signed_int_value(char *out_int, uint64_t uvalue, bool &negative, bool &overflow, bool &badparse)
{
  overflow = overflow || overflow_check<T>::is_overflow(uvalue, negative);
  if (!overflow && !badparse) {
    *reinterpret_cast<T *>(out_int) =
        static_cast<T>(negative ? -static_cast<int64_t>(uvalue) : static_cast<int64_t>(uvalue));
  }
}

inline void assign_signed_int128_value(char *out_int, uint128 uvalue, bool &negative, bool &overflow, bool &badparse)
{
  overflow = overflow || overflow_check<int128>::is_overflow(uvalue, negative);
  if (!overflow && !badparse) {
    *reinterpret_cast<int128 *>(out_int) = negative ? -static_cast<int128>(uvalue) : static_cast<int128>(uvalue);
  }
}

/**
 * Converts a string containing only a floating point number into
 * a float64/C double.
 */
template <assign_error_mode ErrorMode>
double checked_string_to_float64(const char *begin, const char *end)
{
  bool negative = false;
  const char *pos = begin;
  if (pos < end && *pos == '-') {
    negative = true;
    ++pos;
  }
  // First check for various NaN/Inf inputs
  size_t size = end - pos;
  if (size == 3) {
    if ((pos[0] == 'N' || pos[0] == 'n') && (pos[1] == 'A' || pos[1] == 'a') && (pos[2] == 'N' || pos[2] == 'n')) {
      return make_double_nan(negative);
    }
    else if ((pos[0] == 'I' || pos[0] == 'i') && (pos[1] == 'N' || pos[1] == 'n') && (pos[2] == 'F' || pos[2] == 'f')) {
      return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }
  }
  else if (size == 7) {
    if ((pos[0] == '1') && (pos[1] == '.') && (pos[2] == '#') && (pos[3] == 'Q' || pos[3] == 'q') &&
        (pos[4] == 'N' || pos[4] == 'n') && (pos[5] == 'A' || pos[5] == 'a') && (pos[6] == 'N' || pos[6] == 'n')) {
      return make_double_nan(negative);
    }
  }
  else if (size == 6) {
    if ((pos[0] == '1') && (pos[1] == '.') && (pos[2] == '#')) {
      if ((pos[3] == 'I' || pos[3] == 'i') && (pos[4] == 'N' || pos[4] == 'n') && (pos[5] == 'D' || pos[5] == 'd')) {
        return make_double_nan(negative);
      }
      else if ((pos[3] == 'I' || pos[3] == 'i') && (pos[4] == 'N' || pos[4] == 'n') &&
               (pos[5] == 'F' || pos[5] == 'f')) {
        return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
      }
    }
  }
  else if (size == 8) {
    if ((pos[0] == 'I' || pos[0] == 'i') && (pos[1] == 'N' || pos[1] == 'n') && (pos[2] == 'F' || pos[2] == 'f') &&
        (pos[3] == 'I' || pos[3] == 'i') && (pos[4] == 'N' || pos[4] == 'n') && (pos[5] == 'I' || pos[5] == 'i') &&
        (pos[6] == 'T' || pos[6] == 't') && (pos[7] == 'Y' || pos[7] == 'y')) {
      return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }
  }

  // TODO: use http://www.netlib.org/fp/dtoa.c
  char *end_ptr;
  std::string s(begin, end);
  double value = strtod(s.c_str(), &end_ptr);
  if (ErrorMode != assign_error_nocheck && (size_t)(end_ptr - s.c_str()) != s.size()) {
    std::stringstream ss;
    ss << "parse error converting string ";
    print_escaped_utf8_string(ss, begin, end);
    ss << " to float64";
    throw std::invalid_argument(ss.str());
  }

  return value;
}

template <class T>
void assign_unsigned_int_value(char *out_int, uint64_t uvalue, bool &negative, bool &overflow, bool &badparse)
{
  overflow = overflow || negative || overflow_check<T>::is_overflow(uvalue);
  if (!overflow && !badparse) {
    *reinterpret_cast<T *>(out_int) = static_cast<T>(uvalue);
  }
}

/**
 * Converts a string containing a number (no leading or trailing space)
 * into a Num with the specified builtin type id, using the specified error
 * mode to handle errors. If ``option`` is true, writes to option[Num].
 *
 * \param out  The address of the Num or option[Num].
 * \param tid  The type id of the Num.
 * \param begin  The start of the UTF8 string buffer.
 * \param end  The end of the UTF8 string buffer.
 * \param option  If true, treat it as option[Num] instead of just Num.
 * \param errmode  The error handling mode.
 */
inline void string_to_number(char *out, type_id_t tid, const char *begin, const char *end, bool option,
                             assign_error_mode errmode)
{

  uint64_t uvalue;
  const char *saved_begin = begin;
  bool negative = false, overflow = false, badparse = false;

  if (option && matches_option_type_na_token(begin, end)) {
    switch (tid) {
    case int8_type_id:
      *reinterpret_cast<int8_t *>(out) = DYND_INT8_NA;
      return;
    case int16_type_id:
      *reinterpret_cast<int16_t *>(out) = DYND_INT16_NA;
      return;
    case int32_type_id:
      *reinterpret_cast<int32_t *>(out) = DYND_INT32_NA;
      return;
    case int64_type_id:
      *reinterpret_cast<int64_t *>(out) = DYND_INT64_NA;
      return;
    case int128_type_id:
      *reinterpret_cast<int128 *>(out) = DYND_INT128_NA;
      return;
    case float16_type_id:
      *reinterpret_cast<uint16_t *>(out) = DYND_FLOAT16_NA_AS_UINT;
      return;
    case float32_type_id:
      *reinterpret_cast<uint32_t *>(out) = DYND_FLOAT32_NA_AS_UINT;
      return;
    case float64_type_id:
      *reinterpret_cast<uint64_t *>(out) = DYND_FLOAT64_NA_AS_UINT;
      return;
    case complex_float32_type_id:
      reinterpret_cast<uint32_t *>(out)[0] = DYND_FLOAT32_NA_AS_UINT;
      reinterpret_cast<uint32_t *>(out)[1] = DYND_FLOAT32_NA_AS_UINT;
      return;
    case complex_float64_type_id:
      reinterpret_cast<uint64_t *>(out)[0] = DYND_FLOAT64_NA_AS_UINT;
      reinterpret_cast<uint64_t *>(out)[1] = DYND_FLOAT64_NA_AS_UINT;
      return;
    default:
      break;
    }
    std::stringstream ss;
    ss << "No NA value has been configured for option[" << ndt::type(tid) << "]";
    throw type_error(ss.str());
  }

  if (begin < end && *begin == '-') {
    negative = true;
    ++begin;
  }
  if (errmode != assign_error_nocheck) {
    switch (tid) {
    case int8_type_id:
      uvalue = parse<uint64_t>(begin, end);
      assign_signed_int_value<int8_t>(out, uvalue, negative, overflow, badparse);
      break;
    case int16_type_id:
      uvalue = parse<uint64_t>(begin, end);
      assign_signed_int_value<int16_t>(out, uvalue, negative, overflow, badparse);
      break;
    case int32_type_id:
      uvalue = parse<uint64_t>(begin, end);
      assign_signed_int_value<int32_t>(out, uvalue, negative, overflow, badparse);
      break;
    case int64_type_id:
      uvalue = parse<uint64_t>(begin, end);
      assign_signed_int_value<int64_t>(out, uvalue, negative, overflow, badparse);
      break;
    case int128_type_id: {
      uint128 buvalue = parse<uint128>(begin, end);
      assign_signed_int128_value(out, buvalue, negative, overflow, badparse);
      break;
    }
    case uint8_type_id:
      uvalue = parse<uint64_t>(begin, end);
      negative = negative && (uvalue != 0);
      assign_unsigned_int_value<uint8_t>(out, uvalue, negative, overflow, badparse);
      break;
    case uint16_type_id:
      uvalue = parse<uint64_t>(begin, end);
      negative = negative && (uvalue != 0);
      assign_unsigned_int_value<uint16_t>(out, uvalue, negative, overflow, badparse);
      break;
    case uint32_type_id:
      uvalue = parse<uint64_t>(begin, end);
      negative = negative && (uvalue != 0);
      assign_unsigned_int_value<uint32_t>(out, uvalue, negative, overflow, badparse);
      break;
    case uint64_type_id:
      uvalue = parse<uint64_t>(begin, end);
      negative = negative && (uvalue != 0);
      overflow = overflow || negative;
      if (!overflow && !badparse) {
        *reinterpret_cast<uint64_t *>(out) = uvalue;
      }
      break;
    case uint128_type_id: {
      uint128 buvalue = parse<uint128>(begin, end);
      negative = negative && (buvalue != 0);
      overflow = overflow || negative;
      if (!overflow && !badparse) {
        *reinterpret_cast<uint128 *>(out) = buvalue;
      }
      break;
    }
    case float16_type_id: {
      double value = checked_string_to_float64(saved_begin, end, errmode);
      *reinterpret_cast<uint16_t *>(out) = float16(value).bits();
      break;
    }
    case float32_type_id: {
      double value = checked_string_to_float64(saved_begin, end, errmode);
      *reinterpret_cast<float *>(out) = checked_float64_to_float32(value, errmode);
      break;
    }
    case float64_type_id: {
      *reinterpret_cast<double *>(out) = checked_string_to_float64(saved_begin, end, errmode);
      break;
    }
    default: {
      std::stringstream ss;
      ss << "cannot parse number, got invalid type id " << tid;
      throw std::runtime_error(ss.str());
    }
    }
    if (overflow) {
      std::stringstream ss;
      ss << "overflow converting string ";
      print_escaped_utf8_string(ss, begin, end);
      ss << " to ";
      if (option) {
        ss << "?";
      }
      ss << tid;
      throw std::overflow_error(ss.str());
    }
    else if (badparse) {
      std::stringstream ss;
      ss << "parse error converting string ";
      print_escaped_utf8_string(ss, begin, end);
      ss << " to ";
      if (option) {
        ss << "?";
      }
      ss << tid;
      throw std::invalid_argument(ss.str());
    }
  }
  else {
    // errmode == assign_error_nocheck
    switch (tid) {
    case int8_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<int8_t *>(out) =
          static_cast<int8_t>(negative ? -static_cast<int64_t>(uvalue) : static_cast<int64_t>(uvalue));
      break;
    case int16_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<int16_t *>(out) =
          static_cast<int16_t>(negative ? -static_cast<int64_t>(uvalue) : static_cast<int64_t>(uvalue));
      break;
    case int32_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<int32_t *>(out) =
          static_cast<int32_t>(negative ? -static_cast<int64_t>(uvalue) : static_cast<int64_t>(uvalue));
      break;
    case int64_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<int64_t *>(out) = negative ? -static_cast<int64_t>(uvalue) : static_cast<int64_t>(uvalue);
      break;
    case int128_type_id: {
      uint128 buvalue = parse<uint128>(begin, end, nocheck);
      *reinterpret_cast<int128 *>(out) = negative ? -static_cast<int128>(buvalue) : static_cast<int128>(buvalue);
      break;
    }
    case uint8_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<uint8_t *>(out) = static_cast<uint8_t>(negative ? 0 : uvalue);
      break;
    case uint16_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<uint16_t *>(out) = static_cast<uint16_t>(negative ? 0 : uvalue);
      break;
    case uint32_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<uint32_t *>(out) = static_cast<uint32_t>(negative ? 0 : uvalue);
      break;
    case uint64_type_id:
      uvalue = parse<uint64_t>(begin, end, nocheck);
      *reinterpret_cast<uint64_t *>(out) = negative ? 0 : uvalue;
      break;
    case uint128_type_id: {
      uint128 buvalue = parse<uint128>(begin, end, nocheck);
      *reinterpret_cast<uint128 *>(out) = negative ? static_cast<uint128>(0) : buvalue;
      break;
    }
    case float16_type_id: {
      double value = checked_string_to_float64(saved_begin, end, errmode);
      *reinterpret_cast<uint16_t *>(out) = float16(value).bits();
      break;
    }
    case float32_type_id: {
      double value = checked_string_to_float64(saved_begin, end, errmode);
      *reinterpret_cast<float *>(out) = checked_float64_to_float32(value, errmode);
      break;
    }
    case float64_type_id: {
      *reinterpret_cast<double *>(out) = checked_string_to_float64(saved_begin, end, errmode);
      break;
    }
    default: {
      std::stringstream ss;
      ss << "cannot parse number, got invalid type id " << tid;
      throw std::runtime_error(ss.str());
    }
    }
  }
}

inline double checked_string_to_float64(const char *begin, const char *end, assign_error_mode error_mode)
{
  switch (error_mode) {
  case assign_error_nocheck:
    return checked_string_to_float64<assign_error_nocheck>(begin, end);
  case assign_error_inexact:
    return checked_string_to_float64<assign_error_inexact>(begin, end);
  case assign_error_overflow:
    return checked_string_to_float64<assign_error_overflow>(begin, end);
  case assign_error_fractional:
    return checked_string_to_float64<assign_error_fractional>(begin, end);
  default:
  case assign_error_default:
    return checked_string_to_float64<assign_error_default>(begin, end);
  }
}

} // namespace dynd
