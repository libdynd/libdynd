//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {
namespace parse {

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
    explicit saved_begin_state(const char *&begin) : m_begin(begin), m_saved_begin(begin), m_succeeded(false)
    {
    }

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

    inline bool fail()
    {
      return false;
    }

    inline const char *saved_begin() const
    {
      return m_saved_begin;
    }
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
    virtual ~parse_error() throw()
    {
    }
    const char *get_position() const
    {
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
    using namespace std;
    const char *begin = rbegin;
    if (begin < end && DYND_ISSPACE(*begin)) {
      ++begin;
      while (begin < end && DYND_ISSPACE(*begin)) {
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
  inline bool parse_token(const char *&rbegin, const char *end, const char (&token)[N])
  {
    const char *begin = rbegin;
    skip_whitespace(begin, end);
    if (N - 1 <= end - begin && memcmp(begin, token, N - 1) == 0) {
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
  inline bool parse_token_no_ws(const char *&rbegin, const char *end, const char (&token)[N])
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
  inline bool compare_range_to_literal(const char *begin, const char *end, const char (&token)[N])
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
   * Converts a string containing only an unsigned integer (no leading or
   * trailing space, etc) into a uint64, setting the output over flow or
   * bad parse flags if there are problems.
   */
  DYND_API uint64_t
  checked_string_to_uint64(const char *begin, const char *end, bool &out_overflow, bool &out_badparse);

  /**
   * Converts a string containing only an unsigned integer (no leading or
   * trailing space, etc) into a uint128, setting the output over flow or
   * bad parse flags if there are problems.
   */
  DYND_API uint128
  checked_string_to_uint128(const char *begin, const char *end, bool &out_overflow, bool &out_badparse);

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
   * trailing space, etc) into a uint64, ignoring any problems.
   */
  DYND_API uint64_t unchecked_string_to_uint64(const char *begin, const char *end);

  /**
   * Converts a string containing only an unsigned integer (no leading or
   * trailing space, etc) into a uint128, ignoring any problems.
   */
  DYND_API uint128 unchecked_string_to_uint128(const char *begin, const char *end);

  /**
   * Converts a string containing only a floating point number into
   * a float64/C double.
   */
  DYND_API double checked_string_to_float64(const char *begin, const char *end, assign_error_mode errmode);

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
  DYND_API void string_to_number(char *out, type_id_t tid, const char *begin, const char *end, bool option,
                                 assign_error_mode errmode);

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
    DYND_CONSTEXPR named_value(const char *name_, int value_) : name(name_), value(value_)
    {
    }
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
  inline bool parse_ci_alpha_str_named_value_no_ws(const char *&rbegin, const char *end, named_value (&nvt)[N],
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
  struct overflow_check<signed char> {
    inline static bool is_overflow(uint64_t value, bool negative)
    {
      return (value & ~0x7fULL) != 0 && !(negative && value == 0x80ULL);
    }
    inline static bool is_overflow(int64_t value)
    {
      return (value < -0x80) || (value > 0x7f);
    }
  };
  template <>
  struct overflow_check<short> {
    inline static bool is_overflow(uint64_t value, bool negative)
    {
      return (value & ~0x7fffULL) != 0 && !(negative && value == 0x8000ULL);
    }
    inline static bool is_overflow(int64_t value)
    {
      return (value < -0x8000) || (value > 0x7fff);
    }
  };
  template <>
  struct overflow_check<int> {
    inline static bool is_overflow(uint64_t value, bool negative)
    {
      return (value & ~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
    }
    inline static bool is_overflow(int64_t value)
    {
      return (value < -0x80000000LL) || (value > 0x7fffffffLL);
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
    inline static bool is_overflow(int64_t value)
    {
      return (value < -0x80000000LL) || (value > 0x7fffffffLL);
    }
#else
    inline static bool is_overflow(int64_t DYND_UNUSED(value))
    {
      return false;
    }
#endif
  };
  template <>
  struct overflow_check<long long> {
    inline static bool is_overflow(uint64_t value, bool negative)
    {
      return (value & ~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
    }
    inline static bool is_overflow(int64_t DYND_UNUSED(value))
    {
      return false;
    }
  };
  template <>
  struct overflow_check<int128> {
    inline static bool is_overflow(const uint128 &value, bool negative)
    {
      return (value.m_hi & ~0x7fffffffffffffffULL) != 0 &&
             !(negative && value.m_hi == 0x8000000000000000ULL && value.m_lo == 0ULL);
    }
  };
  template <>
  struct overflow_check<uint8_t> {
    inline static bool is_overflow(uint64_t value)
    {
      return (value & ~0xffULL) != 0;
    }
  };
  template <>
  struct overflow_check<uint16_t> {
    inline static bool is_overflow(uint64_t value)
    {
      return (value & ~0xffffULL) != 0;
    }
  };
  template <>
  struct overflow_check<uint32_t> {
    inline static bool is_overflow(uint64_t value)
    {
      return (value & ~0xffffffffULL) != 0;
    }
  };

} // namespace dynd::parse

DYND_API int parse_int64(int64_t &res, const char *begin, const char *end);

DYND_API int parse_uint64(uint64_t &res, const char *begin, const char *end);

DYND_API int parse_double(double &res, const char *begin, const char *end);

} // namespace dynd
