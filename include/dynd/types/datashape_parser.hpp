//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

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
DYNDT_API ndt::type type_from_datashape(const char *datashape_begin, const char *datashape_end);

inline ndt::type type_from_datashape(const std::string &datashape) {
  return type_from_datashape(datashape.data(), datashape.data() + datashape.size());
}

inline ndt::type type_from_datashape(const char *datashape) {
  return type_from_datashape(datashape, datashape + strlen(datashape));
}

template <int N>
inline ndt::type type_from_datashape(const char (&datashape)[N]) {
  return type_from_datashape(datashape, datashape + N - 1);
}

/**
 * Low level parsing function for parsing the argument list passed to a datashape type constructor.
 *
 * Returns a NULL nd::array if there is no arg list, and an nd::array with datashape
 *   "{pos: N * arg, kw: {name: arg, ...}}" otherwise.
 */
DYNDT_API nd::buffer parse_type_constr_args(const char *&rbegin, const char *end,
                                            std::map<std::string, ndt::type> &symtable);

DYNDT_API nd::buffer parse_type_constr_args(const std::string &str);

/**
 * An internal exception for communicating a parse error location to the outermost parse function, which can then
 * extract the line and column number for the final error message. Not used to communicate errors to outside code.
 */
class DYNDT_API internal_datashape_parse_error {
  /** The position of the error within the buffer being parsed */
  const char *m_position;
  const std::string m_message;

public:
  internal_datashape_parse_error(const char *position, const std::string &message)
      : m_position(position), m_message(message) {}
  internal_datashape_parse_error(const char *position, std::string &&message)
      : m_position(position), m_message(std::move(message)) {}
  ~internal_datashape_parse_error() {}
  const char *get_position() const { return m_position; }
  const std::string &get_message() const { return m_message; }
};

} // namespace dynd
