//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <dynd/irange.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

// Forward declaration of object class, for broadcast_error
namespace ndt {
  class type;
} // namespace ndt
namespace nd {
  class array;
} // namespace nd

// Avoid subclassing std::exception to not have
// to export std::exception across dll boundaries.
class DYNDT_API dynd_exception {
protected:
  std::string m_message, m_what;

public:
  dynd_exception() {}
  dynd_exception(const char *exception_name, const std::string &msg)
      : m_message(msg), m_what(std::string() + exception_name + ": " + msg)
  {
  }

  virtual const char *message() const throw();
  virtual const char *what() const throw();

  virtual ~dynd_exception() throw();
};

/**
 * An exception for an index out of bounds
 */
class DYNDT_API too_many_indices : public dynd_exception {
public:
  /**
   * An exception for when too many indices are provided in
   * an indexing operation (nindex > ndim).
   */
  too_many_indices(const ndt::type &dt, intptr_t nindices, intptr_t ndim);

  virtual ~too_many_indices() throw();
};

class DYNDT_API index_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't within bounds for
   * the specified axis of the given shape
   */
  index_out_of_bounds(intptr_t i, size_t axis, intptr_t ndim, const intptr_t *shape);
  index_out_of_bounds(intptr_t i, size_t axis, const std::vector<intptr_t> &shape);
  index_out_of_bounds(intptr_t i, intptr_t dimension_size);

  virtual ~index_out_of_bounds() throw();
};

class DYNDT_API axis_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't a valid axis
   * for the number of dimensions.
   */
  axis_out_of_bounds(size_t i, intptr_t ndim);

  virtual ~axis_out_of_bounds() throw();
};

/**
 * An exception for a range out of bounds.
 */
class DYNDT_API irange_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't within bounds for
   * the specified axis of the given shape
   */
  irange_out_of_bounds(const irange &i, size_t axis, intptr_t ndim, const intptr_t *shape);
  irange_out_of_bounds(const irange &i, size_t axis, const std::vector<intptr_t> &shape);
  irange_out_of_bounds(const irange &i, intptr_t dimension_size);

  virtual ~irange_out_of_bounds() throw();
};

/**
 * An exception for zero division
 */
class DYNDT_API zero_division_error : public dynd_exception {
public:
  zero_division_error(const std::string &msg) : dynd_exception("zero division error", msg) {}
  virtual ~zero_division_error() throw();
};

/**
 * An exception for errors related to types.
 */
class DYNDT_API type_error : public dynd_exception {
public:
  type_error(const char *exception_name, const std::string &msg) : dynd_exception(exception_name, msg) {}
  type_error(const std::string &msg) : dynd_exception("type error", msg) {}

  virtual ~type_error() throw();
};

/**
 * An exception for an invalid type ID.
 */
class DYNDT_API invalid_id : public type_error {
public:
  invalid_id(int type_id);

  virtual ~invalid_id() throw();
};

/**
 * An exception for when input can't be decoded
 */
class DYNDT_API string_decode_error : public dynd_exception {
  std::string m_bytes;
  string_encoding_t m_encoding;

public:
  string_decode_error(const char *begin, const char *end, string_encoding_t encoding);

  virtual ~string_decode_error() throw();

  const std::string &bytes() const;

  string_encoding_t encoding() const;
};

/**
 * An exception for when a codepoint can't encode to
 * the destination.
 */
class DYNDT_API string_encode_error : public dynd_exception {
  uint32_t m_cp;
  string_encoding_t m_encoding;

public:
  string_encode_error(uint32_t cp, string_encoding_t encoding);

  virtual ~string_encode_error() throw();

  uint32_t cp() const;

  string_encoding_t encoding() const;
};

enum comparison_type_t {
  /**
   * A less than operation suitable for sorting
   * (one of a < b or b < a must be true when a != b).
   */
  comparison_type_sorting_less,
  /** Standard comparisons */
  comparison_type_less,
  comparison_type_less_equal,
  comparison_type_equal,
  comparison_type_not_equal,
  comparison_type_greater_equal,
  comparison_type_greater
};

/**
 * An exception for when two dynd types cannot be compared
 * a particular comparison operator.
 */
class DYNDT_API not_comparable_error : public dynd_exception {
public:
  not_comparable_error(const ndt::type &lhs, const ndt::type &rhs, comparison_type_t comptype);

  virtual ~not_comparable_error() throw();
};

#ifdef DYND_CUDA

/**
 * An exception for errors from the CUDA runtime.
 */
class DYNDT_API cuda_runtime_error : public std::runtime_error {
  cudaError_t m_error;

public:
  cuda_runtime_error(cudaError_t error);
};

inline void cuda_throw_if_not_success(cudaError_t error = cudaPeekAtLastError())
{
  if (error != cudaSuccess) {
    throw cuda_runtime_error(error);
  }
}

#endif

} // namespace dynd
