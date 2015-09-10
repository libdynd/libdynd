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
#include <dynd/kernels/comparison_kernels.hpp>

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
class DYND_API dynd_exception {
protected:
  std::string m_message, m_what;

public:
  dynd_exception() {}
  dynd_exception(const char *exception_name, const std::string &msg)
      : m_message(msg), m_what(std::string() + exception_name + ": " + msg)
  {
  }

  virtual const char *message() const throw() { return m_message.c_str(); }
  virtual const char *what() const throw() { return m_what.c_str(); }

  virtual ~dynd_exception() throw() {}
};

/**
 * An exception for various kinds of broadcast errors.
 */
class DYND_API broadcast_error : public dynd_exception {
public:
  broadcast_error(const std::string &m);

  /**
   * An exception for when 'src' doesn't broadcast to 'dst'
   */
  broadcast_error(intptr_t dst_ndim, const intptr_t *dst_shape,
                  intptr_t src_ndim, const intptr_t *src_shape);

  /**
   * An exception for when 'src' doesn't broadcast to 'dst'
   */
  broadcast_error(const nd::array &dst, const nd::array &src);

  /**
   * An exception for when a number of input operands can't be broadcast
   * together.
   */
  broadcast_error(intptr_t ninputs, const nd::array *inputs);

  broadcast_error(const ndt::type &dst_tp, const char *dst_arrmeta,
                  const ndt::type &src_tp, const char *src_arrmeta);

  broadcast_error(const ndt::type &dst_tp, const char *dst_arrmeta,
                  const char *src_name);

  /**
   * For when broadcasting is occurring in a context where
   * much of the global information about the broadcasting isn't
   * available, e.g. broadcasting a var dim inside a kernel.
   */
  broadcast_error(intptr_t dst_size, intptr_t src_size, const char *dst_name,
                  const char *src_name);

  virtual ~broadcast_error() throw() {}
};

/**
 * An exception for an index out of bounds
 */
class DYND_API too_many_indices : public dynd_exception {
public:
  /**
   * An exception for when too many indices are provided in
   * an indexing operation (nindex > ndim).
   */
  too_many_indices(const ndt::type &dt, intptr_t nindices, intptr_t ndim);

  virtual ~too_many_indices() throw() {}
};

class DYND_API index_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't within bounds for
   * the specified axis of the given shape
   */
  index_out_of_bounds(intptr_t i, size_t axis, intptr_t ndim,
                      const intptr_t *shape);
  index_out_of_bounds(intptr_t i, size_t axis,
                      const std::vector<intptr_t> &shape);
  index_out_of_bounds(intptr_t i, intptr_t dimension_size);

  virtual ~index_out_of_bounds() throw() {}
};

class DYND_API axis_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't a valid axis
   * for the number of dimensions.
   */
  axis_out_of_bounds(size_t i, intptr_t ndim);

  virtual ~axis_out_of_bounds() throw() {}
};

/**
 * An exception for a range out of bounds.
 */
class DYND_API irange_out_of_bounds : public dynd_exception {
public:
  /**
   * An exception for when 'i' isn't within bounds for
   * the specified axis of the given shape
   */
  irange_out_of_bounds(const irange &i, size_t axis, intptr_t ndim,
                       const intptr_t *shape);
  irange_out_of_bounds(const irange &i, size_t axis,
                       const std::vector<intptr_t> &shape);
  irange_out_of_bounds(const irange &i, intptr_t dimension_size);

  virtual ~irange_out_of_bounds() throw() {}
};

/**
 * An exception for errors related to types.
 */
class DYND_API type_error : public dynd_exception {
public:
  type_error(const char *exception_name, const std::string &msg)
      : dynd_exception(exception_name, msg)
  {
  }
  type_error(const std::string &msg) : dynd_exception("type error", msg) {}

  virtual ~type_error() throw() {}
};

/**
 * An exception for an invalid type ID.
 */
class DYND_API invalid_type_id : public type_error {
public:
  invalid_type_id(int type_id);

  virtual ~invalid_type_id() throw() {}
};

/**
 * An exception for when input can't be decoded
 */
class DYND_API string_decode_error : public dynd_exception {
  std::string m_bytes;
  string_encoding_t m_encoding;

public:
  string_decode_error(const char *begin, const char *end,
                      string_encoding_t encoding);

  virtual ~string_decode_error() throw() {}

  const std::string &bytes() const { return m_bytes; }

  string_encoding_t encoding() const { return m_encoding; }
};

/**
 * An exception for when a codepoint can't encode to
 * the destination.
 */
class DYND_API string_encode_error : public dynd_exception {
  uint32_t m_cp;
  string_encoding_t m_encoding;

public:
  string_encode_error(uint32_t cp, string_encoding_t encoding);

  virtual ~string_encode_error() throw() {}

  uint32_t cp() const { return m_cp; }

  string_encoding_t encoding() const { return m_encoding; }
};

/**
 * An exception for when two dynd types cannot be compared
 * a particular comparison operator.
 */
class DYND_API not_comparable_error : public dynd_exception {
public:
  not_comparable_error(const ndt::type &lhs, const ndt::type &rhs,
                       comparison_type_t comptype);

  virtual ~not_comparable_error() throw() {}
};

#ifdef DYND_CUDA

/**
 * An exception for errors from the CUDA runtime.
 */
class DYND_API cuda_runtime_error : public std::runtime_error {
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
