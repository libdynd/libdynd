//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <utility>
#include <iostream>
#include <dynd/visibility.hpp>

namespace dynd {

namespace ndt {
  class DYND_API type;
}

namespace nd {
  class DYND_API array;
}

namespace eval {
  struct DYND_API eval_context;
  extern DYND_API eval_context default_eval_context;
} // namespace eval

/**
 * An enumeration for the error checks during assignment.
 */
enum assign_error_mode {
  /** No error checking during assignment */
  assign_error_nocheck,
  /** Check overflow, but allow precision loss. Checks loss of imaginary
     component  */
  assign_error_overflow,
  /** Overflow and loss of fractional part (for float -> int) checking */
  assign_error_fractional,
  /** Overflow and floating point precision loss checking */
  assign_error_inexact,
  /** Use the mode specified in the eval_context */
  assign_error_default
};

DYND_API std::ostream& operator<<(std::ostream& o, assign_error_mode errmode);

/** If 'src' can always be cast to 'dst' with no loss of information */
DYND_API bool is_lossless_assignment(const ndt::type &dst_tp,
                                     const ndt::type &src_tp);

/**
 * Copies a value from one location to another, where the types of the source
 * and destination are the same.
 *
 * \param tp  The type for the copy operation.
 * \param dst_arrmeta  The arrmeta of the destination.
 * \param dst_data  The data where the destination element is stored.
 * \param src_arrmeta  The arrmeta of the source.
 * \param src_data  The data where the source element is stored.
 */
DYND_API void typed_data_copy(const ndt::type &tp, const char *dst_arrmeta,
                              char *dst_data, const char *src_arrmeta,
                              const char *src_data);

/**
 * Assign one element where src and dst may have different types.
 * Requires that the data be aligned. To assign unaligned data,
 * use ndt::make_unaligned().
 */
DYND_API void
typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                  char *dst_data, const ndt::type &src_tp,
                  const char *src_arrmeta, const char *src_data,
                  const eval::eval_context *ectx = &eval::default_eval_context);

/**
 * Assign one element where src and dst may have different types.
 * Requires that the data be aligned. To assign unaligned data,
 * use ndt::make_unaligned().
 */
DYND_API void
typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                  char *dst_data, const nd::array &src_arr,
                  const eval::eval_context *ectx = &eval::default_eval_context);

} // namespace dynd
