//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_ASSIGN_HPP_
#define _DYND__DTYPE_ASSIGN_HPP_

#include <utility>
#include <iostream>

namespace dynd {

namespace ndt {
  class type;
}

namespace nd {
  class array;
}

namespace eval {
  struct eval_context;
  extern eval_context default_eval_context;
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

std::ostream& operator<<(std::ostream& o, assign_error_mode errmode);

/** If 'src' can always be cast to 'dst' with no loss of information */
bool is_lossless_assignment(const ndt::type &dst_tp, const ndt::type &src_tp);

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
void typed_data_copy(const ndt::type &tp, const char *dst_arrmeta,
                     char *dst_data, const char *src_arrmeta,
                     const char *src_data);

/** 
 * Assign one element where src and dst may have different types.
 * Requires that the data be aligned. To assign unaligned data,
 * use ndt::make_unaligned().
 */
void
typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                  char *dst_data, const ndt::type &src_tp,
                  const char *src_arrmeta, const char *src_data,
                  const eval::eval_context *ectx = &eval::default_eval_context);

/** 
 * Assign one element where src and dst may have different types.
 * Requires that the data be aligned. To assign unaligned data,
 * use ndt::make_unaligned().
 */
void
typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta,
                  char *dst_data, const nd::array &src_arr,
                  const eval::eval_context *ectx = &eval::default_eval_context);

} // namespace dynd

#endif // _DYND__DTYPE_ASSIGN_HPP_
