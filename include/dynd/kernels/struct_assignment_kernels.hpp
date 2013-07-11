//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtypes/struct_type.hpp>
#include <dynd/dtypes/cstruct_type.hpp>

namespace dynd {

/**
 * Gets a kernel which copies values of the same struct dtype.
 *
 * \param val_struct_dt  The struct-kind dtype of both source and destination values.
 */
size_t make_struct_identical_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& val_struct_dt,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Gets a kernel which converts from one struct to another.
 *
 * \param dst_struct_dt  The struct-kind dtype of the destination.
 * \param src_struct_dt  The struct-kind dtype of the source.
 * \param errmode  The error handling mode of the assignment.
 */
size_t make_struct_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_struct_dt, const char *dst_metadata,
                const ndt::type& src_struct_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
