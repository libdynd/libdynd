//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_

#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>

namespace dynd {

/**
 * Gets a kernel which copies values of the same struct type.
 *
 * \param val_struct_tp  The struct-kind type of both source and destination values.
 */
size_t make_struct_identical_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& val_struct_tp,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Gets a kernel which converts from one struct to another.
 *
 * \param dst_struct_tp  The struct-kind dtype of the destination.
 * \param src_struct_tp  The struct-kind dtype of the source.
 * \param errmode  The error handling mode of the assignment.
 */
size_t make_struct_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_struct_tp, const char *dst_metadata,
                const ndt::type& src_struct_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Gets a kernel which broadcasts the source value to all the fields
 * of the destination struct.
 */
size_t make_broadcast_to_struct_assignment_kernel(
                ckernel_builder *out_ckb, size_t ckb_offset,
                const ndt::type& dst_struct_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
