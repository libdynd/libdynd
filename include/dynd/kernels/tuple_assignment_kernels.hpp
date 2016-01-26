//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/types/struct_type.hpp>

namespace dynd {

/**
 * Creates a ckernel which applies the provided callable to a
 * series of fields at fixed offsets within a tuple or struct.
 *
 * This function applies a single callable to all the fields.
 *
 * \param af  The callable to apply to each field.
 * \param ckb  The ckernel_builder.
 * \param ckb_offset  The offset within the ckernel builder at which to make the
 *                    ckernel.
 * \param field_count  The number of fields.
 * \param dst_offsets  An array with an offset into dst memory for each field.
 * \param dst_tp  An array with one dst type for each field.
 * \param dst_arrmeta  An array with one dst arrmeta for each field.
 * \param src_offsets  An array with an offset into src memory for each field.
 * \param src_tp  An array with one src type for each field.
 * \param src_arrmeta  An array with one src arrmeta for each field.
 * \param kernreq  What kind of ckernel to create (single, strided).
 */
DYND_API void make_tuple_unary_op_ckernel(const nd::base_callable *af, const ndt::callable_type *af_tp,
                                          nd::kernel_builder *ckb, intptr_t field_count, const uintptr_t *dst_offsets,
                                          const ndt::type *dst_tp, const char *const *dst_arrmeta,
                                          const uintptr_t *src_offsets, const ndt::type *src_tp,
                                          const char *const *src_arrmeta, kernel_request_t kernreq);

/**
 * Creates a ckernel which applies the provided callables to a
 * series of fields at fixed offsets within a tuple or struct.
 *
 * This function accepts an array of callables, one for each field.
 *
 * \param af  The callable to apply to each field.
 * \param ckb  The ckernel_builder.
 * \param ckb_offset  The offset within the ckernel builder at which to make the
 *                    ckernel.
 * \param field_count  The number of fields.
 * \param dst_offsets  An array with an offset into dst memory for each field.
 * \param dst_tp  An array with one dst type for each field.
 * \param dst_arrmeta  An array with one dst arrmeta for each field.
 * \param src_offsets  An array with an offset into src memory for each field.
 * \param src_tp  An array with one src type for each field.
 * \param src_arrmeta  An array with one src arrmeta for each field.
 * \param kernreq  What kind of ckernel to create (single, strided).
 */
DYND_API void make_tuple_unary_op_ckernel(const nd::base_callable *const *af, const ndt::callable_type *const *af_tp,
                                          nd::kernel_builder *ckb, intptr_t field_count, const uintptr_t *dst_offsets,
                                          const ndt::type *dst_tp, const char *const *dst_arrmeta,
                                          const uintptr_t *src_offsets, const ndt::type *src_tp,
                                          const char *const *src_arrmeta, kernel_request_t kernreq);

/**
 * Gets a kernel which copies values of the same tuple or struct type.
 *
 * \param val_tup_tp  The struct-kind type of both source and destination
 *                    values.
 */
DYND_API void make_tuple_identical_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &val_tup_tp,
                                                     const char *dst_arrmeta, const char *src_arrmeta,
                                                     kernel_request_t kernreq);

/**
 * Gets a kernel which converts from one tuple/struct to another.
 * This does the assignment by copying values based on position
 * within the tuple or struct, ignoring any names.
 *
 * \param dst_tuple_tp  The tuple/struct-kind dtype of the destination.
 * \param src_tuple_tp  The tuple/struct-kind dtype of the source.
 */
DYND_API void make_tuple_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tuple_tp,
                                           const char *dst_arrmeta, const ndt::type &src_tuple_tp,
                                           const char *src_arrmeta, kernel_request_t kernreq);

/**
 * Gets a kernel which broadcasts the source value to all the fields
 * of the destination tuple/struct.
 */
DYND_API void make_broadcast_to_tuple_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tuple_tp,
                                                        const char *dst_arrmeta, const ndt::type &src_tp,
                                                        const char *src_arrmeta, kernel_request_t kernreq);

} // namespace dynd
