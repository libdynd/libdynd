//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_

#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Creates a ckernel which applies the provided arrfunc to a
 * series of fields at fixed offsets within a tuple or struct.
 *
 * This function applies a single arrfunc to all the fields.
 *
 * \param af  The arrfunc to apply to each field.
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
 * \param ectx  The evaluation context.
 */
intptr_t make_tuple_unary_op_ckernel(
    const arrfunc_type_data *af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t field_count, const uintptr_t *dst_offsets,
    const ndt::type *dst_tp, const char *const *dst_arrmeta,
    const uintptr_t *src_offsets, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Creates a ckernel which applies the provided arrfuncs to a
 * series of fields at fixed offsets within a tuple or struct.
 *
 * This function accepts an array of arrfuncs, one for each field.
 *
 * \param af  The arrfunc to apply to each field.
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
 * \param ectx  The evaluation context.
 */
intptr_t make_tuple_unary_op_ckernel(
    const arrfunc_type_data *const *af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t field_count, const uintptr_t *dst_offsets,
    const ndt::type *dst_tp, const char *const *dst_arrmeta,
    const uintptr_t *src_offsets, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Gets a kernel which copies values of the same struct type.
 *
 * \param val_struct_tp  The struct-kind type of both source and destination
 *values.
 */
size_t make_struct_identical_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &val_struct_tp,
    const char *dst_arrmeta, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Gets a kernel which converts from one struct to another.
 *
 * \param dst_struct_tp  The struct-kind dtype of the destination.
 * \param src_struct_tp  The struct-kind dtype of the source.
 */
size_t make_struct_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_struct_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);

/**
 * Gets a kernel which broadcasts the source value to all the fields
 * of the destination struct.
 */
size_t make_broadcast_to_struct_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_struct_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__STRUCT_ASSIGNMENT_KERNELS_HPP_
