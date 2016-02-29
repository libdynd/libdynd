//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API assign : declfunc<assign> {
    static callable make();
    static callable &get();
  } assign;

} // namespace dynd::nd

/**
 * Creates an assignment kernel for one data value from the
 * src type/arrmeta to the dst type/arrmeta. This adds the
 * kernel at the 'ckb_offset' position in 'ckb's data, as part
 * of a hierarchy matching the dynd type's hierarchy.
 *
 * This function should always be called with this == dst_tp first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 *
 * \param ckb  The ckernel_builder being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_tp  The destination dynd type.
 * \param dst_arrmeta  Arrmeta for the destination data.
 * \param src_tp  The source dynd type.
 * \param src_arrmeta  Arrmeta for the source data
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'ckb' immediately after the
 *           created kernel.
 */
DYND_API void make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                     const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                     const eval::eval_context *ectx);

inline void make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                   const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                   const eval::eval_context *ectx)
{
  make_assignment_kernel(ckb, dst_tp, dst_arrmeta, *src_tp, *src_arrmeta, kernreq, ectx);
}

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
DYND_API void make_pod_typed_data_assignment_kernel(nd::kernel_builder *ckb, size_t data_size, size_t data_alignment,
                                                    kernel_request_t kernreq);

} // namespace dynd
