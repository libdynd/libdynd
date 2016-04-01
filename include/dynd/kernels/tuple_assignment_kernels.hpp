//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/types/struct_type.hpp>

namespace dynd {
namespace nd {

  struct tuple_unary_op_item {
    size_t child_kernel_offset;
    size_t dst_data_offset;
    size_t src_data_offset;
  };

  struct tuple_unary_op_ck : nd::base_strided_kernel<tuple_unary_op_ck, 1> {
    std::vector<tuple_unary_op_item> m_fields;

    ~tuple_unary_op_ck() {
      for (size_t i = 0; i < m_fields.size(); ++i) {
        get_child(m_fields[i].child_kernel_offset)->destroy();
      }
    }

    void single(char *dst, char *const *src) {
      const tuple_unary_op_item *fi = &m_fields[0];
      intptr_t field_count = m_fields.size();
      kernel_prefix *child;
      kernel_single_t child_fn;

      for (intptr_t i = 0; i < field_count; ++i) {
        const tuple_unary_op_item &item = fi[i];
        child = get_child(item.child_kernel_offset);
        child_fn = child->get_function<kernel_single_t>();
        char *child_src = src[0] + item.src_data_offset;
        child_fn(child, dst + item.dst_data_offset, &child_src);
      }
    }
  };

} // namespace dynd::nd

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

} // namespace dynd
