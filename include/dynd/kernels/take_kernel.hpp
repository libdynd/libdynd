//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/shape_tools.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {

  struct DYND_API masked_take_ck : base_strided_kernel<masked_take_ck, 2> {
    ndt::type m_dst_tp;
    const char *m_dst_meta;
    intptr_t m_dim_size, m_src0_stride, m_mask_stride;

    ~masked_take_ck() { get_child()->destroy(); }

    void single(char *dst, char *const *src)
    {
      kernel_prefix *child = get_child();
      kernel_strided_t child_fn = child->get_function<kernel_strided_t>();
      char *src0 = src[0];
      char *mask = src[1];
      intptr_t dim_size = m_dim_size, src0_stride = m_src0_stride, mask_stride = m_mask_stride;
      // Start with the dst matching the dim size. (Maybe better to
      // do smaller? This means no resize required in the loop.)
      ndt::var_dim_type::data_type *vdd = reinterpret_cast<ndt::var_dim_type::data_type *>(dst);
      vdd->begin = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->blockref->alloc(dim_size);
      vdd->size = dim_size;
      char *dst_ptr = vdd->begin;
      intptr_t dst_stride = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)->stride;
      intptr_t dst_count = 0;
      intptr_t i = 0;
      while (i < dim_size) {
        // Run of false
        for (; i < dim_size && *mask == 0; src0 += src0_stride, mask += mask_stride, ++i) {
        }
        // Run of true
        intptr_t i_saved = i;
        for (; i < dim_size && *mask != 0; mask += mask_stride, ++i) {
        }
        // Copy the run of true
        if (i > i_saved) {
          intptr_t run_count = i - i_saved;
          child_fn(child, dst_ptr, dst_stride, &src0, &src0_stride, run_count);
          dst_ptr += run_count * dst_stride;
          src0 += run_count * src0_stride;
          dst_count += run_count;
        }
      }

      vdd->begin = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_meta)
                       ->blockref->resize(vdd->begin, dst_count);
      vdd->size = dst_count;
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      typedef nd::masked_take_ck self_type;

      intptr_t ckb_offset = ckb->size();
      ckb->emplace_back<self_type>(kernreq);
      self_type *self = ckb->get_at<self_type>(ckb_offset);

      if (dst_tp.get_id() != var_dim_id) {
        std::stringstream ss;
        ss << "masked take arrfunc: could not process type " << dst_tp;
        ss << " as a var dimension";
        throw type_error(ss.str());
      }
      self->m_dst_tp = dst_tp;
      self->m_dst_meta = dst_arrmeta;
      ndt::type dst_el_tp = self->m_dst_tp.extended<ndt::var_dim_type>()->get_element_type();
      const char *dst_el_meta = self->m_dst_meta + sizeof(ndt::var_dim_type::metadata_type);

      intptr_t src0_dim_size, mask_dim_size;
      ndt::type src0_el_tp, mask_el_tp;
      const char *src0_el_meta, *mask_el_meta;
      if (!src_tp[0].get_as_strided(src_arrmeta[0], &src0_dim_size, &self->m_src0_stride, &src0_el_tp, &src0_el_meta)) {
        std::stringstream ss;
        ss << "masked take arrfunc: could not process type " << src_tp[0];
        ss << " as a strided dimension";
        throw type_error(ss.str());
      }
      if (!src_tp[1].get_as_strided(src_arrmeta[1], &mask_dim_size, &self->m_mask_stride, &mask_el_tp, &mask_el_meta)) {
        std::stringstream ss;
        ss << "masked take arrfunc: could not process type " << src_tp[1];
        ss << " as a strided dimension";
        throw type_error(ss.str());
      }
      if (src0_dim_size != mask_dim_size) {
        std::stringstream ss;
        ss << "masked take arrfunc: source data and mask have different sizes, ";
        ss << src0_dim_size << " and " << mask_dim_size;
        throw std::invalid_argument(ss.str());
      }
      self->m_dim_size = src0_dim_size;
      if (mask_el_tp.get_id() != bool_id) {
        std::stringstream ss;
        ss << "masked take arrfunc: mask type should be bool, not ";
        ss << mask_el_tp;
        throw type_error(ss.str());
      }

      // Create the child element assignment ckernel
      nd::array error_mode = assign_error_default;
      assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, dst_el_tp, dst_el_meta, 1, &src0_el_tp,
                                 &src0_el_meta, kernel_request_strided, 1, &error_mode, tp_vars);
    }
  };

  /**
   * CKernel which does an indexed take operation. The child ckernel
   * should be a single unary operation.
   */
  struct DYND_API indexed_take_ck : base_strided_kernel<indexed_take_ck, 2> {
    intptr_t m_dst_dim_size, m_dst_stride, m_index_stride;
    intptr_t m_src0_dim_size, m_src0_stride;

    ~indexed_take_ck() { get_child()->destroy(); }

    void single(char *dst, char *const *src)
    {
      kernel_prefix *child = get_child();
      kernel_single_t child_fn = child->get_function<kernel_single_t>();
      char *src0 = src[0];
      const char *index = src[1];
      intptr_t dst_dim_size = m_dst_dim_size, src0_dim_size = m_src0_dim_size, dst_stride = m_dst_stride,
               src0_stride = m_src0_stride, index_stride = m_index_stride;
      for (intptr_t i = 0; i < dst_dim_size; ++i) {
        intptr_t ix = *reinterpret_cast<const intptr_t *>(index);
        // Handle Python-style negative index, bounds checking
        ix = apply_single_index(ix, src0_dim_size, NULL);
        // Copy one element at a time
        char *child_src0 = src0 + ix * src0_stride;
        child_fn(child, dst, &child_src0);
        dst += dst_stride;
        index += index_stride;
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                            intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t self_offset = ckb->size();
      ckb->emplace_back<indexed_take_ck>(kernreq);
      indexed_take_ck *self = ckb->get_at<indexed_take_ck>(self_offset);

      ndt::type dst_el_tp;
      const char *dst_el_meta;
      if (!dst_tp.get_as_strided(dst_arrmeta, &self->m_dst_dim_size, &self->m_dst_stride, &dst_el_tp, &dst_el_meta)) {
        std::stringstream ss;
        ss << "indexed take arrfunc: could not process type " << dst_tp;
        ss << " as a strided dimension";
        throw type_error(ss.str());
      }

      intptr_t index_dim_size;
      ndt::type src0_el_tp, index_el_tp;
      const char *src0_el_meta, *index_el_meta;
      if (!src_tp[0].get_as_strided(src_arrmeta[0], &self->m_src0_dim_size, &self->m_src0_stride, &src0_el_tp,
                                    &src0_el_meta)) {
        std::stringstream ss;
        ss << "indexed take arrfunc: could not process type " << src_tp[0];
        ss << " as a strided dimension";
        throw type_error(ss.str());
      }
      if (!src_tp[1].get_as_strided(src_arrmeta[1], &index_dim_size, &self->m_index_stride, &index_el_tp,
                                    &index_el_meta)) {
        std::stringstream ss;
        ss << "take arrfunc: could not process type " << src_tp[1];
        ss << " as a strided dimension";
        throw type_error(ss.str());
      }
      if (self->m_dst_dim_size != index_dim_size) {
        std::stringstream ss;
        ss << "indexed take arrfunc: index data and dest have different sizes, ";
        ss << index_dim_size << " and " << self->m_dst_dim_size;
        throw std::invalid_argument(ss.str());
      }
      if (index_el_tp.get_id() != (type_id_t)type_id_of<intptr_t>::value) {
        std::stringstream ss;
        ss << "indexed take arrfunc: index type should be intptr, not ";
        ss << index_el_tp;
        throw type_error(ss.str());
      }

      // Create the child element assignment ckernel
      nd::array error_mode = assign_error_default;
      assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, dst_el_tp, dst_el_meta, 1, &src0_el_tp,
                                 &src0_el_meta, kernel_request_single, 1, &error_mode, tp_vars);
    }
  };

  struct DYND_API take_ck : base_kernel<take_ck> {
    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
      if (mask_el_tp.get_id() == bool_id) {
        dst_tp = ndt::var_dim_type::make(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
      }
      else if (mask_el_tp.get_id() == (type_id_t)type_id_of<intptr_t>::value) {
        if (src_tp[1].get_id() == var_dim_id) {
          dst_tp = ndt::var_dim_type::make(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        }
        else {
          dst_tp = ndt::make_fixed_dim(src_tp[1].get_dim_size(NULL, NULL),
                                       src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        }
      }
      else {
        std::stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp << ", need bool or intptr";
        throw std::invalid_argument(ss.str());
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                            const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
      if (mask_el_tp.get_id() == bool_id) {
        masked_take_ck::instantiate(NULL, NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                                    kwds, tp_vars);
        return;
      }
      else if (mask_el_tp.get_id() == (type_id_t)type_id_of<intptr_t>::value) {
        indexed_take_ck::instantiate(NULL, NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                                     kwds, tp_vars);
        return;
      }
      else {
        std::stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp << ", need bool or intptr";
        throw std::invalid_argument(ss.str());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
