//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/take_kernel.hpp>

namespace dynd {
namespace nd {

  class masked_take_callable : public base_callable {
  public:
    masked_take_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &tp_vars) {
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
      assign->instantiate(NULL, ckb, dst_el_tp, dst_el_meta, 1, &src0_el_tp, &src0_el_meta, kernel_request_strided, 1,
                          &error_mode, tp_vars);
    }
  };

  class indexed_take_callable : public base_callable {
  public:
    indexed_take_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &cg, const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &tp_vars) {
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
      assign->instantiate(NULL, ckb, dst_el_tp, dst_el_meta, 1, &src0_el_tp, &src0_el_meta, kernel_request_single, 1,
                          &error_mode, tp_vars);
    }
  };

  class take_dispatch_callable : public base_callable {
  public:
    take_dispatch_callable() : base_callable(ndt::type("(Dims... * T, N * Ix) -> R * T")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), call_graph &DYND_UNUSED(cg), const ndt::type &dst_tp,
                      size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return dst_tp;
    }

    void resolve_dst_type(char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                          const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
      if (mask_el_tp.get_id() == bool_id) {
        dst_tp = ndt::var_dim_type::make(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
      } else if (mask_el_tp.get_id() == (type_id_t)type_id_of<intptr_t>::value) {
        if (src_tp[1].get_id() == var_dim_id) {
          dst_tp = ndt::var_dim_type::make(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        } else {
          dst_tp = ndt::make_fixed_dim(src_tp[1].get_dim_size(NULL, NULL),
                                       src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        }
      } else {
        std::stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp << ", need bool or intptr";
        throw std::invalid_argument(ss.str());
      }
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                     intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
      if (mask_el_tp.get_id() == bool_id) {
        callable f = make_callable<masked_take_callable>();
        f->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
        return;
      } else if (mask_el_tp.get_id() == (type_id_t)type_id_of<intptr_t>::value) {
        callable f = make_callable<indexed_take_callable>();
        f->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
        return;
      } else {
        std::stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp << ", need bool or intptr";
        throw std::invalid_argument(ss.str());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
