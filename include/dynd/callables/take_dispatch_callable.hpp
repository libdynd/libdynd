//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/take_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID>
  class take_callable;

  template <>
  class take_callable<bool_id> : public base_callable {
  public:
    take_callable() : base_callable(ndt::type("(Any, Fixed * bool) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.push_back([](call_node *&node, kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                      intptr_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        typedef nd::masked_take_ck self_type;

        intptr_t ckb_offset = ckb->size();
        ckb->emplace_back<masked_take_ck>(kernreq);
        node = next(node);

        self_type *self = ckb->get_at<self_type>(ckb_offset);
        self->m_dst_meta = dst_arrmeta;

        const char *src0_el_meta = src_arrmeta[0] + sizeof(size_stride_t);
        intptr_t src0_dim_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;
        self->m_src0_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->stride;

        intptr_t mask_dim_size = reinterpret_cast<const size_stride_t *>(src_arrmeta[1])->dim_size;
        self->m_mask_stride = reinterpret_cast<const size_stride_t *>(src_arrmeta[1])->stride;

        if (src0_dim_size != mask_dim_size) {
          std::stringstream ss;
          ss << "masked take arrfunc: source data and mask have different sizes, ";
          ss << src0_dim_size << " and " << mask_dim_size;
          throw std::invalid_argument(ss.str());
        }
        self->m_dim_size = src0_dim_size;

        // Create the child element assignment ckernel
        node->instantiate(node, ckb, kernel_request_strided, dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), 1,
                          &src0_el_meta);
      });

      ndt::type src0_element_tp = src_tp[0].extended<ndt::base_dim_type>()->get_element_type();

      nd::array error_mode = assign_error_default;
      assign->resolve(this, nullptr, cg, src0_element_tp, 1, &src0_element_tp, 1, &error_mode, tp_vars);

      return ndt::make_type<ndt::var_dim_type>(src0_element_tp);
    }
  };

  class indexed_take_callable : public base_callable {
  public:
    indexed_take_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.push_back([=](call_node *&node, kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                       intptr_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        intptr_t self_offset = ckb->size();
        ckb->emplace_back<indexed_take_ck>(kernreq);
        node = next(node);

        indexed_take_ck *self = ckb->get_at<indexed_take_ck>(self_offset);
        std::cout << "here" << std::endl;

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
        if (index_el_tp.get_id() != type_id_of<intptr_t>::value) {
          std::stringstream ss;
          ss << "indexed take arrfunc: index type should be intptr, not ";
          ss << index_el_tp;
          throw type_error(ss.str());
        }

        // Create the child element assignment ckernel
        node->instantiate(node, ckb, kernel_request_single, dst_el_meta, 1, &src0_el_meta);
      });

      ndt::type src0_element_tp = src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type();

      nd::array error_mode = assign_error_default;
      assign->resolve(this, nullptr, cg, src0_element_tp, 1, &src0_element_tp, 1, &error_mode, tp_vars);

      if (src_tp[1].get_id() == var_dim_id) {
        return ndt::var_dim_type::make(src0_element_tp);
      } else {
        return ndt::make_fixed_dim(src_tp[1].get_dim_size(NULL, NULL), src0_element_tp);
      }
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      std::cout << "here" << std::endl;

      intptr_t self_offset = ckb->size();
      ckb->emplace_back<indexed_take_ck>(kernreq);
      node = next(node);

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
      if (index_el_tp.get_id() != type_id_of<intptr_t>::value) {
        std::stringstream ss;
        ss << "indexed take arrfunc: index type should be intptr, not ";
        ss << index_el_tp;
        throw type_error(ss.str());
      }

      // Create the child element assignment ckernel
      node->instantiate(node, ckb, kernel_request_single, dst_el_meta, 1, &src0_el_meta);
    }
  };

  class take_dispatch_callable : public base_callable {
  public:
    take_dispatch_callable() : base_callable(ndt::type("(Dims... * T, N * Ix) -> R * T")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
      if (mask_el_tp.get_id() == bool_id) {
        static callable f = make_callable<take_callable<bool_id>>();
        return f->resolve(this, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      } else if (mask_el_tp.get_id() == type_id_of<intptr_t>::value) {
        static callable f = make_callable<indexed_take_callable>();
        return f->resolve(this, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      } else {
        std::stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp << ", need bool or intptr";
        throw std::invalid_argument(ss.str());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
