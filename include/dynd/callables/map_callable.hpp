//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>

namespace dynd {
namespace nd {

  class map_callable : public base_callable {
    callable m_child;

  public:
    map_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(call_node *DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                     intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      auto dst_sd = dst_tp.extended<ndt::tuple_type>();
      auto src_sd = src_tp[0].extended<ndt::tuple_type>();
      intptr_t field_count = dst_sd->get_field_count();

      const std::vector<uintptr_t> &src_arrmeta_offsets = src_sd->get_arrmeta_offsets();
      shortvector<const char *> src_fields_arrmeta(field_count);
      for (intptr_t i = 0; i != field_count; ++i) {
        src_fields_arrmeta[i] = src_arrmeta[0] + src_arrmeta_offsets[i];
      }

      const std::vector<uintptr_t> &dst_arrmeta_offsets = dst_sd->get_arrmeta_offsets();
      shortvector<const char *> dst_fields_arrmeta(field_count);
      for (intptr_t i = 0; i != field_count; ++i) {
        dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
      }

      auto dst_offsets = dst_sd->get_data_offsets(dst_arrmeta);
      auto src_offsets = src_sd->get_data_offsets(src_arrmeta[0]);
      auto src_field_tp = src_sd->get_field_types();
      auto dst_field_tp = dst_sd->get_field_types();

      intptr_t self_offset = ckb->size();
      ckb->emplace_back<nd::tuple_unary_op_ck>(kernreq);
      nd::tuple_unary_op_ck *self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
      self->m_fields.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
        nd::tuple_unary_op_item &field = self->m_fields[i];
        field.child_kernel_offset = ckb->size() - self_offset;
        field.dst_data_offset = dst_offsets[i];
        field.src_data_offset = src_offsets[i];
        nd::array error_mode = ndt::traits<assign_error_mode>::na();
        m_child->instantiate(NULL, NULL, ckb, dst_field_tp[i], dst_fields_arrmeta[i], 1, &src_field_tp[i],
                             &src_fields_arrmeta[i], kernel_request_single, 1, &error_mode,
                             std::map<std::string, ndt::type>());
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
