//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/parse_kernel.hpp>

namespace dynd {
namespace nd {
  namespace json {

    template <type_id_t ResID>
    class parse_callable : public default_instantiable_callable<parse_kernel<ResID>> {
    public:
      parse_callable()
          : default_instantiable_callable<parse_kernel<ResID>>(
                ndt::callable_type::make(ResID, {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}
    };

    template <>
    class parse_callable<option_id> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::callable_type::make(option_id, {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t ckb_offset = ckb->size();
        intptr_t self_offset = ckb_offset;
        ckb->emplace_back<parse_kernel<option_id>>(kernreq);
        ckb_offset = ckb->size();
        node = next(node);

        assign_na->instantiate(node, data, ckb, dst_tp, dst_arrmeta, 0, nullptr, nullptr,
                               kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();

        ckb->get_at<parse_kernel<option_id>>(self_offset)->parse_offset = ckb_offset - self_offset;
        dynamic_parse->instantiate(node, data, ckb, dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta,
                                   nsrc, src_tp, src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    };

    template <>
    class parse_callable<struct_id> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::callable_type::make(struct_id, {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        intptr_t ckb_offset = ckb->size();
        size_t field_count = dst_tp.extended<ndt::struct_type>()->get_field_count();
        const std::vector<uintptr_t> &arrmeta_offsets = dst_tp.extended<ndt::struct_type>()->get_arrmeta_offsets();

        intptr_t self_offset = ckb_offset;
        ckb->emplace_back<parse_kernel<struct_id>>(kernreq, dst_tp, field_count,
                                                   dst_tp.extended<ndt::struct_type>()->get_data_offsets(dst_arrmeta));
        node = next(node);


        ckb_offset = ckb->size();

        for (size_t i = 0; i < field_count; ++i) {
          ckb->get_at<parse_kernel<struct_id>>(self_offset)->child_offsets[i] = ckb_offset - self_offset;
          dynamic_parse->instantiate(node, data, ckb, dst_tp.extended<ndt::struct_type>()->get_field_type(i),
                                     dst_arrmeta + arrmeta_offsets[i], nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds,
                                     tp_vars);
          ckb_offset = ckb->size();
        }
      }
    };

    template <>
    class parse_callable<fixed_dim_id> : public base_callable {
    public:
      parse_callable()
          : base_callable(
                ndt::callable_type::make(fixed_dim_id, {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        ckb->emplace_back<parse_kernel<fixed_dim_id>>(kernreq, dst_tp,
                                                      reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size,
                                                      reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride);
        node = next(node);


        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        dynamic_parse->instantiate(node, data, ckb, child_dst_tp,
                                   dst_arrmeta + sizeof(ndt::fixed_dim_type::metadata_type), nsrc, src_tp, src_arrmeta,
                                   kernreq, nkwd, kwds, tp_vars);
      }
    };

    template <>
    class parse_callable<var_dim_id> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::callable_type::make(var_dim_id, {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                       const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                       kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &tp_vars) {
        ckb->emplace_back<parse_kernel<var_dim_id>>(
            kernreq, dst_tp, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref,
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride);
        node = next(node);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::var_dim_type>()->get_element_type();
        dynamic_parse->instantiate(node, data, ckb, child_dst_tp,
                                   dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), nsrc, src_tp, src_arrmeta,
                                   kernreq, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::json
} // namespace dynd::nd
} // namespace dynd
