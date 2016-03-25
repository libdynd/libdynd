//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/mean_kernel.hpp>

namespace dynd {
namespace nd {

  class mean_callable : public base_callable {
    ndt::type m_tp;

  public:
    struct data_type {
      char *sum_data;
      char *compound_div_data;
    };

    mean_callable(const ndt::type &tp) : base_callable(nd::sum::get().get_array_type()), m_tp(tp) {}

    char *data_init(const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                    const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      char *data = reinterpret_cast<char *>(new data_type());
      reinterpret_cast<data_type *>(data)->sum_data =
          nd::sum::get().get()->data_init(dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      reinterpret_cast<data_type *>(data)->compound_div_data =
          nd::compound_div::get().get()->data_init(dst_tp, 1, &m_tp, 0, NULL, tp_vars);

      return data;
    }

    void resolve_dst_type(char *data, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                          const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      nd::sum::get().get()->resolve_dst_type(reinterpret_cast<data_type *>(data)->sum_data, dst_tp, nsrc, src_tp, nkwd,
                                             kwds, tp_vars);
    }

    void instantiate(char *static_data, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t mean_offset = ckb->size();
      ckb->emplace_back<mean_kernel>(kernreq, src_tp[0].get_size(src_arrmeta[0]));

      nd::sum::get().get()->instantiate(nd::sum::get().get()->static_data(),
                                        reinterpret_cast<data_type *>(data)->sum_data, ckb, dst_tp, dst_arrmeta, nsrc,
                                        src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);

      mean_kernel *self = ckb->get_at<mean_kernel>(mean_offset);
      self->compound_div_offset = ckb->size();
      nd::compound_div::get().get()->instantiate(
          nd::compound_div::get().get()->static_data(), reinterpret_cast<data_type *>(data)->compound_div_data, ckb,
          dst_tp, dst_arrmeta, 1, reinterpret_cast<ndt::type *>(static_data), NULL, kernreq, nkwd, kwds, tp_vars);

      delete reinterpret_cast<data_type *>(data);
    }
  };

} // namespace dynd::nd
} // namespace dynd
