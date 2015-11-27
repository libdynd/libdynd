//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct DYND_API strided_rolling_ck : base_kernel<strided_rolling_ck, 1> {
      intptr_t m_window_size;
      intptr_t m_dim_size, m_dst_stride, m_src_stride;
      size_t m_window_op_offset;
      arrmeta_holder m_src_winop_meta;

      ~strided_rolling_ck()
      {
        // The NA filler
        get_child()->destroy();
        // The window op
        get_child(m_window_op_offset)->destroy();
      }

      void single(char *dst, char *const *src);
    };

    struct DYND_API var_rolling_ck : base_kernel<var_rolling_ck, 1> {
      intptr_t m_window_size;
      intptr_t m_src_stride, m_src_offset;
      ndt::type m_dst_tp;
      const char *m_dst_meta;
      size_t m_window_op_offset;

      ~var_rolling_ck()
      {
        // The NA filler
        get_child()->destroy();
        // The window op
        get_child(m_window_op_offset)->destroy();
      }

      void single(char *dst, char *const *src);
    };

    struct DYND_API rolling_ck : base_virtual_kernel<rolling_ck> {
      struct static_data_type {
        callable window_op;
        intptr_t window_size;
      };

      static char *data_init(char *_static_data, const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                             intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        static_data_type *static_data = *reinterpret_cast<static_data_type **>(_static_data);

        return static_data->window_op.get()->data_init(static_data->window_op.get()->static_data(), dst_tp, nsrc,
                                                       src_tp, nkwd, kwds, tp_vars);
      }

      static void resolve_dst_type(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                   const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                   const std::map<std::string, ndt::type> &tp_vars);

      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars);
    };

    typedef rolling_ck::static_data_type rolling_callable_data;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
