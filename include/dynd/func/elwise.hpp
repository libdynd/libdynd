//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/elwise.hpp>

#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

namespace dynd {
namespace decl {
  namespace nd {
    class elwise : public arrfunc<elwise> {
    public:
      static int resolve_dst_type(const arrfunc_type_data *child,
                                  const arrfunc_type *child_tp, intptr_t nsrc,
                                  const ndt::type *src_tp, int throw_on_error,
                                  ndt::type &dst_tp,
                                  const dynd::nd::array &kwds);

      /**
       * Lifts the provided ckernel, broadcasting it as necessary to execute
       * across the additional dimensions in the ``lifted_types`` array.
       *
       * This version is for 'expr' ckernels.
       *
       * \param child  The arrfunc being lifted
       * \param ckb  The ckernel_builder into which to place the ckernel.
       * \param ckb_offset  Where within the ckernel_builder to place the
       *ckernel.
       * \param dst_tp  The destination type to lift to.
       * \param dst_arrmeta  The destination arrmeta to lift to.
       * \param src_tp  The source types to lift to.
       * \param src_arrmeta  The source arrmetas to lift to.
       * \param kernreq  Either kernel_request_single or kernel_request_strided,
       *                 as required by the caller.
       * \param ectx  The evaluation context.
       */
      static intptr_t
      instantiate(const arrfunc_type_data *child, const arrfunc_type *child_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, const ndt::type *src_tp,
                  const char *const *src_arrmeta,
                  dynd::kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds);

      static dynd::nd::arrfunc make();
    };
  } // namespace nd
} // namespace decl

namespace nd {
  extern decl::nd::elwise elwise;
}

} // namespace dynd