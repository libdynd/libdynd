//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

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
    arrfunc elwise(const arrfunc &child);

    ndt::type elwise_make_type(const arrfunc_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
