//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__MAKE_LIFTED_CKERNEL_HPP_
#define _DYND__MAKE_LIFTED_CKERNEL_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>

namespace dynd {

/**
 * Lifts the provided ckernel, broadcasting it as necessary to execute
 * across the additional dimensions in the ``lifted_types`` array.
 *
 * This version is for 'expr' ckernels.
 *
 * \param elwise_handler  The ckernel_deferred being lifted
 * \param out_ckb  The ckernel_builder into which to place the ckernel.
 * \param ckb_offset  Where within the ckernel_builder to place the ckernel.
 * \param lifted_types  The types to lift the ckernel to. The output ckernel
 *                      is for these types.
 * \param dynd_metadata  Array metadata corresponding to the lifted_types.
 * \param kernreq  Either dynd::kernel_request_single or dynd::kernel_request_strided,
 *                  as required by the caller.
 */
size_t make_lifted_expr_ckernel(const ckernel_deferred *elwise_handler,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const ndt::type *lifted_types,
                const char *const* dynd_metadata,
                dynd::kernel_request_t kernreq);

} // namespace dynd

#endif // _DYND__MAKE_LIFTED_CKERNEL_HPP_
