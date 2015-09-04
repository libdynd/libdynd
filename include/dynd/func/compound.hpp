//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Adds an adapter ckernel which wraps a child binary expr ckernel
     * as a unary reduction ckernel. The three types of the binary
     * expr kernel must all be equal.
     *
     * \param ckb  The ckernel_builder into which the kernel adapter is placed.
     * \param ckb_offset  The offset within the ckernel_builder at which to
     *place the adapter.
     * \param right_associative  If true, the reduction is to be evaluated right
     *to left,
     *                           (x0 * (x1 * (x2 * x3))), if false, the
     *reduction is to be
     *                           evaluted left to right (((x0 * x1) * x2) * x3).
     * \param kernreq  The type of kernel to produce (single or strided).
     *
     * \returns  The ckb_offset where the child ckernel should be placed.
     */
    DYND_API callable left_compound(const callable &child);

    DYND_API callable right_compound(const callable &child);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
