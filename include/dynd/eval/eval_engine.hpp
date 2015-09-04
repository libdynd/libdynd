//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/eval/eval_context.hpp>

namespace dynd { namespace eval {

/**
 * The main evaluation function, which evaluates an arbitrary ndarray
 * node into a strided array node.
 *
 * @param node  The node to evaluate.
 * @param ectx  The evaluation context to use.
 * @param copy  If set to true, always makes a copy of the data.
 * @param access_flags  The requested access flags for the result, or 0 if anything is ok.
 */
DYND_API ndarray_node_ptr evaluate(ndarray_node *node, const eval::eval_context *ectx = &eval::default_eval_context,
                    bool copy = false, uint32_t access_flags = 0);

/**
 * Analyzes whether a copy is required from the src to the dst because of the permissions.
 *
 * Sets the dst_access_flags, and flips inout_copy_required to true when a copy is needed.
 */
DYND_API void process_access_flags(uint32_t &dst_access_flags, uint32_t src_access_flags, bool &inout_copy_required);

}} // namespace dynd::eval
