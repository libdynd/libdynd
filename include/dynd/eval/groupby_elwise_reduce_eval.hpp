//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <deque>

#include <dynd/eval/eval_context.hpp>

namespace dynd { namespace eval {

/**
 * Applies the unary kernel to the input strided array node.
 */
DYND_API ndarray_node_ptr evaluate_groupby_elwise_reduce(ndarray_node *node, const eval::eval_context *ectx,
                                bool copy, uint32_t access_flags);

}} // namespace dynd::eval
