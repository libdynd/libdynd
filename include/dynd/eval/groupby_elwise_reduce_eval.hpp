//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__GROUPBY_UNARY_ELWISE_EVAL_HPP_
#define _DYND__GROUPBY_UNARY_ELWISE_EVAL_HPP_

#include <deque>

#include <dynd/eval/eval_context.hpp>

namespace dynd { namespace eval {

/**
 * Applies the unary kernel to the input strided array node.
 */
ndarray_node_ptr evaluate_groupby_elwise_reduce(ndarray_node *node, const eval::eval_context *ectx,
                                bool copy, uint32_t access_flags);

}} // namespace dynd::eval

#endif // _DYND__GROUPBY_UNARY_ELWISE_EVAL_HPP_
