//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_REDUCE_EVAL_HPP_
#define _DND__ELWISE_REDUCE_EVAL_HPP_

#include <dynd/nodes/ndarray_node.hpp>

namespace dynd { namespace eval {

ndarray_node_ptr evaluate_elwise_reduce_array(ndarray_node* node,
                    const eval::eval_context *ectx, bool copy, uint32_t access_flags);

}} // namespace dynd::eval

#endif // _DND__ELWISE_REDUCE_EVAL_HPP_
