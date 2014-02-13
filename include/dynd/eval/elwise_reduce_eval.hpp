//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_REDUCE_EVAL_HPP_
#define _DYND__ELWISE_REDUCE_EVAL_HPP_

namespace dynd { namespace eval {

ndarray_node_ptr evaluate_elwise_reduce_array(ndarray_node* node,
                    const eval::eval_context *ectx, bool copy, uint32_t access_flags);

}} // namespace dynd::eval

#endif // _DYND__ELWISE_REDUCE_EVAL_HPP_
