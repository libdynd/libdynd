//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__EVAL_ENGINE_HPP_
#define _DND__EVAL_ENGINE_HPP_

#include <dnd/eval/eval_context.hpp>
#include <dnd/nodes/ndarray_node.hpp>

namespace dnd {

/**
 * The main evaluation function, which evaluates an arbitrary ndarray
 * node into a strided array node.
 *
 * @param node  The node to evaluate.
 * @param ectx  The evaluation context to use.
 * @param copy  If set to true, always makes a copy of the data.
 * @param access_flags  The requested access flags for the result, or 0 if anything is ok.
 */
ndarray_node_ptr evaluate(ndarray_node *node, const eval_context *ectx = &default_eval_context,
                    bool copy = false, uint32_t access_flags = 0);

} // namespace dnd

#endif // _DND__EVAL_ENGINE_HPP_