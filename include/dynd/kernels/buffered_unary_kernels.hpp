//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFERED_UNARY_KERNELS_HPP_
#define _DYND__BUFFERED_UNARY_KERNELS_HPP_

#include <vector>
#include <deque>

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/buffer_storage.hpp>

namespace dynd {

/**
 * Given a size-N deque of kernel instances and a size-(N+1) vector
 * of the start, intermediate, and final element sizes, creates a kernel which chains
 * them all together through intermediate buffers.
 *
 * The deque is used instead of vector because kernel_instance's shouldn't
 * be copied unless you want an expensive copy operation, and we can't rely
 * on C++11 move semantics for this library.
 *
 * For efficiency, the kernels are swapped out of the deque instead of copied,
 * so the deque 'kernels' no longer contains them on exit.
 */
void make_buffered_chain_unary_kernel(std::deque<kernel_instance<unary_operation_pair_t>>& kernels,
                    std::deque<dtype>& dtypes, kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * This uses push_front calls on the output kernels and element_sizes
 * deques to create a chain of kernels which can transform the dtype's
 * storage_dtype values into its value_dtype values. It assumes
 * contiguous arrays are used for the intermediate buffers.
 *
 * This function assumes 'dt' is an expression_kind dtype, the
 * caller must verify this before calling.
 */
void push_front_dtype_storage_to_value_kernels(const dynd::dtype& dt,
                    const eval::eval_context *ectx,
                    std::deque<kernel_instance<unary_operation_pair_t>>& out_kernels,
                    std::deque<dtype>& out_dtypes);

/**
 * This uses push_back calls on the output kernels and element_sizes
 * deques to create a chain of kernels which can transform the dtype's
 * value_dtype values into its storage_dtype values. It assumes
 * contiguous arrays are used for the intermediate buffers.
 *
 * This function assumes 'dt' is an expression_kind dtype, the
 * caller must verify this before calling.
 */
void push_back_dtype_value_to_storage_kernels(const dynd::dtype& dt,
                    const eval::eval_context *ectx,
                    std::deque<kernel_instance<unary_operation_pair_t>>& out_kernels,
                    std::deque<dtype>& out_dtypes);

} // namespace dynd

#endif // _DYND__BUFFERED_UNARY_KERNELS_HPP_
