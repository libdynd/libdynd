//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__BUFFERED_BINARY_KERNELS_HPP_
#define _DND__BUFFERED_BINARY_KERNELS_HPP_

#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/buffer_storage.hpp>

namespace dnd {

/**
 * Given a binary kernel, an array of three unary kernels to be connected to
 * the output, the first input, and the second input respectively, and an
 * array of the three corresponding intermediate element sizes, produces
 * a new binary kernel which chains them together.
 *
 * To indicate that a specific input or output has no unary kernel to be attached,
 * specify NULL in the kernel's function pointer.
 *
 * For efficiency, the kernels are swapped instead of copied,
 * so the provided kernels are empty on exit.
 */
void make_buffered_binary_kernel(kernel_instance<binary_operation_t>& kernel,
                    kernel_instance<unary_operation_t>* adapters, const intptr_t *buffer_element_sizes,
                    kernel_instance<binary_operation_t>& out_kernel);

} // namespace dnd

#endif // _DND__BUFFERED_BINARY_KERNELS_HPP_