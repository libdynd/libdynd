//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__BUFFERED_BINARY_KERNELS_HPP_
#define _DYND__BUFFERED_BINARY_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/buffer_storage.hpp>

namespace dynd {

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
                    kernel_instance<unary_operation_pair_t>* adapters, const intptr_t *buffer_element_sizes,
                    kernel_instance<binary_operation_t>& out_kernel);

} // namespace dynd

#endif // _DYND__BUFFERED_BINARY_KERNELS_HPP_
