//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__UNARY_OPERATIONS_HPP_
#define _DND__UNARY_OPERATIONS_HPP_

#include <vector>

#include <dnd/unary_operations.hpp>
#include <dnd/buffer_storage.hpp>

namespace dnd {

/**
 * This is a unary kernel function + auxiliary data for chaining two
 * kernel functions together, using a single intermediate buffer.
 *
 * Example usage:
 *   void produce_kernel(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel)
 *   {
 *       // Set the kernel function
 *       out_kernel.kernel = &unary_2chain_kernel;
 *       // Allocate the auxiliary data for the kernel
 *       make_auxiliary_data<unary_2chain_auxdata>(out_kernel.auxdata);
 *       // Get a reference to the auxiliary data just allocated
 *       unary_2chain_auxdata &auxdata = out_kernel.auxdata.get<unary_2chain_auxdata>();
 *       // Allocate the buffering memory
 *       auxdata.buf.allocate(intermediate_element_size);
 *       // Get the two kernels in the chain
 *       produce_first_kernel(auxdata.buf.element_size(), src_fixedstride, auxdata.kernels[0]);
 *       produce_second_kernel(dst_fixedstride, auxdata.buf.element_size(), auxdata.kernels[1]);
 *   }
 */
void unary_2chain_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata);
struct unary_2chain_auxdata {
    kernel_instance<unary_operation_t> kernels[2];
    buffer_storage buf;
};

/**
 * Given a size-N vector of kernel instances and a size-(N-1) vector
 * of the intermediate element sizes, creates a kernel which chains
 * them all together through intermediate buffers
 */
void make_unary_chain_kernel(std::vector<kernel_instance<unary_operation_t> >& kernels,
                    std::vector<intptr_t>& element_sizes, kernel_instance<unary_operation_t>& out_kernel);

} // namepsace dnd

#endif // _DND__UNARY_OPERATIONS_HPP_