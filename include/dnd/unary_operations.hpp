//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__UNARY_OPERATIONS_HPP_
#define _DND__UNARY_OPERATIONS_HPP_

#include <vector>
#include <deque>

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

void unary_chain_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata);
struct unary_chain_auxdata {
    kernel_instance<unary_operation_t>* kernels;
    buffer_storage* bufs;
    int buf_count;

    unary_chain_auxdata() {
        kernels = 0;
        bufs = 0;
    }

    ~unary_chain_auxdata() {
        delete[] kernels;
        delete[] bufs;
    }

    void init(int buf_count) {
        kernels = new kernel_instance<unary_operation_t>[buf_count + 1];
        try {
            bufs = new buffer_storage[buf_count];
        } catch(const std::exception&) {
            delete[] kernels;
            kernels = 0;
            throw;
        }
    }
};

/**
 * Given a size-N deque of kernel instances and a size-(N-1) vector
 * of the intermediate element sizes, creates a kernel which chains
 * them all together through intermediate buffers.
 *
 * The deque is used instead of vector because kernel_instance's shouldn't
 * be copied unless you want an expensive copy operation, and we can't rely
 * on C++11 move semantics for this library.
 *
 * For efficiency, the kernels are swapped out of the deque instead of copied,
 * so the deque 'kernels' no longer contains them on exit.
 */
void make_unary_chain_kernel(std::deque<kernel_instance<unary_operation_t> >& kernels,
                    std::deque<intptr_t>& element_sizes, kernel_instance<unary_operation_t>& out_kernel);

} // namespace dnd

#endif // _DND__UNARY_OPERATIONS_HPP_