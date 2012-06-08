//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#include <dnd/dtype.hpp>
#include <dnd/unary_operations.hpp>

using namespace std;
using namespace dnd;

void dnd::unary_2chain_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata)
{
    const unary_2chain_auxdata& ad = get_auxiliary_data<unary_2chain_auxdata>(auxdata);
    do {
        intptr_t block_count = ad.buf.element_count();
        if (count < block_count) {
            block_count = count;
        }

        // First link of the chain
        ad.kernels[0].kernel(ad.buf.storage(), ad.buf.element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
        // Second link of the chain
        ad.kernels[1].kernel(dst, dst_stride, ad.buf.storage(), ad.buf.element_size(), block_count, ad.kernels[1].auxdata);

        src += block_count * src_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

void dnd::unary_chain_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata)
{
    const unary_chain_auxdata& ad = get_auxiliary_data<unary_chain_auxdata>(auxdata);
    int buf_count = ad.buf_count;
    do {
        intptr_t block_count = ad.bufs[0].element_count();
        if (count < block_count) {
            block_count = count;
        }

        // From the source into the first buffer
        ad.kernels[0].kernel(ad.bufs[0].storage(), ad.bufs[0].element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
        // All the links from buffer to buffer
        for (int i = 1; i < buf_count; ++i) {
            ad.kernels[i].kernel(ad.bufs[i].storage(), ad.bufs[i].element_size(), ad.bufs[i-1].storage(), ad.bufs[i-1].element_size(), block_count, ad.kernels[i].auxdata);
        }
        // From the last buffer into the destination
        ad.kernels[buf_count].kernel(dst, dst_stride, ad.bufs[buf_count-1].storage(), ad.bufs[buf_count-1].element_size(), block_count, ad.kernels[buf_count].auxdata);

        src += block_count * src_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}


void dnd::make_unary_chain_kernel(std::deque<kernel_instance<unary_operation_t> >& kernels,
                    std::deque<intptr_t>& element_sizes, kernel_instance<unary_operation_t>& out_kernel)
{
    if (kernels.size() != element_sizes.size() + 1) {
        throw std::runtime_error("make_unary_chain_kernel: the size of 'kernels' must be one more than 'element_sizes'");
    }

    switch (kernels.size()) {
    case 1:
        kernels[0].swap(out_kernel);
        return;
    case 2: {
        out_kernel.kernel = &unary_2chain_kernel;
        make_auxiliary_data<unary_2chain_auxdata>(out_kernel.auxdata);
        unary_2chain_auxdata &auxdata = out_kernel.auxdata.get<unary_2chain_auxdata>();

        auxdata.buf.allocate(element_sizes[0]); // TODO: pass buffering data through here

        auxdata.kernels[0].swap(kernels[0]);
        auxdata.kernels[1].swap(kernels[1]);
        return;
        }
    default: {
        out_kernel.kernel = &unary_chain_kernel;
        make_auxiliary_data<unary_chain_auxdata>(out_kernel.auxdata);
        unary_chain_auxdata &auxdata = out_kernel.auxdata.get<unary_chain_auxdata>();
        auxdata.init((int)element_sizes.size());

        for (size_t i = 0; i < element_sizes.size(); ++i) {
            auxdata.bufs[i].allocate(element_sizes[i]); // TODO: pass buffering data through here
        }

        for (size_t i = 0; i < kernels.size(); ++i) {
            auxdata.kernels[i].swap(kernels[i]);
        }
        }
    }
}
