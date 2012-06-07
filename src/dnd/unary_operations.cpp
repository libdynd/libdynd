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
