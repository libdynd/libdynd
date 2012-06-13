//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <stdexcept>

#include <dnd/diagnostics.hpp>
#include <dnd/kernels/alignment_kernels.hpp>

using namespace std;
using namespace dnd;

/**
 * An unaligned copy kernel function for a predefined fixed size,
 * for data which is contiguous.
 */
template<int N>
static void unaligned_contig_fixed_size_copy_kernel(char *dst, intptr_t,
                    const char *src, intptr_t,
                    intptr_t count, const AuxDataBase *)
{
    memcpy(dst, src, N * count);
}

/**
 * An unaligned copy kernel function for a predefined fixed size.
 */
template<int N>
static void unaligned_fixed_size_copy_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *)
{
    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst, src, N);
                
        dst += dst_stride;
        src += src_stride;
    }
}

/**
 * An unaligned copy kernel function for arbitrary-sized copies
 * of contiguous data. The auxiliary data should
 * be created by calling make_auxiliary_data<intptr_t>(), and initialized
 * with the element size.
 */
static void unaligned_contig_copy_kernel(char *dst, intptr_t,
                    const char *src, intptr_t,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = get_auxiliary_data<intptr_t>(auxdata);

    memcpy(dst, src, element_size * count);
}

/**
 * An unaligned copy kernel function for arbitrary-sized copies. The auxiliary data should
 * be created by calling make_auxiliary_data<intptr_t>(), and initialized
 * with the element size.
 */
static void unaligned_copy_kernel(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t element_size = get_auxiliary_data<intptr_t>(auxdata);

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst, src, element_size);
                
        dst += dst_stride;
        src += src_stride;
    }
}


void dnd::get_unaligned_copy_kernel(intptr_t element_size,
                intptr_t dst_fixedstride, intptr_t src_fixedstride,
                kernel_instance<unary_operation_t>& out_kernel)
{
    if (dst_fixedstride == element_size && dst_fixedstride == element_size) {
        switch (element_size) {
        case 2:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_contig_fixed_size_copy_kernel<2>;
            break;
        case 4:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_contig_fixed_size_copy_kernel<4>;
            break;
        case 8:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_contig_fixed_size_copy_kernel<8>;
            break;
        case 16:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_contig_fixed_size_copy_kernel<16>;
            break;
        default:
            make_auxiliary_data<intptr_t>(out_kernel.auxdata, element_size);
            out_kernel.kernel = &unaligned_contig_copy_kernel;
            break;
        }
    } else {
        switch (element_size) {
        case 2:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_fixed_size_copy_kernel<2>;
            break;
        case 4:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_fixed_size_copy_kernel<4>;
            break;
        case 8:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_fixed_size_copy_kernel<8>;
            break;
        case 16:
            out_kernel.auxdata.free();
            out_kernel.kernel = &unaligned_fixed_size_copy_kernel<16>;
            break;
        default:
            make_auxiliary_data<intptr_t>(out_kernel.auxdata, element_size);
            out_kernel.kernel = &unaligned_copy_kernel;
            break;
        }
    }
}
