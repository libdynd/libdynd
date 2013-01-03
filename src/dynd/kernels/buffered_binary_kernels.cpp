//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable

#include <dynd/dtype.hpp>
#include <dynd/kernels/buffered_binary_kernels.hpp>

using namespace std;
using namespace dynd;

namespace {
//
// There is a specialized kernel/auxdata for each nonempty subset of the output
// and two inputs, for a total of 7 specializations.
//
struct buffered_binary_out_in0_in1_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[3];
    buffer_storage<> bufs[3];
};
static void buffered_binary_out_in0_in1_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    buffered_binary_out_in0_in1_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_out_in0_in1_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // First input kernel
        ad.adapter_kernels[1].kernel(ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            src0, src0_stride,
                            block_count, ad.adapter_kernels[1].auxdata);
        // Second input kernel
        ad.adapter_kernels[2].kernel(ad.bufs[2].storage(), ad.bufs[2].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.adapter_kernels[2].auxdata);
        // Binary kernel
        ad.kernel.kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            ad.bufs[2].storage(), ad.bufs[2].get_data_size(),
                            block_count, ad.kernel.auxdata);
        // Output kernel
        ad.adapter_kernels[0].kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            block_count, ad.adapter_kernels[0].auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_out_in0_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[2];
    buffer_storage<> bufs[2];
};
static void buffered_binary_out_in0_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_out_in0_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_out_in0_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // First input kernel
        ad.adapter_kernels[1].kernel(ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            src0, src0_stride,
                            block_count, ad.adapter_kernels[1].auxdata);
        // Binary kernel
        ad.kernel.kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.kernel.auxdata);
        // Output kernel
        ad.adapter_kernels[0].kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            block_count, ad.adapter_kernels[0].auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_out_in1_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[2];
    buffer_storage<> bufs[2];
};
static void buffered_binary_out_in1_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_out_in1_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_out_in1_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // Second input kernel
        ad.adapter_kernels[1].kernel(ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.adapter_kernels[1].auxdata);
        // Binary kernel
        ad.kernel.kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src0, src0_stride,
                            ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            block_count, ad.kernel.auxdata);
        // Output kernel
        ad.adapter_kernels[0].kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            block_count, ad.adapter_kernels[0].auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_out_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[1];
    buffer_storage<> bufs[1];
};
static void buffered_binary_out_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_out_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_out_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // Binary kernel
        ad.kernel.kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src0, src0_stride,
                            src1, src1_stride,
                            block_count, ad.kernel.auxdata);
        // Output kernel
        ad.adapter_kernels[0].kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            block_count, ad.adapter_kernels[0].auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_in0_in1_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[2];
    buffer_storage<> bufs[2];
};
static void buffered_binary_in0_in1_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_in0_in1_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_in0_in1_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // First input kernel
        ad.adapter_kernels[0].kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src0, src0_stride,
                            block_count, ad.adapter_kernels[0].auxdata);
        // Second input kernel
        ad.adapter_kernels[1].kernel(ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.adapter_kernels[1].auxdata);
        // Binary kernel
        ad.kernel.kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            ad.bufs[1].storage(), ad.bufs[1].get_data_size(),
                            block_count, ad.kernel.auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_in0_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[1];
    buffer_storage<> bufs[1];
};
static void buffered_binary_in0_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_in0_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_in0_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // First input kernel
        ad.adapter_kernels[0].kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src0, src0_stride,
                            block_count, ad.adapter_kernels[0].auxdata);
        // Binary kernel
        ad.kernel.kernel(dst, dst_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.kernel.auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

struct buffered_binary_in1_kernel_auxdata {
    kernel_instance<binary_operation_t> kernel;
    kernel_instance<unary_operation_pair_t> adapter_kernels[1];
    buffer_storage<> bufs[1];
};
static void buffered_binary_in1_kernel(char *dst, intptr_t dst_stride,
                    const char *src0, intptr_t src0_stride,
                    const char *src1, intptr_t src1_stride,
                    intptr_t count, const AuxDataBase *auxdata)
{
    const buffered_binary_in1_kernel_auxdata& ad = get_auxiliary_data<buffered_binary_in1_kernel_auxdata>(auxdata);
    do {
        intptr_t block_count = buffer_storage<>::element_count;
        if (count < block_count) {
            block_count = count;
        }

        // Second input kernel
        ad.adapter_kernels[0].kernel(ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            src1, src1_stride,
                            block_count, ad.adapter_kernels[0].auxdata);
        // Binary kernel
        ad.kernel.kernel(dst, dst_stride,
                            src0, src0_stride,
                            ad.bufs[0].storage(), ad.bufs[0].get_data_size(),
                            block_count, ad.kernel.auxdata);

        src0 += block_count * src0_stride;
        src1 += block_count * src1_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}

} // anonymous namespace

void dynd::make_buffered_binary_kernel(kernel_instance<binary_operation_t>& kernel,
                    kernel_instance<unary_operation_pair_t>* adapters, const intptr_t *buffer_element_sizes,
                    kernel_instance<binary_operation_t>& out_kernel)
{
    //cout << "adapters: " << adapters[0].kernel << ", " << adapters[1].kernel << ", " << adapters[2].kernel << endl;
    if (adapters[0].kernel != 0) {
        if (adapters[1].kernel != 0) {
            if (adapters[2].kernel != 0) {
                // All three adapter kernels are there
                out_kernel.kernel = &buffered_binary_out_in0_in1_kernel;
                make_auxiliary_data<buffered_binary_out_in0_in1_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_out_in0_in1_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_out_in0_in1_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[0]); // TODO: pass buffering data through here
                auxdata.bufs[1].allocate(buffer_element_sizes[1]);
                auxdata.bufs[2].allocate(buffer_element_sizes[2]);

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[0]);
                auxdata.adapter_kernels[1].swap(adapters[1]);
                auxdata.adapter_kernels[2].swap(adapters[2]);
            } else {  // adapters[2].kernel is NULL
                // The output and first input kernels are there
                out_kernel.kernel = &buffered_binary_out_in0_kernel;
                make_auxiliary_data<buffered_binary_out_in0_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_out_in0_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_out_in0_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[0]); // TODO: pass buffering data through here
                auxdata.bufs[1].allocate(buffer_element_sizes[1]);

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[0]);
                auxdata.adapter_kernels[1].swap(adapters[1]);
            }
        } else { // adapters[1].kernel is NULL
            if (adapters[2].kernel != 0) {
                // The output and second input kernels are there
                out_kernel.kernel = &buffered_binary_out_in1_kernel;
                make_auxiliary_data<buffered_binary_out_in1_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_out_in1_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_out_in1_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[0]); // TODO: pass buffering data through here
                auxdata.bufs[1].allocate(buffer_element_sizes[2]);

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[0]);
                auxdata.adapter_kernels[1].swap(adapters[2]);
            } else {  // adapters[2].kernel is NULL
                // The output kernel is there
                out_kernel.kernel = &buffered_binary_out_kernel;
                make_auxiliary_data<buffered_binary_out_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_out_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_out_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[0]); // TODO: pass buffering data through here

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[0]);
            }
        }
    } else { // adapters[0].kernel is NULL
        if (adapters[1].kernel != 0) {
            if (adapters[2].kernel != 0) {
                // The first and second input kernels are there
                out_kernel.kernel = &buffered_binary_in0_in1_kernel;
                make_auxiliary_data<buffered_binary_in0_in1_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_in0_in1_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_in0_in1_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[1]); // TODO: pass buffering data through here
                auxdata.bufs[1].allocate(buffer_element_sizes[2]);

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[1]);
                auxdata.adapter_kernels[1].swap(adapters[2]);
            } else {  // adapters[2].kernel is NULL
                // The first input kernel is there
                out_kernel.kernel = &buffered_binary_in0_kernel;
                make_auxiliary_data<buffered_binary_in0_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_in0_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_in0_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[1]); // TODO: pass buffering data through here

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[1]);
            }
        } else { // adapters[1].kernel is NULL
            if (adapters[2].kernel != 0) {
                // The second input kernel is there
                out_kernel.kernel = &buffered_binary_in1_kernel;
                make_auxiliary_data<buffered_binary_in1_kernel_auxdata>(out_kernel.extra.auxdata);
                buffered_binary_in1_kernel_auxdata &auxdata = out_kernel.extra.auxdata.get<buffered_binary_in1_kernel_auxdata>();

                auxdata.bufs[0].allocate(buffer_element_sizes[2]); // TODO: pass buffering data through here

                kernel.swap(auxdata.kernel);
                auxdata.adapter_kernels[0].swap(adapters[2]);
            } else {  // adapters[2].kernel is NULL
                // All three adapters are NULL, just give back the same kernel
                kernel.swap(out_kernel);
            }
        }
    }
}

#endif // TODO reenable