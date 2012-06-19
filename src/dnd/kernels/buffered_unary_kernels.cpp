//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtype.hpp>
#include <dnd/kernels/buffered_unary_kernels.hpp>

using namespace std;
using namespace dnd;

namespace {
    /**
     * This is a unary kernel function + auxiliary data for chaining two
     * kernel functions together, using a single intermediate buffer.
     *
     * Example usage:
     *   void produce_kernel(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel)
     *   {
     *       // Set the kernel function
     *       out_kernel.kernel = &buffered_2chain_unary_kernel;
     *       // Allocate the auxiliary data for the kernel
     *       make_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(out_kernel.auxdata);
     *       // Get a reference to the auxiliary data just allocated
     *       buffered_2chain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_2chain_unary_kernel_auxdata>();
     *       // Allocate the buffering memory
     *       auxdata.buf.allocate(intermediate_element_size);
     *       // Get the two kernels in the chain
     *       produce_first_kernel(auxdata.buf.element_size(), src_fixedstride, auxdata.kernels[0]);
     *       produce_second_kernel(dst_fixedstride, auxdata.buf.element_size(), auxdata.kernels[1]);
     *   }
     */
    struct buffered_2chain_unary_kernel_auxdata {
        unary_specialization_kernel_instance kernels[2];
        buffer_storage buf;
    };

    /**
     * Just like the 2chain kernel, but for 3 kernels chained together.
     */
    struct buffered_3chain_unary_kernel_auxdata {
        unary_specialization_kernel_instance kernels[3];
        buffer_storage bufs[2];
    };

    struct buffered_nchain_unary_kernel_auxdata {
        // We use raw heap-allocated arrays for performance.
        // If we were using C++11, would use unique_ptr<T[]>.
        unary_specialization_kernel_instance* m_kernels;
        buffer_storage* m_bufs;
        int m_buf_count;

        buffered_nchain_unary_kernel_auxdata() {
            m_kernels = 0;
            m_bufs = 0;
        }

        ~buffered_nchain_unary_kernel_auxdata() {
            delete[] m_kernels;
            delete[] m_bufs;
        }

        void init(int buf_count) {
            m_kernels = new unary_specialization_kernel_instance[buf_count + 1];
            try {
                m_bufs = new buffer_storage[buf_count];
            } catch(const std::exception&) {
                delete[] m_kernels;
                m_kernels = 0;
                throw;
            }
            m_buf_count = buf_count;
        }
    };

    struct buffered_2chain_unary_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(auxdata);
            if (src_stride != 0) {
                // The fully general case
                unary_operation_t link0 = ad.kernels[0].specializations[general_unary_specialization];
                unary_operation_t link1 = ad.kernels[1].specializations[general_unary_specialization];
                do {
                    intptr_t block_count = ad.buf.element_count();
                    if (count < block_count) {
                        block_count = count;
                    }

                    // First link of the chain
                    link0(ad.buf.storage(), ad.buf.element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
                    // Second link of the chain
                    link1(dst, dst_stride, ad.buf.storage(), ad.buf.element_size(), block_count, ad.kernels[1].auxdata);

                    src += block_count * src_stride;
                    dst += block_count * dst_stride;
                    count -= block_count;
                } while (count > 0);
            } else {
                // Deal with the src stride == 0 case specially, since it isn't fully specialized at the function pointer level
                unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
                unary_operation_t link1 = ad.kernels[1].specializations[general_unary_specialization];

                // First link of the chain
                link0(ad.buf.storage(), 0, src, 0, count, ad.kernels[0].auxdata);
                // Second link of the chain
                link1(dst, dst_stride, ad.buf.storage(), 0, count, ad.kernels[1].auxdata);
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[scalar_unary_specialization];

            // First link of the chain
            link0(ad.buf.storage(), 0, src, 0, count, ad.kernels[0].auxdata);
            // Second link of the chain
            link1(dst, 0, ad.buf.storage(), 0, count, ad.kernels[1].auxdata);
        }

        static void contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[contiguous_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[contiguous_unary_specialization];

            do {
                intptr_t block_count = ad.buf.element_count();
                if (count < block_count) {
                    block_count = count;
                }

                // First link of the chain
                link0(ad.buf.storage(), ad.buf.element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
                // Second link of the chain
                link1(dst, dst_stride, ad.buf.storage(), ad.buf.element_size(), block_count, ad.kernels[1].auxdata);

                src += block_count * src_stride;
                dst += block_count * dst_stride;
                count -= block_count;
            } while (count > 0);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[scalar_to_contiguous_unary_specialization];

            // First link of the chain
            link0(ad.buf.storage(), 0, src, 0, count, ad.kernels[0].auxdata);
            // Second link of the chain
            link1(dst, dst_stride, ad.buf.storage(), 0, count, ad.kernels[1].auxdata);
        }
    };

    struct buffered_3chain_unary_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(auxdata);
            if (src_stride != 0) {
                // The fully general case
                unary_operation_t link0 = ad.kernels[0].specializations[general_unary_specialization];
                unary_operation_t link1 = ad.kernels[1].specializations[contiguous_unary_specialization];
                unary_operation_t link2 = ad.kernels[2].specializations[general_unary_specialization];
                do {
                    intptr_t block_count = ad.bufs[0].element_count();
                    if (count < block_count) {
                        block_count = count;
                    }

                    // First link of the chain
                    link0(ad.bufs[0].storage(), ad.bufs[0].element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
                    // Second link of the chain
                    link1(ad.bufs[1].storage(), ad.bufs[1].element_size(), ad.bufs[0].storage(), ad.bufs[0].element_size(), block_count, ad.kernels[1].auxdata);
                    // Third link of the chain
                    link2(dst, dst_stride, ad.bufs[1].storage(), ad.bufs[1].element_size(), block_count, ad.kernels[2].auxdata);

                    src += block_count * src_stride;
                    dst += block_count * dst_stride;
                    count -= block_count;
                } while (count > 0);
            } else {
                // Deal with the src stride == 0 case specially, since it isn't fully specialized at the function pointer level
                unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
                unary_operation_t link1 = ad.kernels[1].specializations[scalar_unary_specialization];
                unary_operation_t link2 = ad.kernels[2].specializations[general_unary_specialization];

                // First link of the chain
                link0(ad.bufs[0].storage(), 0, src, 0, count, ad.kernels[0].auxdata);
                // Second link of the chain
                link1(ad.bufs[1].storage(), 0, ad.bufs[0].storage(), 0, count, ad.kernels[1].auxdata);
                // Third link of the chain
                link2(dst, dst_stride, ad.bufs[1].storage(), 0, count, ad.kernels[2].auxdata);
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[scalar_unary_specialization];
            unary_operation_t link2 = ad.kernels[2].specializations[scalar_unary_specialization];

            // First link of the chain
            link0(ad.bufs[0].storage(), 0, src, 0, count, ad.kernels[0].auxdata);
            // Second link of the chain
            link1(ad.bufs[1].storage(), 0, ad.bufs[0].storage(), 0, count, ad.kernels[1].auxdata);
            // Third link of the chain
            link2(dst, 0, ad.bufs[1].storage(), 0, count, ad.kernels[2].auxdata);
        }

        static void contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[contiguous_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[contiguous_unary_specialization];
            unary_operation_t link2 = ad.kernels[2].specializations[contiguous_unary_specialization];

            do {
                intptr_t block_count = ad.bufs[0].element_count();
                if (count < block_count) {
                    block_count = count;
                }

                // First link of the chain
                link0(ad.bufs[0].storage(), ad.bufs[0].element_size(), src, src_stride, block_count, ad.kernels[0].auxdata);
                // Second link of the chain
                link1(ad.bufs[1].storage(), ad.bufs[1].element_size(), ad.bufs[0].storage(), ad.bufs[0].element_size(), block_count, ad.kernels[1].auxdata);
                // Third link of the chain
                link2(dst, dst_stride, ad.bufs[1].storage(), ad.bufs[1].element_size(), block_count, ad.kernels[2].auxdata);

                src += block_count * src_stride;
                dst += block_count * dst_stride;
                count -= block_count;
            } while (count > 0);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(auxdata);
            unary_operation_t link0 = ad.kernels[0].specializations[scalar_unary_specialization];
            unary_operation_t link1 = ad.kernels[1].specializations[scalar_unary_specialization];
            unary_operation_t link2 = ad.kernels[2].specializations[scalar_to_contiguous_unary_specialization];

            // First link of the chain
            link0(ad.bufs[0].storage(), 0, src, 0, count, ad.kernels[0].auxdata);
            // Second link of the chain
            link1(ad.bufs[1].storage(), 0, ad.bufs[0].storage(), 0, count, ad.kernels[1].auxdata);
            // Second link of the chain
            link2(dst, dst_stride, ad.bufs[1].storage(), 0, count, ad.kernels[2].auxdata);
        }
    };

    struct buffered_nchain_unary_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(auxdata);
            int buf_count = ad.m_buf_count;
            if (src_stride != 0) {
                // The fully general case
                do {
                    intptr_t block_count = ad.m_bufs[0].element_count();
                    if (count < block_count) {
                        block_count = count;
                    }

                    // From the source into the first buffer
                    ad.m_kernels[0].specializations[general_unary_specialization](
                                ad.m_bufs[0].storage(), ad.m_bufs[0].element_size(), src, src_stride, block_count, ad.m_kernels[0].auxdata);
                    // All the links from buffer to buffer
                    for (int i = 1; i < buf_count; ++i) {
                        ad.m_kernels[i].specializations[contiguous_unary_specialization](
                                ad.m_bufs[i].storage(), ad.m_bufs[i].element_size(), ad.m_bufs[i-1].storage(), ad.m_bufs[i-1].element_size(), block_count, ad.m_kernels[i].auxdata);
                    }
                    // From the last buffer into the destination
                    ad.m_kernels[buf_count].specializations[general_unary_specialization](
                                dst, dst_stride, ad.m_bufs[buf_count-1].storage(), ad.m_bufs[buf_count-1].element_size(), block_count, ad.m_kernels[buf_count].auxdata);

                    src += block_count * src_stride;
                    dst += block_count * dst_stride;
                    count -= block_count;
                } while (count > 0);
            } else {
                // Deal with the src stride == 0 case specially, since it isn't fully specialized at the function pointer level

                // From the source into the first buffer
                ad.m_kernels[0].specializations[scalar_unary_specialization](
                            ad.m_bufs[0].storage(), 0, src, 0, count, ad.m_kernels[0].auxdata);
                // All the links from buffer to buffer
                for (int i = 1; i < buf_count; ++i) {
                    ad.m_kernels[i].specializations[scalar_unary_specialization](
                            ad.m_bufs[i].storage(), 0, ad.m_bufs[i-1].storage(), 0, count, ad.m_kernels[i].auxdata);
                }
                // From the last buffer into the destination
                ad.m_kernels[buf_count].specializations[general_unary_specialization](
                            dst, dst_stride, ad.m_bufs[buf_count-1].storage(), 0, count, ad.m_kernels[buf_count].auxdata);
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(auxdata);
            int buf_count = ad.m_buf_count;

            // From the source into the first buffer
            ad.m_kernels[0].specializations[scalar_unary_specialization](
                        ad.m_bufs[0].storage(), 0, src, 0, count, ad.m_kernels[0].auxdata);
            // All the links from buffer to buffer
            for (int i = 1; i < buf_count; ++i) {
                ad.m_kernels[i].specializations[scalar_unary_specialization](
                        ad.m_bufs[i].storage(), 0, ad.m_bufs[i-1].storage(), 0, count, ad.m_kernels[i].auxdata);
            }
            // From the last buffer into the destination
            ad.m_kernels[buf_count].specializations[scalar_unary_specialization](
                        dst, 0, ad.m_bufs[buf_count-1].storage(), 0, count, ad.m_kernels[buf_count].auxdata);
        }

        static void contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(auxdata);
            int buf_count = ad.m_buf_count;

            do {
                intptr_t block_count = ad.m_bufs[0].element_count();
                if (count < block_count) {
                    block_count = count;
                }

                // From the source into the first buffer
                ad.m_kernels[0].specializations[contiguous_unary_specialization](
                            ad.m_bufs[0].storage(), ad.m_bufs[0].element_size(), src, src_stride, block_count, ad.m_kernels[0].auxdata);
                // All the links from buffer to buffer
                for (int i = 1; i < buf_count; ++i) {
                    ad.m_kernels[i].specializations[contiguous_unary_specialization](
                            ad.m_bufs[i].storage(), ad.m_bufs[i].element_size(), ad.m_bufs[i-1].storage(), ad.m_bufs[i-1].element_size(), block_count, ad.m_kernels[i].auxdata);
                }
                // From the last buffer into the destination
                ad.m_kernels[buf_count].specializations[contiguous_unary_specialization](
                            dst, dst_stride, ad.m_bufs[buf_count-1].storage(), ad.m_bufs[buf_count-1].element_size(), block_count, ad.m_kernels[buf_count].auxdata);

                src += block_count * src_stride;
                dst += block_count * dst_stride;
                count -= block_count;
            } while (count > 0);
        }

        static void scalar_to_contiguous_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(auxdata);
            int buf_count = ad.m_buf_count;

            // From the source into the first buffer
            ad.m_kernels[0].specializations[scalar_unary_specialization](
                        ad.m_bufs[0].storage(), 0, src, 0, count, ad.m_kernels[0].auxdata);
            // All the links from buffer to buffer
            for (int i = 1; i < buf_count; ++i) {
                ad.m_kernels[i].specializations[scalar_unary_specialization](
                        ad.m_bufs[i].storage(), 0, ad.m_bufs[i-1].storage(), 0, count, ad.m_kernels[i].auxdata);
            }
            // From the last buffer into the destination
            ad.m_kernels[buf_count].specializations[scalar_to_contiguous_unary_specialization](
                        dst, dst_stride, ad.m_bufs[buf_count-1].storage(), 0, count, ad.m_kernels[buf_count].auxdata);
        }
    };

} // anonymous namespace

void dnd::make_buffered_chain_unary_kernel(std::deque<unary_specialization_kernel_instance>& kernels,
                    std::deque<intptr_t>& element_sizes, unary_specialization_kernel_instance& out_kernel)
{
    if (kernels.size() != element_sizes.size() + 1) {
        std::stringstream ss;
        ss << "make_buffered_nchain_unary_kernel: the size of 'kernels' (" << kernels.size()
            << ") must be one more than 'element_sizes' (" << element_sizes.size() << ")";
        throw std::runtime_error(ss.str());
    }

    static specialized_unary_operation_table_t optable_2chain = {
        buffered_2chain_unary_kernel::general_kernel, 
        buffered_2chain_unary_kernel::scalar_kernel,
        buffered_2chain_unary_kernel::contiguous_kernel,
        buffered_2chain_unary_kernel::scalar_to_contiguous_kernel};

    static specialized_unary_operation_table_t optable_3chain  = {
        buffered_3chain_unary_kernel::general_kernel, 
        buffered_3chain_unary_kernel::scalar_kernel,
        buffered_3chain_unary_kernel::contiguous_kernel,
        buffered_3chain_unary_kernel::scalar_to_contiguous_kernel};

    static specialized_unary_operation_table_t optable_nchain = {
        buffered_nchain_unary_kernel::general_kernel, 
        buffered_nchain_unary_kernel::scalar_kernel,
        buffered_nchain_unary_kernel::contiguous_kernel,
        buffered_nchain_unary_kernel::scalar_to_contiguous_kernel};

    switch (kernels.size()) {
    case 1:
        kernels[0].swap(out_kernel);
        break;
    case 2: {
        out_kernel.specializations = optable_2chain;
        make_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(out_kernel.auxdata);
        buffered_2chain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_2chain_unary_kernel_auxdata>();

        auxdata.buf.allocate(element_sizes[0]); // TODO: pass buffering data through here

        auxdata.kernels[0].swap(kernels[0]);
        auxdata.kernels[1].swap(kernels[1]);
        break;
        }
    case 3: {
        out_kernel.specializations = optable_3chain;
        make_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(out_kernel.auxdata);
        buffered_3chain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_3chain_unary_kernel_auxdata>();

        auxdata.bufs[0].allocate(element_sizes[0]); // TODO: pass buffering data through here
        auxdata.bufs[1].allocate(element_sizes[1]);

        auxdata.kernels[0].swap(kernels[0]);
        auxdata.kernels[1].swap(kernels[1]);
        auxdata.kernels[2].swap(kernels[2]);
        break;
        }
    default: {
        out_kernel.specializations = optable_nchain;
        make_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(out_kernel.auxdata);
        buffered_nchain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_nchain_unary_kernel_auxdata>();
        auxdata.init((int)element_sizes.size());

        for (size_t i = 0; i < element_sizes.size(); ++i) {
            auxdata.m_bufs[i].allocate(element_sizes[i]); // TODO: pass buffering data through here
        }

        for (size_t i = 0; i < kernels.size(); ++i) {
            auxdata.m_kernels[i].swap(kernels[i]);
        }
        break;
        }
    }
}
void dnd::push_front_dtype_storage_to_value_kernels(const dnd::dtype& dt,
                    std::deque<unary_specialization_kernel_instance>& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const dtype* front_dt = &dt;
    const dtype* next_dt = &dt.extended()->operand_dtype(dt);
    if (next_dt->kind() != expression_kind) {
        // Special case when there is just one
        out_kernels.push_front(dt.extended()->get_operand_to_value_kernel());
    } else {
        intptr_t front_buffer_size;

        do {
            // Add this kernel to the deque
            out_element_sizes.push_front(next_dt->value_dtype().itemsize());
            out_kernels.push_front(front_dt->extended()->get_operand_to_value_kernel());
            // Shift to the next dtype
            front_dt = next_dt;
            next_dt = &front_dt->extended()->operand_dtype(*front_dt);
        } while (next_dt->kind() == expression_kind);
        // Add the final kernel from the source
        out_kernels.push_front(front_dt->extended()->get_operand_to_value_kernel());
    }
}

void dnd::push_back_dtype_value_to_storage_kernels(const dnd::dtype& dt,
                    std::deque<unary_specialization_kernel_instance>& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const dtype* back_dt = &dt;
    const dtype* next_dt = &dt.extended()->operand_dtype(dt);
    if (next_dt->kind() != expression_kind) {
        // Special case when there is just one
        out_kernels.push_back(dt.extended()->get_value_to_operand_kernel());
    } else {
        do {
            // Add this kernel to the deque
            out_element_sizes.push_back(next_dt->value_dtype().itemsize());
            out_kernels.push_back(back_dt->extended()->get_value_to_operand_kernel());
            // Shift to the next dtype
            back_dt = next_dt;
            next_dt = &back_dt->extended()->operand_dtype(*back_dt);
        } while (next_dt->kind() == expression_kind);
        // Add the final kernel from the source
        out_kernels.push_back(back_dt->extended()->get_value_to_operand_kernel());
    }
}
