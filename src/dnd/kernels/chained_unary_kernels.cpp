//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#include <dnd/dtype.hpp>
#include <dnd/kernels/chained_unary_kernels.hpp>

using namespace std;
using namespace dnd;

void dnd::chained_2_unary_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata)
{
    const chained_2_unary_kernel_auxdata& ad = get_auxiliary_data<chained_2_unary_kernel_auxdata>(auxdata);
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

void dnd::chained_unary_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                        intptr_t count, const AuxDataBase *auxdata)
{
    const chained_unary_kernel_auxdata& ad = get_auxiliary_data<chained_unary_kernel_auxdata>(auxdata);
    int buf_count = ad.m_buf_count;
    do {
        intptr_t block_count = ad.m_bufs[0].element_count();
        if (count < block_count) {
            block_count = count;
        }

        // From the source into the first buffer
        ad.m_kernels[0].kernel(ad.m_bufs[0].storage(), ad.m_bufs[0].element_size(), src, src_stride, block_count, ad.m_kernels[0].auxdata);
        // All the links from buffer to buffer
        for (int i = 1; i < buf_count; ++i) {
            ad.m_kernels[i].kernel(ad.m_bufs[i].storage(), ad.m_bufs[i].element_size(), ad.m_bufs[i-1].storage(), ad.m_bufs[i-1].element_size(), block_count, ad.m_kernels[i].auxdata);
        }
        // From the last buffer into the destination
        ad.m_kernels[buf_count].kernel(dst, dst_stride, ad.m_bufs[buf_count-1].storage(), ad.m_bufs[buf_count-1].element_size(), block_count, ad.m_kernels[buf_count].auxdata);

        src += block_count * src_stride;
        dst += block_count * dst_stride;
        count -= block_count;
    } while (count > 0);
}


void dnd::make_chained_unary_kernel(std::deque<kernel_instance<unary_operation_t> >& kernels,
                    std::deque<intptr_t>& element_sizes, kernel_instance<unary_operation_t>& out_kernel)
{
    if (kernels.size() != element_sizes.size() + 1) {
        throw std::runtime_error("make_chained_unary_kernel: the size of 'kernels' must be one more than 'element_sizes'");
    }

    switch (kernels.size()) {
    case 1:
        kernels[0].swap(out_kernel);
        return;
    case 2: {
        out_kernel.kernel = &chained_2_unary_kernel;
        make_auxiliary_data<chained_2_unary_kernel_auxdata>(out_kernel.auxdata);
        chained_2_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<chained_2_unary_kernel_auxdata>();

        auxdata.buf.allocate(element_sizes[0]); // TODO: pass buffering data through here

        auxdata.kernels[0].swap(kernels[0]);
        auxdata.kernels[1].swap(kernels[1]);
        return;
        }
    default: {
        out_kernel.kernel = &chained_unary_kernel;
        make_auxiliary_data<chained_unary_kernel_auxdata>(out_kernel.auxdata);
        chained_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<chained_unary_kernel_auxdata>();
        auxdata.init((int)element_sizes.size());

        for (size_t i = 0; i < element_sizes.size(); ++i) {
            auxdata.m_bufs[i].allocate(element_sizes[i]); // TODO: pass buffering data through here
        }

        for (size_t i = 0; i < kernels.size(); ++i) {
            auxdata.m_kernels[i].swap(kernels[i]);
        }
        }
    }
}

void dnd::push_front_dtype_storage_to_value_kernels(const dnd::dtype& dt,
                    intptr_t dst_fixedstride, intptr_t src_fixedstride,
                    std::deque<kernel_instance<unary_operation_t> >& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const dtype* front_dt = &dt;
    const dtype* next_dt = &dt.extended()->operand_dtype(dt);
    if (next_dt->kind() != expression_kind || next_dt == &next_dt->extended()->operand_dtype(*next_dt)) {
        // Special case when there is just one
        out_kernels.push_front(kernel_instance<unary_operation_t>());
        dt.extended()->get_operand_to_value_operation(dst_fixedstride, src_fixedstride, out_kernels.front());
    } else {
        intptr_t front_dst_fixedstride = dst_fixedstride;
        intptr_t front_buffer_size;

        do {
            front_buffer_size = next_dt->value_dtype().itemsize();
            // Add this kernel to the deque
            out_kernels.push_front(kernel_instance<unary_operation_t>());
            out_element_sizes.push_front(front_buffer_size);
            front_dt->extended()->get_operand_to_value_operation(front_dst_fixedstride, front_buffer_size, out_kernels.front());
            // Shift to the next dtype
            front_dst_fixedstride = front_buffer_size;
            front_dt = next_dt;
            next_dt = &front_dt->extended()->operand_dtype(*front_dt);
        } while (next_dt->kind() == expression_kind && next_dt != &next_dt->extended()->operand_dtype(*next_dt));
        // Add the final kernel from the source
        out_kernels.push_front(kernel_instance<unary_operation_t>());
        front_dt->extended()->get_operand_to_value_operation(front_dst_fixedstride, src_fixedstride, out_kernels.front());
    }
}

void dnd::push_back_dtype_value_to_storage_kernels(const dnd::dtype& dt,
                    intptr_t dst_fixedstride, intptr_t src_fixedstride,
                    std::deque<kernel_instance<unary_operation_t> >& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const dtype* back_dt = &dt;
    const dtype* next_dt = &dt.extended()->operand_dtype(dt);
    if (next_dt->kind() != expression_kind || next_dt == &next_dt->extended()->operand_dtype(*next_dt)) {
        // Special case when there is just one
        out_kernels.push_back(kernel_instance<unary_operation_t>());
        dt.extended()->get_value_to_operand_operation(dst_fixedstride, src_fixedstride, out_kernels.back());
    } else {
        intptr_t back_src_fixedstride = src_fixedstride;
        intptr_t back_buffer_size;

        do {
            back_buffer_size = next_dt->value_dtype().itemsize();
            // Add this kernel to the deque
            out_kernels.push_back(kernel_instance<unary_operation_t>());
            out_element_sizes.push_back(back_buffer_size);
            back_dt->extended()->get_value_to_operand_operation(back_buffer_size, back_src_fixedstride, out_kernels.back());
            // Shift to the next dtype
            back_src_fixedstride = back_buffer_size;
            back_dt = next_dt;
            next_dt = &back_dt->extended()->operand_dtype(*back_dt);
        } while (next_dt->kind() == expression_kind && next_dt != &next_dt->extended()->operand_dtype(*next_dt));
        // Add the final kernel from the source
        out_kernels.push_back(kernel_instance<unary_operation_t>());
        back_dt->extended()->get_value_to_operand_operation(dst_fixedstride, back_src_fixedstride, out_kernels.back());
    }
}
