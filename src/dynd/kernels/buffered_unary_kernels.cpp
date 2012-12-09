//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>

using namespace std;
using namespace dynd;

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
     *       produce_first_kernel(auxdata.buf.get_element_size(), src_fixedstride, auxdata.kernels[0]);
     *       produce_second_kernel(dst_fixedstride, auxdata.buf.get_element_size(), auxdata.kernels[1]);
     *   }
     */
    struct buffered_2chain_unary_kernel_auxdata {
        kernel_instance<unary_operation_pair_t> kernels[2];
        unary_kernel_static_data kernel_extras[2];
        buffer_storage<> buf;
        size_t dst_item_size, src_item_size;
    };

    /**
     * Just like the 2chain kernel, but for 3 kernels chained together.
     */
    struct buffered_3chain_unary_kernel_auxdata {
        kernel_instance<unary_operation_pair_t> kernels[3];
        unary_kernel_static_data kernel_extras[3];
        buffer_storage<> bufs[2];
        size_t dst_item_size, src_item_size;
    };

    struct buffered_nchain_unary_kernel_auxdata {
        // We use raw heap-allocated arrays for performance.
        // If we were using C++11, would use unique_ptr<T[]>.
        kernel_instance<unary_operation_pair_t> *kernels;
        buffer_storage<> *bufs;
        unary_kernel_static_data *kernel_extras;
        int buf_count;
        size_t dst_item_size, src_item_size;

        buffered_nchain_unary_kernel_auxdata()
            : kernels(NULL), bufs(NULL), kernel_extras(NULL), buf_count(0)
        {}

        buffered_nchain_unary_kernel_auxdata(const buffered_nchain_unary_kernel_auxdata& rhs)
            : kernels(NULL), bufs(NULL), kernel_extras(NULL), buf_count(0)
        {
            throw std::runtime_error("buffered_nchain_unary_kernel_auxdata copy constructor not implemented");
        }

        ~buffered_nchain_unary_kernel_auxdata() {
            delete[] kernels;
            delete[] kernel_extras;
            delete[] bufs;
        }

        void init(int buffer_count) {
            kernels = new kernel_instance<unary_operation_pair_t>[buffer_count + 1];
            try {
                kernel_extras = new unary_kernel_static_data[buffer_count + 1];
                bufs = new buffer_storage<>[buffer_count];
            } catch(const std::exception&) {
                delete[] kernels;
                kernels = NULL;
                delete[] kernel_extras;
                kernel_extras = NULL;
                throw;
            }
            buf_count = buffer_count;
        }
    };

    struct buffered_2chain_unary_kernel {
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(extra->auxdata);
            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernels[0].kernel.single(ad.buf.get_storage(), src, &ad.kernel_extras[0]);
            ad.kernel_extras[1].dst_metadata = extra->dst_metadata;
            ad.kernels[1].kernel.single(dst, ad.buf.get_storage(), &ad.kernel_extras[1]);
            ad.buf.reset_metadata();
        }

        static void contig(char *dst, const char *src, size_t count, unary_kernel_static_data *extra)
        {
            buffered_2chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(extra->auxdata);
            size_t dst_item_size = ad.dst_item_size, src_item_size = ad.src_item_size;

            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernel_extras[1].dst_metadata = extra->dst_metadata;
            while (count > buffer_storage<>::element_count) {
                ad.kernels[0].kernel.contig(ad.buf.get_storage(), src, buffer_storage<>::element_count, &ad.kernel_extras[0]);
                ad.kernels[1].kernel.contig(dst, ad.buf.get_storage(), buffer_storage<>::element_count, &ad.kernel_extras[1]);
                ad.buf.reset_metadata();

                count -= buffer_storage<>::element_count;
                dst += buffer_storage<>::element_count * dst_item_size;
                src += buffer_storage<>::element_count * src_item_size;
            }
            ad.kernels[0].kernel.contig(ad.buf.get_storage(), src, count, &ad.kernel_extras[0]);
            ad.kernels[1].kernel.contig(dst, ad.buf.get_storage(), count, &ad.kernel_extras[1]);
            ad.buf.reset_metadata();
        }
    };

    struct buffered_3chain_unary_kernel {
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(extra->auxdata);
            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernels[0].kernel.single(ad.bufs[0].get_storage(), src, &ad.kernel_extras[0]);
            ad.kernels[1].kernel.single(ad.bufs[1].get_storage(), ad.bufs[0].get_storage(), &ad.kernel_extras[1]);
            ad.kernel_extras[2].dst_metadata = extra->dst_metadata;
            ad.kernels[2].kernel.single(dst, ad.bufs[1].get_storage(), &ad.kernel_extras[2]);
            ad.bufs[0].reset_metadata();
            ad.bufs[1].reset_metadata();
        }

        static void contig(char *dst, const char *src, size_t count, unary_kernel_static_data *extra)
        {
            buffered_3chain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(extra->auxdata);
            size_t dst_item_size = ad.dst_item_size, src_item_size = ad.src_item_size;

            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernel_extras[2].dst_metadata = extra->dst_metadata;
            while (count > buffer_storage<>::element_count) {
                ad.kernels[0].kernel.contig(ad.bufs[0].get_storage(), src, buffer_storage<>::element_count, &ad.kernel_extras[0]);
                ad.kernels[1].kernel.contig(ad.bufs[1].get_storage(), ad.bufs[0].get_storage(), buffer_storage<>::element_count, &ad.kernel_extras[1]);
                ad.kernels[2].kernel.contig(dst, ad.bufs[1].get_storage(), buffer_storage<>::element_count, &ad.kernel_extras[2]);
                ad.bufs[0].reset_metadata();
                ad.bufs[1].reset_metadata();

                count -= buffer_storage<>::element_count;
                dst += buffer_storage<>::element_count * dst_item_size;
                src += buffer_storage<>::element_count * src_item_size;
            }
            ad.kernels[0].kernel.contig(ad.bufs[0].get_storage(), src, count, &ad.kernel_extras[0]);
            ad.kernels[1].kernel.contig(ad.bufs[1].get_storage(), ad.bufs[0].get_storage(), count, &ad.kernel_extras[1]);
            ad.kernels[2].kernel.contig(dst, ad.bufs[1].get_storage(), count, &ad.kernel_extras[2]);
            ad.bufs[0].reset_metadata();
            ad.bufs[1].reset_metadata();
        }
    };

    struct buffered_nchain_unary_kernel {
        static void single(char *dst, const char *src, unary_kernel_static_data *extra)
        {
            buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(extra->auxdata);
            int buf_count = ad.buf_count;

            // From the source into the first buffer
            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernels[0].kernel.single(ad.bufs[0].get_storage(), src, &ad.kernel_extras[0]);
            // All the links from buffer to buffer
            for (int i = 1; i < buf_count; ++i) {
                ad.kernels[i].kernel.single(ad.bufs[i].get_storage(), ad.bufs[i-1].get_storage(), &ad.kernel_extras[i]);
            }
            // From the last buffer into the destination
            ad.kernel_extras[buf_count].dst_metadata = extra->dst_metadata;
            ad.kernels[buf_count].kernel.single(dst, ad.bufs[buf_count-1].get_storage(), &ad.kernel_extras[buf_count]);
            // Reset any metadata blockrefs
            for (int i = 0; i < buf_count; ++i) {
                ad.bufs[i].reset_metadata();
            }
        }

        static void contig(char *dst, const char *src, size_t count, unary_kernel_static_data *extra)
        {
            buffered_nchain_unary_kernel_auxdata& ad = get_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(extra->auxdata);
            size_t dst_item_size = ad.dst_item_size, src_item_size = ad.src_item_size;
            int buf_count = ad.buf_count;

            ad.kernel_extras[0].src_metadata = extra->src_metadata;
            ad.kernel_extras[buf_count].dst_metadata = extra->dst_metadata;
            while (count > buffer_storage<>::element_count) {
                // From the source into the first buffer
                ad.kernels[0].kernel.contig(ad.bufs[0].get_storage(), src, buffer_storage<>::element_count, &ad.kernel_extras[0]);
                // All the links from buffer to buffer
                for (int i = 1; i < buf_count; ++i) {
                    ad.kernels[i].kernel.contig(ad.bufs[i].get_storage(), ad.bufs[i-1].get_storage(), buffer_storage<>::element_count, &ad.kernel_extras[i]);
                }
                // From the last buffer into the destination
                ad.kernels[buf_count].kernel.contig(dst, ad.bufs[buf_count-1].get_storage(), buffer_storage<>::element_count, &ad.kernel_extras[buf_count]);
                // Reset any metadata blockrefs
                for (int i = 0; i < buf_count; ++i) {
                    ad.bufs[i].reset_metadata();
                }

                count -= buffer_storage<>::element_count;
                dst += buffer_storage<>::element_count * dst_item_size;
                src += buffer_storage<>::element_count * src_item_size;
            }
            // From the source into the first buffer
            ad.kernels[0].kernel.contig(ad.bufs[0].get_storage(), src, count, &ad.kernel_extras[0]);
            // All the links from buffer to buffer
            for (int i = 1; i < buf_count; ++i) {
                ad.kernels[i].kernel.contig(ad.bufs[i].get_storage(), ad.bufs[i-1].get_storage(), count, &ad.kernel_extras[i]);
            }
            // From the last buffer into the destination
            ad.kernels[buf_count].kernel.contig(dst, ad.bufs[buf_count-1].get_storage(), count, &ad.kernel_extras[buf_count]);
            // Reset any metadata blockrefs
            for (int i = 0; i < buf_count; ++i) {
                ad.bufs[i].reset_metadata();
            }
        }
    };
} // anonymous namespace

void dynd::make_buffered_chain_unary_kernel(std::deque<kernel_instance<unary_operation_pair_t>>& kernels,
                    std::deque<dtype>& dtypes, kernel_instance<unary_operation_pair_t>& out_kernel)
{
    if (kernels.size() != dtypes.size() - 1) {
        std::stringstream ss;
        ss << "make_buffered_nchain_unary_kernel: the size of 'kernels' (" << kernels.size()
            << ") must be one less than 'dtypes' (" << dtypes.size() << ")";
        throw std::runtime_error(ss.str());
    }

    switch (kernels.size()) {
        case 1:
            kernels[0].swap(out_kernel);
            break;
        case 2: {
            if (kernels[0].kernel.single != NULL && kernels[1].kernel.single != NULL) {
                out_kernel.kernel.single = &buffered_2chain_unary_kernel::single;
                if (kernels[0].kernel.contig != NULL && kernels[1].kernel.contig != NULL) {
                    out_kernel.kernel.contig = &buffered_2chain_unary_kernel::contig;
                } else {
                    out_kernel.kernel.contig = NULL;
                }
                make_auxiliary_data<buffered_2chain_unary_kernel_auxdata>(out_kernel.auxdata);
                buffered_2chain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_2chain_unary_kernel_auxdata>();

                auxdata.buf.allocate(dtypes[1]);
                auxdata.src_item_size = dtypes.front().get_element_size();
                auxdata.dst_item_size = dtypes.back().get_element_size();

                auxdata.kernels[0].swap(kernels[0]);
                auxdata.kernels[1].swap(kernels[1]);
            } else {
                out_kernel.kernel = unary_operation_pair_t();
                out_kernel.auxdata.free();
            }
            break;
        }
        case 3: {
            if (kernels[0].kernel.single != NULL && kernels[1].kernel.single != NULL &&
                            kernels[2].kernel.single != NULL) {
                out_kernel.kernel.single = &buffered_3chain_unary_kernel::single;
                if (kernels[0].kernel.contig != NULL && kernels[1].kernel.contig != NULL &&
                            kernels[2].kernel.contig != NULL) {
                    out_kernel.kernel.contig = &buffered_2chain_unary_kernel::contig;
                } else {
                    out_kernel.kernel.contig = NULL;
                }
                make_auxiliary_data<buffered_3chain_unary_kernel_auxdata>(out_kernel.auxdata);
                buffered_3chain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_3chain_unary_kernel_auxdata>();

                auxdata.bufs[0].allocate(dtypes[1]);
                auxdata.bufs[1].allocate(dtypes[2]);
                auxdata.src_item_size = dtypes.front().get_element_size();
                auxdata.dst_item_size = dtypes.back().get_element_size();

                auxdata.kernels[0].swap(kernels[0]);
                auxdata.kernels[1].swap(kernels[1]);
                auxdata.kernels[2].swap(kernels[2]);
            } else {
                out_kernel.kernel = unary_operation_pair_t();
                out_kernel.auxdata.free();
            }
            break;
        }
        default: {
            bool all_contig_nonnull = true;
            for (size_t i = 0, i_end = kernels.size(); i != i_end; ++i) {
                if (kernels[i].kernel.single == NULL) {
                    out_kernel.kernel = unary_operation_pair_t();
                    out_kernel.auxdata.free();
                    return;
                } else if (kernels[i].kernel.contig == NULL) {
                    all_contig_nonnull = false;
                }
            }
            out_kernel.kernel.single = &buffered_nchain_unary_kernel::single;
            if (all_contig_nonnull) {
                out_kernel.kernel.contig = &buffered_nchain_unary_kernel::contig;
            }
            make_auxiliary_data<buffered_nchain_unary_kernel_auxdata>(out_kernel.auxdata);
            buffered_nchain_unary_kernel_auxdata &auxdata = out_kernel.auxdata.get<buffered_nchain_unary_kernel_auxdata>();
            auxdata.init((int)kernels.size() - 1);

            for (size_t i = 0, i_end = kernels.size()-1; i != i_end; ++i) {
                auxdata.bufs[i].allocate(dtypes[i+1]);
            }
            auxdata.src_item_size = dtypes.front().get_element_size();
            auxdata.dst_item_size = dtypes.back().get_element_size();

            for (size_t i = 0; i < kernels.size(); ++i) {
                auxdata.kernels[i].swap(kernels[i]);
            }
            break;
        }
    }
}

void dynd::push_front_dtype_storage_to_value_kernels(const dynd::dtype& dt,
                    const eval::eval_context *ectx,
                    std::deque<kernel_instance<unary_operation_pair_t>>& out_kernels,
                    std::deque<dtype>& out_dtypes)
{
    const dtype* front_dt = &dt;
    const extended_expression_dtype* front_dt_extended = static_cast<const extended_expression_dtype *>(front_dt->extended());
    const dtype* next_dt = &static_cast<const extended_expression_dtype *>(dt.extended())->get_operand_dtype();
    if (next_dt->get_kind() != expression_kind) {
        // Special case when there is just one
        if (out_kernels.empty()) {
            out_dtypes.push_front(dt.value_dtype());
        }
        out_dtypes.push_front(dt.storage_dtype());
        out_kernels.push_front(kernel_instance<unary_operation_pair_t>());
        front_dt_extended->get_operand_to_value_kernel(ectx, out_kernels.front());
    } else {
        // The final element size, if not yet provided
        if (out_kernels.empty()) {
            out_dtypes.push_front(dt.value_dtype());
        }
        do {
            // Add this kernel to the deque
            out_dtypes.push_front(next_dt->value_dtype());
            out_kernels.push_front(kernel_instance<unary_operation_pair_t>());
            front_dt_extended->get_operand_to_value_kernel(ectx, out_kernels.front());
            // Shift to the next dtype
            front_dt = next_dt;
            front_dt_extended = static_cast<const extended_expression_dtype *>(front_dt->extended());
            next_dt = &front_dt_extended->get_operand_dtype();
        } while (next_dt->get_kind() == expression_kind);
        // Add the final kernel from the source
        out_dtypes.push_front(*next_dt);
        out_kernels.push_front(kernel_instance<unary_operation_pair_t>());
        front_dt_extended->get_operand_to_value_kernel(ectx, out_kernels.front());
    }
}

void dynd::push_back_dtype_value_to_storage_kernels(const dynd::dtype& dt,
                    const eval::eval_context *ectx,
                    std::deque<kernel_instance<unary_operation_pair_t>>& out_kernels,
                    std::deque<dtype>& out_dtypes)
{
    const dtype* back_dt = &dt;
    const extended_expression_dtype* back_dt_extended = static_cast<const extended_expression_dtype *>(back_dt->extended());
    const dtype* next_dt = &back_dt_extended->get_operand_dtype();
    if (next_dt->get_kind() != expression_kind) {
        // Special case when there is just one
        if (out_kernels.empty()) {
            out_dtypes.push_back(dt.value_dtype());
        }
        out_dtypes.push_back(dt.storage_dtype());
        out_kernels.push_back(kernel_instance<unary_operation_pair_t>());
        back_dt_extended->get_value_to_operand_kernel(ectx, out_kernels.back());
    } else {
        // The first element size, if not yet provided
        if (out_kernels.empty()) {
            out_dtypes.push_back(dt.value_dtype());
        }
        do {
            // Add this kernel to the deque
            out_dtypes.push_back(next_dt->value_dtype());
            out_kernels.push_back(kernel_instance<unary_operation_pair_t>());
            back_dt_extended->get_value_to_operand_kernel(ectx, out_kernels.back());
            // Shift to the next dtype
            back_dt = next_dt;
            back_dt_extended = static_cast<const extended_expression_dtype *>(back_dt->extended());
            next_dt = &back_dt_extended->get_operand_dtype();
        } while (next_dt->get_kind() == expression_kind);
        // Add the final kernel from the source
        out_dtypes.push_back(*next_dt);
        out_kernels.push_back(kernel_instance<unary_operation_pair_t>());
        back_dt_extended->get_value_to_operand_kernel(ectx, out_kernels.back());
    }
}
