//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/kernels/single_compare_kernel_instance.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

// Static instance of a void pointer to use as the storage of pointer dtypes
dtype dynd::pointer_dtype::m_void_pointer_dtype(new void_pointer_dtype());


dynd::pointer_dtype::pointer_dtype(const dtype& target_dtype)
    : m_target_dtype(target_dtype)
{
    // I'm not 100% sure how blockref pointer dtypes should interact with
    // the computational subsystem, the details will have to shake out
    // when we want to actually do something with them.
    if (target_dtype.kind() == expression_kind && target_dtype.type_id() != pointer_type_id) {
        stringstream ss;
        ss << "A pointer dtype's target cannot be the expression dtype ";
        ss << target_dtype;
        throw runtime_error(ss.str());
    }
}

void dynd::pointer_dtype::print_element(std::ostream& o, const char *data) const
{
    const char *target_data = *reinterpret_cast<const char * const *>(data);
    m_target_dtype.print_element(o, target_data);
}

void dynd::pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<" << m_target_dtype << ">";

}

dtype dynd::pointer_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this);
    } else {
        return m_target_dtype.apply_linear_index(nindices, indices, current_i, root_dt);
    }
}

void dynd::pointer_dtype::get_shape(int i, std::vector<intptr_t>& out_shape) const
{
    if (m_target_dtype.extended()) {
        m_target_dtype.extended()->get_shape(i, out_shape);
    }
}

bool dynd::pointer_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        return ::is_lossless_assignment(m_target_dtype, src_dt);
    } else {
        return ::is_lossless_assignment(dst_dt, m_target_dtype);
    }
}

void dynd::pointer_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const {
    throw std::runtime_error("pointer_dtype::get_single_compare_kernel not supported yet");
}

namespace {
    struct pointer_dst_assign_kernel_auxdata {
        kernel_instance<unary_operation_t> m_assign_kernel;
    };

    struct pointer_dst_assign_kernel {
        static void general_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const pointer_dst_assign_kernel_auxdata& ad = get_auxiliary_data<pointer_dst_assign_kernel_auxdata>(auxdata);
            unary_operation_t child_op = ad.m_assign_kernel.kernel;
            const AuxDataBase *child_ad = ad.m_assign_kernel.auxdata;
            for (intptr_t i = 0; i < count; ++i) {
                char *dst_target = *reinterpret_cast<char **>(dst);
                child_op(dst_target, 0, src, 0, 1, child_ad);

                dst += dst_stride;
                src += src_stride;
            }
        }

        static void scalar_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t DND_UNUSED(src_stride),
                            intptr_t, const AuxDataBase *auxdata)
        {
            const pointer_dst_assign_kernel_auxdata& ad = get_auxiliary_data<pointer_dst_assign_kernel_auxdata>(auxdata);
            char *dst_target = *reinterpret_cast<char **>(dst);
            ad.m_assign_kernel.kernel(dst_target, 0, src, 0, 1, ad.m_assign_kernel.auxdata);
        }

        static void contiguous_kernel(char *dst, intptr_t DND_UNUSED(dst_stride), const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
        {
            const pointer_dst_assign_kernel_auxdata& ad = get_auxiliary_data<pointer_dst_assign_kernel_auxdata>(auxdata);
            unary_operation_t child_op = ad.m_assign_kernel.kernel;
            const AuxDataBase *child_ad = ad.m_assign_kernel.auxdata;
            char **dst_cached = reinterpret_cast<char **>(dst);

            for (intptr_t i = 0; i < count; ++i) {
                child_op(*dst_cached, 0, src, 0, 1, child_ad);

                ++dst_cached;
                src += src_stride;
            }
        }
    };
} // anonymous namespace

bool dynd::pointer_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != pointer_type_id) {
        return false;
    } else {
        const pointer_dtype *dt = static_cast<const pointer_dtype*>(&rhs);
        return m_target_dtype == dt->m_target_dtype;
    }
}

void dynd::pointer_dtype::get_operand_to_value_kernel(const eval::eval_context * /*ectx*/,
                        unary_specialization_kernel_instance& /*out_borrowed_kernel*/) const
{
    throw runtime_error("TODO: implement pointer_dtype::get_operand_to_value_kernel");
}
void dynd::pointer_dtype::get_value_to_operand_kernel(const eval::eval_context * /*ectx*/,
                        unary_specialization_kernel_instance& /*out_borrowed_kernel*/) const
{
    throw runtime_error("TODO: implement pointer_dtype::get_value_to_operand_kernel");
}

dtype dynd::pointer_dtype::with_replaced_storage_dtype(const dtype& /*replacement_dtype*/) const
{
    throw runtime_error("TODO: implement pointer_dtype::with_replaced_storage_dtype");
}
