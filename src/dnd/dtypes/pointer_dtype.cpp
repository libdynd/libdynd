//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/pointer_dtype.hpp>
#include <dnd/kernels/single_compare_kernel_instance.hpp>
#include <dnd/kernels/string_assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dnd;

// Static instance of a void pointer to use as the storage of pointer dtypes
dtype dnd::pointer_dtype::m_void_pointer_dtype(make_shared<void_pointer_dtype>());


dnd::pointer_dtype::pointer_dtype(const dtype& target_dtype)
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

void dnd::pointer_dtype::print_element(std::ostream& o, const char *data) const
{
    const char *target_data = *reinterpret_cast<const char * const *>(data);
    m_target_dtype.print_element(o, target_data);
}

void dnd::pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<" << m_target_dtype << ">";

}

bool dnd::pointer_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        return ::is_lossless_assignment(m_target_dtype, src_dt);
    } else {
        return ::is_lossless_assignment(dst_dt, m_target_dtype);
    }
}

void dnd::pointer_dtype::get_single_compare_kernel(single_compare_kernel_instance& DND_UNUSED(out_kernel)) const {
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

bool dnd::pointer_dtype::operator==(const extended_dtype& rhs) const
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
