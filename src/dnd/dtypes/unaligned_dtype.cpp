//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/unaligned_dtype.hpp>
#include <dnd/kernels/unaligned_kernels.hpp>
#include <dnd/dtype_assign.hpp>

using namespace std;
using namespace dnd;


void dnd::unaligned_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    throw runtime_error("internal error: unaligned_dtype::print_data isn't supposed to be called");
}

void dnd::unaligned_dtype::print(std::ostream& o) const
{
    o << "unaligned<" << m_value_dtype << ">";
}

bool dnd::unaligned_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::unaligned_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != unaligned_type_id) {
        return false;
    } else {
        const unaligned_dtype *dt = static_cast<const unaligned_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

void dnd::unaligned_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_unaligned_copy_kernel(m_value_dtype.itemsize(), dst_fixedstride, src_fixedstride, out_kernel);
}

void dnd::unaligned_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_unaligned_copy_kernel(m_value_dtype.itemsize(), dst_fixedstride, src_fixedstride, out_kernel);
}

dtype dnd::unaligned_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    return make_unaligned_dtype(
                m_value_dtype.extended()->with_replaced_storage_dtype(replacement_dtype));
}
