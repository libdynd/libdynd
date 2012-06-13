//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/align_dtype.hpp>
#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/kernels/alignment_kernels.hpp>
#include <dnd/dtype_assign.hpp>

using namespace std;
using namespace dnd;


void dnd::align_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    throw runtime_error("internal error: align_dtype::print_data isn't supposed to be called");
}

void dnd::align_dtype::print(std::ostream& o) const
{
    o << "align<"<< m_value_dtype.alignment() << ", " << m_operand_dtype << ">";
}

bool dnd::align_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::align_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != align_type_id) {
        return false;
    } else {
        const align_dtype *dt = static_cast<const align_dtype*>(&rhs);
        return m_value_dtype.alignment() == dt->m_value_dtype.alignment() &&
                    m_operand_dtype == dt->m_operand_dtype;
    }
}

void dnd::align_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_unaligned_copy_kernel(m_value_dtype.itemsize(), dst_fixedstride, src_fixedstride, out_kernel);
}

void dnd::align_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_unaligned_copy_kernel(m_value_dtype.itemsize(), dst_fixedstride, src_fixedstride, out_kernel);
}

dtype dnd::align_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return make_shared<align_dtype>(m_value_dtype.alignment(),
                        m_operand_dtype.extended()->with_replaced_storage_dtype(replacement_dtype));
    } else {
        if (replacement_dtype.value_dtype().type_id() != align_type_id) {
            std::stringstream ss;
            ss << "align_dtype: Can only chain with a bytes dtype, not with " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        if (replacement_dtype.value_dtype().itemsize() != m_value_dtype.itemsize()) {
            std::stringstream ss;
            ss << "align_dtype: Can only chain with a bytes dtype of unchanged size";
            throw std::runtime_error(ss.str());
        }
        // If the align operation becomes a no-op, drop it
        if (replacement_dtype.alignment() < m_value_dtype.alignment()) {
            return make_shared<align_dtype>(m_value_dtype.alignment(), replacement_dtype);
        } else {
            return replacement_dtype;
        }
    }
}

dtype dnd::make_unaligned_dtype(const dtype& value_dtype)
{
    if (value_dtype.alignment() > 1) {
        // Only do something if it requires alignment
        if (value_dtype.kind() != expression_kind) {
            return make_view_dtype(value_dtype, make_bytes_dtype(value_dtype.itemsize(), 1));
        } else {
            const dtype& sdt = value_dtype.storage_dtype();
            if (sdt.type_id() == bytes_type_id) {
                // If its storage dtype is bytes, substitute an align<> dtype there
                return dtype(value_dtype.extended()->with_replaced_storage_dtype(make_align_dtype(sdt.alignment(), make_bytes_dtype(sdt.itemsize(), 1))));
            } else {
                // Otherwise, substitute a view of unaligned bytes
                return dtype(value_dtype.extended()->with_replaced_storage_dtype(make_view_dtype(sdt, make_bytes_dtype(sdt.itemsize(), 1))));
            }
        }
    } else {
        return value_dtype;
    }
}
