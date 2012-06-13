//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dnd;


void dnd::view_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    throw runtime_error("internal error: view_dtype::print_data isn't supposed to be called");
}

void dnd::view_dtype::print(std::ostream& o) const
{
    o << "view<as=" << m_value_dtype << ", original=" << m_operand_dtype << ">";
}

bool dnd::view_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::view_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != view_type_id) {
        return false;
    } else {
        const view_dtype *dt = static_cast<const view_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

void dnd::view_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_pod_dtype_assignment_kernel(m_value_dtype.itemsize(),
                    std::min(m_value_dtype.alignment(), m_operand_dtype.alignment()),
                    dst_fixedstride, src_fixedstride, out_kernel);
}

void dnd::view_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_pod_dtype_assignment_kernel(m_value_dtype.itemsize(),
                    std::min(m_value_dtype.alignment(), m_operand_dtype.alignment()),
                    dst_fixedstride, src_fixedstride, out_kernel);
}

dtype dnd::view_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return make_shared<view_dtype>(m_value_dtype,
                        m_operand_dtype.extended()->with_replaced_storage_dtype(replacement_dtype));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the view's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return make_shared<view_dtype>(m_value_dtype, replacement_dtype);
    }
}
