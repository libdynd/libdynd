//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dnd;

dnd::view_dtype::view_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
    if (value_dtype.element_size() != operand_dtype.value_dtype().element_size()) {
        std::stringstream ss;
        ss << "view_dtype: Cannot view " << operand_dtype.value_dtype() << " as " << value_dtype << " because they have different sizes";
        throw std::runtime_error(ss.str());
    }
    if (value_dtype.is_object_type() || operand_dtype.is_object_type()) {
        throw std::runtime_error("view_dtype: Only POD dtypes are supported");
    }

    get_pod_dtype_assignment_kernel(m_value_dtype.element_size(),
                    std::min(m_value_dtype.alignment(), m_operand_dtype.alignment()),
                    m_copy_kernel);
}

void dnd::view_dtype::print_data(std::ostream& DND_UNUSED(o), const dtype& DND_UNUSED(dt),
                        const char *DND_UNUSED(data), intptr_t DND_UNUSED(stride),
                        intptr_t DND_UNUSED(size), const char *DND_UNUSED(separator)) const
{
    throw runtime_error("internal error: view_dtype::print_data isn't supposed to be called");
}

void dnd::view_dtype::print_dtype(std::ostream& o) const
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

const unary_specialization_kernel_instance&  dnd::view_dtype::get_operand_to_value_kernel() const
{
    return m_copy_kernel;
}

const unary_specialization_kernel_instance&  dnd::view_dtype::get_value_to_operand_kernel() const
{
    return m_copy_kernel;
}

dtype dnd::view_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return dtype(make_shared<view_dtype>(m_value_dtype,
                        m_operand_dtype.extended()->with_replaced_storage_dtype(replacement_dtype)));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the view's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(make_shared<view_dtype>(m_value_dtype, replacement_dtype));
    }
}
