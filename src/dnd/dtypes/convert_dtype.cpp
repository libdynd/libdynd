//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dnd;

dnd::convert_dtype::convert_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode)
    : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype), m_errmode(errmode)
{
    // An alternative to this error would be to use value_dtype.value_dtype(), cutting
    // away the expression part of the given value_dtype.
    if (m_value_dtype.kind() == expression_kind) {
        std::stringstream ss;
        ss << "convert_dtype: The destination dtype " << m_value_dtype << " should not be an expression_kind";
        throw std::runtime_error(ss.str());
    }

    // Initialize the kernels
    assign_error_mode errmode_to_value, errmode_to_operand;
    if (errmode != assign_error_none) {
        errmode_to_value = ::dnd::is_lossless_assignment(m_value_dtype, m_operand_dtype) ? assign_error_none : errmode;
        errmode_to_operand = ::dnd::is_lossless_assignment(m_operand_dtype, m_value_dtype) ? assign_error_none : errmode;
    } else {
        errmode_to_value = assign_error_none;
        errmode_to_operand = assign_error_none;
    }

    if (errmode_to_value != assign_error_default) {
        ::dnd::get_dtype_assignment_kernel(m_value_dtype, m_operand_dtype.value_dtype(), errmode_to_value, NULL, m_to_value_kernel);
    } else {
        m_to_value_kernel.specializations = NULL;
    }
    if (errmode_to_operand != assign_error_default) {
        ::dnd::get_dtype_assignment_kernel(m_operand_dtype.value_dtype(), m_value_dtype, errmode_to_value, NULL, m_to_operand_kernel);
    } else {
        m_to_operand_kernel.specializations = NULL;
    }
}

void dnd::convert_dtype::print_element(std::ostream& DND_UNUSED(o), const char *DND_UNUSED(data)) const
{
    throw runtime_error("internal error: convert_dtype::print_element isn't supposed to be called");
}

void dnd::convert_dtype::print_dtype(std::ostream& o) const
{
    o << "convert<to=" << m_value_dtype << ", from=" << m_operand_dtype;
    if (m_errmode != assign_error_default) {
        o << ", errmode=" << m_errmode;
    }
    o << ">";
}

bool dnd::convert_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::convert_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != convert_type_id) {
        return false;
    } else {
        const convert_dtype *dt = static_cast<const convert_dtype*>(&rhs);
        return m_errmode == dt->m_errmode &&
            m_value_dtype == dt->m_value_dtype &&
            m_operand_dtype == dt->m_operand_dtype;
    }
}

void dnd::convert_dtype::get_operand_to_value_kernel(const eval_context *ectx,
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    if (m_to_value_kernel.specializations != NULL) {
        out_borrowed_kernel.borrow_from(m_to_value_kernel);
    } else if (ectx != NULL) {
        // If the kernel wasn't set, errmode is assign_error_default, so we must use the eval_context
        ::dnd::get_dtype_assignment_kernel(m_value_dtype, m_operand_dtype.value_dtype(),
                            ectx->default_assign_error_mode, ectx, out_borrowed_kernel);
    } else {
        // An evaluation context is needed to get the kernel, set the output to NULL
        out_borrowed_kernel.specializations = NULL;
    }
}

void dnd::convert_dtype::get_value_to_operand_kernel(const eval_context *ectx,
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    if (m_to_operand_kernel.specializations != NULL) {
        out_borrowed_kernel.borrow_from(m_to_operand_kernel);
    } else if (ectx != NULL) {
        // If the kernel wasn't set, errmode is assign_error_default, so we must use the eval_context
        ::dnd::get_dtype_assignment_kernel(m_operand_dtype.value_dtype(), m_value_dtype,
                            ectx->default_assign_error_mode, ectx, out_borrowed_kernel);
    } else {
        // An evaluation context is needed to get the kernel, set the output to NULL
        out_borrowed_kernel.specializations = NULL;
    }
}

dtype dnd::convert_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return dtype(make_shared<convert_dtype>(m_value_dtype,
                        m_operand_dtype.extended()->with_replaced_storage_dtype(replacement_dtype),
                        m_errmode));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the conversion's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(make_shared<convert_dtype>(m_value_dtype, replacement_dtype, m_errmode));
    }
}
