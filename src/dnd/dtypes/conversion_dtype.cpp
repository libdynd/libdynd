//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dnd;


void dnd::conversion_dtype::print_data(std::ostream& DND_UNUSED(o), const dtype& DND_UNUSED(dt),
                        const char *DND_UNUSED(data), intptr_t DND_UNUSED(stride),
                        intptr_t DND_UNUSED(size), const char *DND_UNUSED(separator)) const
{
    throw runtime_error("internal error: conversion_dtype::print_data isn't supposed to be called");
    /*
    buffer_storage buf(m_value_dtype, size);
    // TODO: This doesn't work with multiple nested expression_kind dtypes
    kernel_instance<unary_operation_t> assign =
                get_dtype_strided_assign_operation(
                                        m_value_dtype, m_value_dtype.itemsize(),
                                        m_operand_dtype, stride,
                                        m_errmode);
    while (size > 0) {
        intptr_t blocksize = buf.element_count();
        if (blocksize > size) {
            blocksize = size;
        }

        assign.first(buf.storage(), m_value_dtype.itemsize(), data, stride, blocksize, assign.second.get());
        m_value_dtype.print_data(o, buf.storage(), m_value_dtype.itemsize(), blocksize, separator);

        data += blocksize * stride;
        size -= blocksize;
        if (size > 0) {
            o << separator;
        }
    }
    */
}

void dnd::conversion_dtype::print(std::ostream& o) const
{
    o << "convert<to=" << m_value_dtype << ", from=" << m_operand_dtype << ", errmode=" << m_errmode << ">";
}

bool dnd::conversion_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::conversion_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != conversion_type_id) {
        return false;
    } else {
        const conversion_dtype *dt = static_cast<const conversion_dtype*>(&rhs);
        return m_errmode == dt->m_errmode &&
            m_value_dtype == dt->m_value_dtype &&
            m_operand_dtype == dt->m_operand_dtype;
    }
}

void dnd::conversion_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_dtype_assignment_kernel(m_value_dtype, dst_fixedstride,
                                m_operand_dtype.value_dtype(), src_fixedstride,
                                m_no_errors_to_value ? assign_error_none : m_errmode,
                                out_kernel);
}

void dnd::conversion_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    get_dtype_assignment_kernel(m_operand_dtype.value_dtype(), dst_fixedstride,
                                m_value_dtype, src_fixedstride,
                                m_no_errors_to_storage ? assign_error_none : m_errmode,
                                out_kernel);
}

dtype dnd::conversion_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return dtype(make_shared<conversion_dtype>(m_value_dtype,
                        m_operand_dtype.extended()->with_replaced_storage_dtype(replacement_dtype),
                        m_errmode));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the conversion's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(make_shared<conversion_dtype>(m_value_dtype, replacement_dtype, m_errmode));
    }
}
