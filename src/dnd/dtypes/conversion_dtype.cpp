//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/buffer_storage.hpp>

using namespace std;
using namespace dnd;


void dnd::conversion_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    throw runtime_error("internal error: conversion_dtype::print_data isn't supposed to be called");
    /*
    buffer_storage buf(m_value_dtype, size);
    // TODO: This doesn't work with multiple nested expression_kind dtypes
    std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > assign =
                get_dtype_strided_assign_operation(
                                        m_value_dtype, m_value_dtype.itemsize(),
                                        m_storage_dtype, stride,
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
    o << "convert<to=" << m_value_dtype << ", from=" << m_storage_dtype << ", errmode=" << m_errmode << ">";
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
            m_storage_dtype == dt->m_storage_dtype;
    }
}

std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > dnd::conversion_dtype::get_expression_to_value(intptr_t dst_fixedstride, intptr_t src_fixedstride)
{
    return get_dtype_strided_assign_operation(m_value_dtype, dst_fixedstride,
                                m_storage_dtype.value_dtype(), src_fixedstride,
                                m_no_errors_to_value ? assign_error_none : m_errmode);
}

std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> > dnd::conversion_dtype::get_expression_from_value(intptr_t dst_fixedstride, intptr_t src_fixedstride)
{
    return get_dtype_strided_assign_operation(m_storage_dtype.value_dtype(), dst_fixedstride,
                                m_value_dtype, src_fixedstride,
                                m_no_errors_to_storage ? assign_error_none : m_errmode);
}

