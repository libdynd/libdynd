//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/buffer_storage.hpp>
#include <dnd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dnd;


void dnd::byteswap_dtype::print_data(std::ostream& DND_UNUSED(o), const dtype& DND_UNUSED(dt), const char *DND_UNUSED(data), 
						intptr_t DND_UNUSED(stride), intptr_t DND_UNUSED(size), const char *DND_UNUSED(separator)) const
{
    throw runtime_error("internal error: byteswap_dtype::print_data isn't supposed to be called");
}

void dnd::byteswap_dtype::print(std::ostream& o) const
{
    o << "byteswap<" << m_value_dtype;
    if (m_operand_dtype.type_id() != bytes_type_id) {
        o << ", " << m_operand_dtype;
    }
    o << ">";
}

bool dnd::byteswap_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dnd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dnd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dnd::byteswap_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != byteswap_type_id) {
        return false;
    } else {
        const byteswap_dtype *dt = static_cast<const byteswap_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

void dnd::byteswap_dtype::get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    if(m_value_dtype.kind() != complex_kind) {
        get_byteswap_kernel(m_value_dtype.itemsize(), m_value_dtype.alignment(), dst_fixedstride, src_fixedstride, out_kernel);
    } else {
        get_pairwise_byteswap_kernel(m_value_dtype.itemsize(), m_value_dtype.alignment(), dst_fixedstride, src_fixedstride, out_kernel);
    }
}

void dnd::byteswap_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{
    if(m_value_dtype.kind() != complex_kind) {
        get_byteswap_kernel(m_value_dtype.itemsize(), m_value_dtype.alignment(), dst_fixedstride, src_fixedstride, out_kernel);
    } else {
        get_pairwise_byteswap_kernel(m_value_dtype.itemsize(), m_value_dtype.alignment(), dst_fixedstride, src_fixedstride, out_kernel);
    }
}

dtype dnd::byteswap_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() != expression_kind) {
        // If there's no expression in the operand, just try substituting (the constructor will error-check)
        return dtype(make_shared<byteswap_dtype>(m_value_dtype, replacement_dtype));
    } else {
        // With an expression operand, replace it farther down the chain
        return dtype(make_shared<byteswap_dtype>(m_value_dtype, replacement_dtype.extended()->with_replaced_storage_dtype(replacement_dtype)));
    }
}
