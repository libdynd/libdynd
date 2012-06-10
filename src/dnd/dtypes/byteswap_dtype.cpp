//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/buffer_storage.hpp>

using namespace std;
using namespace dnd;


void dnd::byteswap_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    throw runtime_error("internal error: byteswap_dtype::print_data isn't supposed to be called");
}

void dnd::byteswap_dtype::print(std::ostream& o) const
{
    o << "byteswap<" << m_value_dtype << ">";
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

}

void dnd::byteswap_dtype::get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const
{

}

