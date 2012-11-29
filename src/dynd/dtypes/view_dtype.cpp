//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

dynd::view_dtype::view_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
    if (value_dtype.element_size() != operand_dtype.value_dtype().element_size()) {
        std::stringstream ss;
        ss << "view_dtype: Cannot view " << operand_dtype.value_dtype() << " as " << value_dtype << " because they have different sizes";
        throw std::runtime_error(ss.str());
    }
    if (value_dtype.get_memory_management() != pod_memory_management ||
                    operand_dtype.get_memory_management() != pod_memory_management) {
        throw std::runtime_error("view_dtype: Only POD dtypes are supported");
    }

    get_pod_dtype_assignment_kernel(m_value_dtype.element_size(),
                    std::min(m_value_dtype.alignment(), m_operand_dtype.alignment()),
                    m_copy_kernel);
}

void dynd::view_dtype::print_element(std::ostream& o, const char *data, const char *metadata) const
{
    // Allow calling print_element in the special case that the view
    // is being used just to align the data
    if (m_operand_dtype.get_type_id() == fixedbytes_type_id) {
        switch (m_operand_dtype.element_size()) {
            case 1:
                m_value_dtype.print_element(o, data, metadata);
                return;
            case 2: {
                uint16_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_element(o, reinterpret_cast<const char *>(&tmp), metadata);
                return;
            }
            case 4: {
                uint32_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_element(o, reinterpret_cast<const char *>(&tmp), metadata);
                return;
            }
            case 8: {
                uint64_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_element(o, reinterpret_cast<const char *>(&tmp), metadata);
                return;
            }
            default: {
                vector<char> storage(m_value_dtype.element_size() + m_value_dtype.alignment());
                char *buffer = &storage[0];
                // Make the storage aligned as needed
                buffer = (char *)(((uintptr_t)buffer + (uintptr_t)m_value_dtype.alignment() - 1) & (m_value_dtype.alignment() - 1));
                memcpy(buffer, data, m_value_dtype.element_size());
                m_value_dtype.print_element(o, reinterpret_cast<const char *>(&buffer), metadata);
                return;
            }
        }
    }

    throw runtime_error("internal error: view_dtype::print_element isn't supposed to be called");
}

void dynd::view_dtype::print_dtype(std::ostream& o) const
{
    // Special case printing of alignment to make it more human-readable
    if (m_value_dtype.alignment() != 1 && m_operand_dtype.get_type_id() == fixedbytes_type_id &&
                    m_operand_dtype.alignment() == 1) {
        o << "unaligned<" << m_value_dtype << ">";
    } else {
        o << "view<as=" << m_value_dtype << ", original=" << m_operand_dtype << ">";
    }
}

dtype dynd::view_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        return m_value_dtype.apply_linear_index(nindices, indices, current_i, root_dt);
    }
}

void dynd::view_dtype::get_shape(int i, intptr_t *out_shape) const
{
    if (m_value_dtype.extended()) {
        m_value_dtype.extended()->get_shape(i, out_shape);
    }
}

bool dynd::view_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dynd::view_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != view_type_id) {
        return false;
    } else {
        const view_dtype *dt = static_cast<const view_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

void dynd::view_dtype::get_operand_to_value_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    out_borrowed_kernel.borrow_from(m_copy_kernel);
}

void dynd::view_dtype::get_value_to_operand_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    out_borrowed_kernel.borrow_from(m_copy_kernel);
}

dtype dynd::view_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() == expression_kind) {
        return dtype(new view_dtype(m_value_dtype,
                        static_cast<const extended_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype)));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the view's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(new view_dtype(m_value_dtype, replacement_dtype));
    }
}
