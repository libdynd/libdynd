//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

dynd::view_dtype::view_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : base_expression_dtype(view_type_id, expression_kind, operand_dtype.get_data_size(),
                    operand_dtype.get_alignment(),
                    (value_dtype.get_flags()&dtype_flag_scalar)|(operand_dtype.get_flags()&dtype_flag_zeroinit),
                    operand_dtype.get_metadata_size()),
            m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
    if (value_dtype.get_data_size() != operand_dtype.value_dtype().get_data_size()) {
        std::stringstream ss;
        ss << "view_dtype: Cannot view " << operand_dtype.value_dtype() << " as " << value_dtype << " because they have different sizes";
        throw std::runtime_error(ss.str());
    }
    if (value_dtype.get_memory_management() != pod_memory_management) {
        throw std::runtime_error("view_dtype: Only POD dtypes are supported");
    }
}

view_dtype::~view_dtype()
{
}

void dynd::view_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    // Allow calling print_data in the special case that the view
    // is being used just to align the data
    if (m_operand_dtype.get_type_id() == fixedbytes_type_id) {
        switch (m_operand_dtype.get_data_size()) {
            case 1:
                m_value_dtype.print_data(o, metadata, data);
                return;
            case 2: {
                uint16_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_data(o, metadata, reinterpret_cast<const char *>(&tmp));
                return;
            }
            case 4: {
                uint32_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_data(o, metadata, reinterpret_cast<const char *>(&tmp));
                return;
            }
            case 8: {
                uint64_t tmp;
                memcpy(&tmp, data, sizeof(tmp));
                m_value_dtype.print_data(o, metadata, reinterpret_cast<const char *>(&tmp));
                return;
            }
            default: {
                vector<char> storage(m_value_dtype.get_data_size() + m_value_dtype.get_alignment());
                char *buffer = &storage[0];
                // Make the storage aligned as needed
                buffer = (char *)(((uintptr_t)buffer + (uintptr_t)m_value_dtype.get_alignment() - 1) & (m_value_dtype.get_alignment() - 1));
                memcpy(buffer, data, m_value_dtype.get_data_size());
                m_value_dtype.print_data(o, metadata, reinterpret_cast<const char *>(&buffer));
                return;
            }
        }
    }

    throw runtime_error("internal error: view_dtype::print_data isn't supposed to be called");
}

void dynd::view_dtype::print_dtype(std::ostream& o) const
{
    // Special case printing of alignment to make it more human-readable
    if (m_value_dtype.get_alignment() != 1 && m_operand_dtype.get_type_id() == fixedbytes_type_id &&
                    m_operand_dtype.get_alignment() == 1) {
        o << "unaligned<" << m_value_dtype << ">";
    } else {
        o << "view<as=" << m_value_dtype << ", original=" << m_operand_dtype << ">";
    }
}

void dynd::view_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
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

bool dynd::view_dtype::operator==(const base_dtype& rhs) const
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

dtype dynd::view_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() == expression_kind) {
        return dtype(new view_dtype(m_value_dtype,
                        static_cast<const base_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype)), false);
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the view's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(new view_dtype(m_value_dtype, replacement_dtype), false);
    }
}

size_t view_dtype::make_operand_to_value_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    return ::make_pod_dtype_assignment_kernel(out, offset_out,
                    m_value_dtype.get_data_size(),
                    std::min(m_value_dtype.get_alignment(), m_operand_dtype.get_alignment()),
                    kernreq);
}

size_t view_dtype::make_value_to_operand_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    return ::make_pod_dtype_assignment_kernel(out, offset_out,
                    m_value_dtype.get_data_size(),
                    std::min(m_value_dtype.get_alignment(), m_operand_dtype.get_alignment()),
                    kernreq);
}
