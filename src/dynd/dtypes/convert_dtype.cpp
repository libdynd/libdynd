//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

convert_dtype::convert_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode)
    : base_expression_dtype(convert_type_id, expression_kind, operand_dtype.get_data_size(),
                        operand_dtype.get_alignment(),
                        inherited_flags(value_dtype.get_flags(), operand_dtype.get_flags()),
                        operand_dtype.get_metadata_size(),
                        value_dtype.get_undim()),
                m_value_dtype(value_dtype), m_operand_dtype(operand_dtype), m_errmode(errmode)
{
    // An alternative to this error would be to use value_dtype.value_dtype(), cutting
    // away the expression part of the given value_dtype.
    if (m_value_dtype.get_kind() == expression_kind) {
        std::stringstream ss;
        ss << "convert_dtype: The destination dtype " << m_value_dtype;
        ss << " should not be an expression_kind";
        throw std::runtime_error(ss.str());
    }

    // Initialize the kernels
    if (errmode != assign_error_none) {
        m_errmode_to_value = ::dynd::is_lossless_assignment(m_value_dtype, m_operand_dtype)
                        ? assign_error_none : errmode;
        m_errmode_to_operand = ::dynd::is_lossless_assignment(m_operand_dtype, m_value_dtype)
                        ? assign_error_none : errmode;
    } else {
        m_errmode_to_value = assign_error_none;
        m_errmode_to_operand = assign_error_none;
    }
}

convert_dtype::~convert_dtype()
{
}


void convert_dtype::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: convert_dtype::print_data isn't supposed to be called");
}

void convert_dtype::print_dtype(std::ostream& o) const
{
    o << "convert<to=" << m_value_dtype << ", from=" << m_operand_dtype;
    if (m_errmode != assign_error_default) {
        o << ", errmode=" << m_errmode;
    }
    o << ">";
}

bool convert_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool convert_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != convert_type_id) {
        return false;
    } else {
        const convert_dtype *dt = static_cast<const convert_dtype*>(&rhs);
        return m_errmode == dt->m_errmode &&
            m_value_dtype == dt->m_value_dtype &&
            m_operand_dtype == dt->m_operand_dtype;
    }
}

dtype convert_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() == expression_kind) {
        return dtype(new convert_dtype(m_value_dtype,
                        static_cast<const base_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype),
                        m_errmode), false);
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the conversion's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(new convert_dtype(m_value_dtype, replacement_dtype, m_errmode), false);
    }
}

size_t convert_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(out, offset_out,
                    m_value_dtype, dst_metadata,
                    m_operand_dtype.value_dtype(), src_metadata,
                    kernreq, m_errmode_to_value, ectx);
}

size_t convert_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(out, offset_out,
                    m_operand_dtype.value_dtype(), dst_metadata,
                    m_value_dtype, src_metadata,
                    kernreq, m_errmode_to_value, ectx);
}
