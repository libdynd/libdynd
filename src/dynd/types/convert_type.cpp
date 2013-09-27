//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

convert_type::convert_type(const ndt::type& value_type, const ndt::type& operand_type, assign_error_mode errmode)
    : base_expression_type(convert_type_id, expression_kind, operand_type.get_data_size(),
                        operand_type.get_data_alignment(),
                        inherited_flags(value_type.get_flags(), operand_type.get_flags()),
                        operand_type.get_metadata_size(),
                        value_type.get_ndim()),
                m_value_type(value_type), m_operand_type(operand_type), m_errmode(errmode)
{
    // An alternative to this error would be to use value_type.value_type(), cutting
    // away the expression part of the given value_type.
    if (m_value_type.get_kind() == expression_kind) {
        std::stringstream ss;
        ss << "convert_type: The destination type " << m_value_type;
        ss << " should not be an expression_kind";
        throw std::runtime_error(ss.str());
    }

    // Initialize the kernels
    if (errmode != assign_error_none) {
        m_errmode_to_value = ::dynd::is_lossless_assignment(m_value_type, m_operand_type)
                        ? assign_error_none : errmode;
        m_errmode_to_operand = ::dynd::is_lossless_assignment(m_operand_type, m_value_type)
                        ? assign_error_none : errmode;
    } else {
        m_errmode_to_value = assign_error_none;
        m_errmode_to_operand = assign_error_none;
    }
}

convert_type::~convert_type()
{
}


void convert_type::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: convert_type::print_data isn't supposed to be called");
}

void convert_type::print_type(std::ostream& o) const
{
    o << "convert<to=" << m_value_type << ", from=" << m_operand_type;
    if (m_errmode != assign_error_default) {
        o << ", errmode=" << m_errmode;
    }
    o << ">";
}

void convert_type::get_shape(size_t ndim, size_t i, intptr_t *out_shape,
                const char *metadata, const char *DYND_UNUSED(data)) const
{
    // Get the shape from the operand type
    if (!m_operand_type.is_builtin()) {
        m_operand_type.extended()->get_shape(ndim, i, out_shape, metadata, NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << ndt::type(this, true);
        throw runtime_error(ss.str());
    }
}

bool convert_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    // Treat this type as the value type for whether assignment is always lossless
    if (src_tp.extended() == this) {
        return dynd::is_lossless_assignment(dst_tp, m_value_type);
    } else {
        return dynd::is_lossless_assignment(m_value_type, src_tp);
    }
}

bool convert_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != convert_type_id) {
        return false;
    } else {
        const convert_type *dt = static_cast<const convert_type*>(&rhs);
        return m_errmode == dt->m_errmode &&
            m_value_type == dt->m_value_type &&
            m_operand_type == dt->m_operand_type;
    }
}

ndt::type convert_type::with_replaced_storage_type(const ndt::type& replacement_type) const
{
    if (m_operand_type.get_kind() == expression_kind) {
        return ndt::type(new convert_type(m_value_type,
                        static_cast<const base_expression_type *>(m_operand_type.extended())->with_replaced_storage_type(replacement_type),
                        m_errmode), false);
    } else {
        if (m_operand_type != replacement_type.value_type()) {
            std::stringstream ss;
            ss << "Cannot chain expression types, because the conversion's storage type, " << m_operand_type;
            ss << ", does not match the replacement's value type, " << replacement_type.value_type();
            throw std::runtime_error(ss.str());
        }
        return ndt::type(new convert_type(m_value_type, replacement_type, m_errmode), false);
    }
}

size_t convert_type::make_operand_to_value_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(out, offset_out,
                    m_value_type, dst_metadata,
                    m_operand_type.value_type(), src_metadata,
                    kernreq, m_errmode_to_value, ectx);
}

size_t convert_type::make_value_to_operand_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(out, offset_out,
                    m_operand_type.value_type(), dst_metadata,
                    m_value_type, src_metadata,
                    kernreq, m_errmode_to_value, ectx);
}
