//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

byteswap_dtype::byteswap_dtype(const ndt::type& value_type)
    : base_expression_dtype(byteswap_type_id, expression_kind, value_type.get_data_size(),
                            value_type.get_data_alignment(), type_flag_scalar, 0),
                    m_value_type(value_type),
                    m_operand_type(make_fixedbytes_dtype(value_type.get_data_size(), value_type.get_data_alignment()))
{
    if (!value_type.is_builtin()) {
        throw std::runtime_error("byteswap_dtype: Only built-in dtypes are supported presently");
    }
}

byteswap_dtype::byteswap_dtype(const ndt::type& value_type, const ndt::type& operand_type)
    : base_expression_dtype(byteswap_type_id, expression_kind, operand_type.get_data_size(),
                    operand_type.get_data_alignment(), type_flag_scalar, 0),
            m_value_type(value_type), m_operand_type(operand_type)
{
    // Only a bytes dtype be the operand to the byteswap
    if (operand_type.value_type().get_type_id() != fixedbytes_type_id) {
        std::stringstream ss;
        ss << "byteswap_dtype: The operand to the dtype must have a value dtype of bytes, not " << operand_type.value_type();
        throw std::runtime_error(ss.str());
    }
    // Automatically realign if needed
    if (operand_type.value_type().get_data_alignment() < value_type.get_data_alignment()) {
        m_operand_type = make_view_dtype(operand_type,
                        make_fixedbytes_dtype(operand_type.get_data_size(), value_type.get_data_alignment()));
    }
}

byteswap_dtype::~byteswap_dtype()
{
}

void byteswap_dtype::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: byteswap_dtype::print_data isn't supposed to be called");
}

void byteswap_dtype::print_dtype(std::ostream& o) const
{
    o << "byteswap<" << m_value_type;
    if (m_operand_type.get_type_id() != fixedbytes_type_id) {
        o << ", " << m_operand_type;
    }
    o << ">";
}

bool byteswap_dtype::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_type);
    } else {
        return ::dynd::is_lossless_assignment(m_value_type, src_dt);
    }
}

bool byteswap_dtype::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != byteswap_type_id) {
        return false;
    } else {
        const byteswap_dtype *dt = static_cast<const byteswap_dtype*>(&rhs);
        return m_value_type == dt->m_value_type;
    }
}

ndt::type byteswap_dtype::with_replaced_storage_type(const ndt::type& replacement_type) const
{
    if (m_operand_type.get_kind() != expression_kind) {
        // If there's no expression in the operand, just try substituting (the constructor will error-check)
        return ndt::type(new byteswap_dtype(m_value_type, replacement_type), false);
    } else {
        // With an expression operand, replace it farther down the chain
        return ndt::type(new byteswap_dtype(m_value_type,
                reinterpret_cast<const base_expression_dtype *>(replacement_type.extended())->with_replaced_storage_type(replacement_type)), false);
    }
}

size_t byteswap_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    if(m_value_type.get_kind() != complex_kind) {
        return make_byteswap_assignment_function(out, offset_out,
                        m_value_type.get_data_size(), m_value_type.get_data_alignment(),
                        kernreq);
    } else {
        return make_pairwise_byteswap_assignment_function(out, offset_out,
                        m_value_type.get_data_size(), m_value_type.get_data_alignment(),
                        kernreq);
    }
}

size_t byteswap_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    if(m_value_type.get_kind() != complex_kind) {
        return make_byteswap_assignment_function(out, offset_out,
                        m_value_type.get_data_size(), m_value_type.get_data_alignment(),
                        kernreq);
    } else {
        return make_pairwise_byteswap_assignment_function(out, offset_out,
                        m_value_type.get_data_size(), m_value_type.get_data_alignment(),
                        kernreq);
    }
}
