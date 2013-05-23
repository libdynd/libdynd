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

byteswap_dtype::byteswap_dtype(const dtype& value_dtype)
    : base_expression_dtype(byteswap_type_id, expression_kind, value_dtype.get_data_size(),
                            value_dtype.get_alignment(), dtype_flag_scalar, 0),
                    m_value_dtype(value_dtype),
                    m_operand_dtype(make_fixedbytes_dtype(value_dtype.get_data_size(), value_dtype.get_alignment()))
{
    if (!value_dtype.is_builtin()) {
        throw std::runtime_error("byteswap_dtype: Only built-in dtypes are supported presently");
    }
}

byteswap_dtype::byteswap_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : base_expression_dtype(byteswap_type_id, expression_kind, operand_dtype.get_data_size(),
                    operand_dtype.get_alignment(), dtype_flag_scalar, 0),
            m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
    // Only a bytes dtype be the operand to the byteswap
    if (operand_dtype.value_dtype().get_type_id() != fixedbytes_type_id) {
        std::stringstream ss;
        ss << "byteswap_dtype: The operand to the dtype must have a value dtype of bytes, not " << operand_dtype.value_dtype();
        throw std::runtime_error(ss.str());
    }
    // Automatically realign if needed
    if (operand_dtype.value_dtype().get_alignment() < value_dtype.get_alignment()) {
        m_operand_dtype = make_view_dtype(operand_dtype, make_fixedbytes_dtype(operand_dtype.get_data_size(), value_dtype.get_alignment()));
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
    o << "byteswap<" << m_value_dtype;
    if (m_operand_dtype.get_type_id() != fixedbytes_type_id) {
        o << ", " << m_operand_dtype;
    }
    o << ">";
}

void byteswap_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_operand_dtype.is_builtin()) {
        m_operand_dtype.extended()->get_shape(i, out_shape);
    }
}

bool byteswap_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool byteswap_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != byteswap_type_id) {
        return false;
    } else {
        const byteswap_dtype *dt = static_cast<const byteswap_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

dtype byteswap_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() != expression_kind) {
        // If there's no expression in the operand, just try substituting (the constructor will error-check)
        return dtype(new byteswap_dtype(m_value_dtype, replacement_dtype), false);
    } else {
        // With an expression operand, replace it farther down the chain
        return dtype(new byteswap_dtype(m_value_dtype,
                reinterpret_cast<const base_expression_dtype *>(replacement_dtype.extended())->with_replaced_storage_dtype(replacement_dtype)), false);
    }
}

size_t byteswap_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    if(m_value_dtype.get_kind() != complex_kind) {
        return make_byteswap_assignment_function(out, offset_out,
                        m_value_dtype.get_data_size(), m_value_dtype.get_alignment(),
                        kernreq);
    } else {
        return make_pairwise_byteswap_assignment_function(out, offset_out,
                        m_value_dtype.get_data_size(), m_value_dtype.get_alignment(),
                        kernreq);
    }
}

size_t byteswap_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx)) const
{
    if(m_value_dtype.get_kind() != complex_kind) {
        return make_byteswap_assignment_function(out, offset_out,
                        m_value_dtype.get_data_size(), m_value_dtype.get_alignment(),
                        kernreq);
    } else {
        return make_pairwise_byteswap_assignment_function(out, offset_out,
                        m_value_dtype.get_data_size(), m_value_dtype.get_alignment(),
                        kernreq);
    }
}
