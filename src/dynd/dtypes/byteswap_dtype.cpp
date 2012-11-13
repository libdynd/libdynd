//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/raw_iteration.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

using namespace std;
using namespace dynd;

dynd::byteswap_dtype::byteswap_dtype(const dtype& value_dtype)
    : m_value_dtype(value_dtype), m_operand_dtype(make_fixedbytes_dtype(value_dtype.element_size(), value_dtype.alignment()))
{
    if (value_dtype.extended() != 0) {
        throw std::runtime_error("byteswap_dtype: Only built-in dtypes are supported presently");
    }

    if(m_value_dtype.kind() != complex_kind) {
        get_byteswap_kernel(value_dtype.element_size(), value_dtype.alignment(), m_byteswap_kernel);
    } else {
        get_pairwise_byteswap_kernel(m_value_dtype.element_size(), m_value_dtype.alignment(), m_byteswap_kernel);
    }
}

dynd::byteswap_dtype::byteswap_dtype(const dtype& value_dtype, const dtype& operand_dtype)
    : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
{
    // Only a bytes dtype be the operand to the byteswap
    if (operand_dtype.value_dtype().type_id() != fixedbytes_type_id) {
        std::stringstream ss;
        ss << "byteswap_dtype: The operand to the dtype must have a value dtype of bytes, not " << operand_dtype.value_dtype();
        throw std::runtime_error(ss.str());
    }
    // Automatically realign if needed
    if (operand_dtype.value_dtype().alignment() < value_dtype.alignment()) {
        m_operand_dtype = make_view_dtype(operand_dtype, make_fixedbytes_dtype(operand_dtype.element_size(), value_dtype.alignment()));
    }

    if(m_value_dtype.kind() != complex_kind) {
        get_byteswap_kernel(value_dtype.element_size(), value_dtype.alignment(), m_byteswap_kernel);
    } else {
        get_pairwise_byteswap_kernel(m_value_dtype.element_size(), m_value_dtype.alignment(), m_byteswap_kernel);
    }
}

void dynd::byteswap_dtype::print_element(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("internal error: byteswap_dtype::print_element isn't supposed to be called");
}

void dynd::byteswap_dtype::print_dtype(std::ostream& o) const
{
    o << "byteswap<" << m_value_dtype;
    if (m_operand_dtype.type_id() != fixedbytes_type_id) {
        o << ", " << m_operand_dtype;
    }
    o << ">";
}

dtype dynd::byteswap_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this);
    } else {
        return m_operand_dtype.apply_linear_index(nindices, indices, current_i, root_dt);
    }
}

void dynd::byteswap_dtype::get_shape(int i, std::vector<intptr_t>& out_shape) const
{
    if (m_operand_dtype.extended()) {
        m_operand_dtype.extended()->get_shape(i, out_shape);
    }
}

bool dynd::byteswap_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool dynd::byteswap_dtype::operator==(const extended_dtype& rhs) const
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

void dynd::byteswap_dtype::get_operand_to_value_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    out_borrowed_kernel.borrow_from(m_byteswap_kernel);
}

void dynd::byteswap_dtype::get_value_to_operand_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        unary_specialization_kernel_instance& out_borrowed_kernel) const
{
    out_borrowed_kernel.borrow_from(m_byteswap_kernel);
}

dtype dynd::byteswap_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.kind() != expression_kind) {
        // If there's no expression in the operand, just try substituting (the constructor will error-check)
        return dtype(new byteswap_dtype(m_value_dtype, replacement_dtype));
    } else {
        // With an expression operand, replace it farther down the chain
        return dtype(new byteswap_dtype(m_value_dtype,
                reinterpret_cast<const extended_expression_dtype *>(replacement_dtype.extended())->with_replaced_storage_dtype(replacement_dtype)));
    }
}
