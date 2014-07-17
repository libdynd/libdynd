//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

convert_type::convert_type(const ndt::type &value_type,
                           const ndt::type &operand_type)
    : base_expr_type(
          convert_type_id, expr_kind, operand_type.get_data_size(),
          operand_type.get_data_alignment(),
          inherited_flags(value_type.get_flags(), operand_type.get_flags()),
          operand_type.get_arrmeta_size(), value_type.get_ndim()),
      m_value_type(value_type), m_operand_type(operand_type)
{
    // An alternative to this error would be to use value_type.value_type(), cutting
    // away the expression part of the given value_type.
    if (m_value_type.get_kind() == expr_kind) {
        std::stringstream ss;
        ss << "convert_type: The destination type " << m_value_type;
        ss << " should not be an expr_kind";
        throw dynd::type_error(ss.str());
    }
}

convert_type::~convert_type()
{
}

void convert_type::print_data(std::ostream &DYND_UNUSED(o),
                              const char *DYND_UNUSED(arrmeta),
                              const char *DYND_UNUSED(data)) const
{
    throw runtime_error(
        "internal error: convert_type::print_data isn't supposed to be called");
}

void convert_type::print_type(std::ostream& o) const
{
    o << "convert[to=" << m_value_type << ", from=" << m_operand_type;
    o << "]";
}

void convert_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *arrmeta, const char *DYND_UNUSED(data)) const
{
    // Get the shape from the operand type
    if (!m_operand_type.is_builtin()) {
        m_operand_type.extended()->get_shape(ndim, i, out_shape, arrmeta, NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << ndt::type(this, true);
        throw runtime_error(ss.str());
    }
}

bool convert_type::is_lossless_assignment(const ndt::type &dst_tp,
                                          const ndt::type &src_tp) const
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
        return m_value_type == dt->m_value_type &&
               m_operand_type == dt->m_operand_type;
    }
}

ndt::type convert_type::with_replaced_storage_type(const ndt::type& replacement_type) const
{
    if (m_operand_type.get_kind() == expr_kind) {
        return ndt::type(
            new convert_type(
                m_value_type,
                m_operand_type.tcast<base_expr_type>()
                    ->with_replaced_storage_type(replacement_type)),
            false);
    } else {
        if (m_operand_type != replacement_type.value_type()) {
            std::stringstream ss;
            ss << "Cannot chain expression types, because the conversion's "
                  "storage type, " << m_operand_type
               << ", does not match the replacement's value type, "
               << replacement_type.value_type();
            throw std::runtime_error(ss.str());
        }
        return ndt::type(new convert_type(m_value_type, replacement_type),
                         false);
    }
}

size_t convert_type::make_operand_to_value_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const char *dst_arrmeta, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(ckb, ckb_offset, m_value_type, dst_arrmeta,
                                    m_operand_type.value_type(), src_arrmeta,
                                    kernreq, ectx);
}

size_t convert_type::make_value_to_operand_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const char *dst_arrmeta, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    return ::make_assignment_kernel(ckb, ckb_offset,
                                    m_operand_type.value_type(), dst_arrmeta,
                                    m_value_type, src_arrmeta, kernreq, ectx);
}
