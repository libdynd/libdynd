//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/property_type.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/builtin_type_properties.hpp>

using namespace std;
using namespace dynd;


property_type::property_type(const ndt::type& operand_type, const std::string& property_name,
                size_t property_index)
    : base_expr_type(property_type_id, expr_kind,
                    operand_type.get_data_size(), operand_type.get_data_alignment(), type_flag_none,
                    operand_type.get_arrmeta_size()),
            m_value_tp(), m_operand_tp(operand_type),
            m_readable(false), m_writable(false),
            m_reversed_property(false),
            m_property_name(property_name),
            m_property_index(property_index)
{
    if (!operand_type.value_type().is_builtin()) {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = m_operand_tp.value_type().extended()->get_elwise_property_index(
                            property_name);
        }
        m_value_tp = m_operand_tp.value_type().extended()->get_elwise_property_type(
                        m_property_index, m_readable, m_writable);
    } else {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = get_builtin_type_elwise_property_index(
                            operand_type.value_type().get_type_id(),
                            property_name);
        }
        m_value_tp = get_builtin_type_elwise_property_type(
                        operand_type.value_type().get_type_id(),
                        m_property_index, m_readable, m_writable);
    }
    m_members.flags = inherited_flags(m_value_tp.get_flags(), m_operand_tp.get_flags());
}

property_type::property_type(const ndt::type& value_tp, const ndt::type& operand_tp,
                const std::string& property_name, size_t property_index)
    : base_expr_type(property_type_id, expr_kind,
                    operand_tp.get_data_size(), operand_tp.get_data_alignment(), type_flag_none,
                    operand_tp.get_arrmeta_size()),
            m_value_tp(value_tp), m_operand_tp(operand_tp),
            m_readable(false), m_writable(false),
            m_reversed_property(true),
            m_property_name(property_name),
            m_property_index(property_index)
{
    if (m_value_tp.get_kind() == expr_kind) {
        stringstream ss;
        ss << "property_type: The destination type " << m_value_tp;
        ss << " should not be an expr_kind";
        throw std::runtime_error(ss.str());
    }

    ndt::type property_dt;
    if (!value_tp.is_builtin()) {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = m_value_tp.extended()->get_elwise_property_index(
                            property_name);
        }
        property_dt = m_value_tp.extended()->get_elwise_property_type(
                        m_property_index, m_writable, m_readable);
    } else {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = get_builtin_type_elwise_property_index(
                            value_tp.get_type_id(), property_name);
        }
        property_dt = get_builtin_type_elwise_property_type(
                        value_tp.get_type_id(), m_property_index,
                            m_writable, m_readable);
    }
    // If the operand type doesn't match the property, add a
    // conversion to the correct type
    if (m_operand_tp.value_type() != property_dt) {
        m_operand_tp = ndt::make_convert(property_dt, m_operand_tp);
    }
    m_members.flags = inherited_flags(m_value_tp.get_flags(), m_operand_tp.get_flags());
}

property_type::~property_type()
{
}

void property_type::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: property_type::print_data isn't supposed to be called");
}

void property_type::print_type(std::ostream& o) const
{
    if (!m_reversed_property) {
        o << "property[name=";
        print_escaped_utf8_string(o, m_property_name, true);
        o << ", operand=" << m_operand_tp << "]";
    } else {
        o << "property[reversed, name=";
        print_escaped_utf8_string(o, m_property_name, true);
        o << ", value=" << m_value_tp;
        o << ", operand=" << m_operand_tp << "]";
    }
}

void property_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *DYND_UNUSED(arrmeta), const char *DYND_UNUSED(data)) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->get_shape(ndim, i, out_shape, NULL, NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << m_value_tp;
        throw runtime_error(ss.str());
    }
}

bool property_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    // Treat this type as the value type for whether assignment is always lossless
    if (src_tp.extended() == this) {
        return dynd::is_lossless_assignment(dst_tp, m_value_tp);
    } else {
        return false;
    }
}

bool property_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != property_type_id) {
        return false;
    } else {
        const property_type *dt = static_cast<const property_type*>(&rhs);
        return m_value_tp == dt->m_value_tp &&
            m_operand_tp == dt->m_operand_tp &&
            m_property_name == dt->m_property_name &&
            m_reversed_property == dt->m_reversed_property;
    }
}

size_t property_type::make_operand_to_value_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const char *dst_arrmeta, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_readable) {
            const ndt::type& ovdt = m_operand_tp.value_type();
            if (!ovdt.is_builtin()) {
                return ovdt.extended()->make_elwise_property_getter_kernel(
                                ckb, ckb_offset,
                                dst_arrmeta,
                                src_arrmeta, m_property_index,
                                kernreq, ectx);
            } else {
                return make_builtin_type_elwise_property_getter_kernel(
                                ckb, ckb_offset,
                                ovdt.get_type_id(),
                                dst_arrmeta,
                                src_arrmeta, m_property_index,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of type " << m_operand_tp;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_readable) {
            if (!m_value_tp.is_builtin()) {
                return m_value_tp.extended()->make_elwise_property_setter_kernel(
                                ckb, ckb_offset,
                                dst_arrmeta, m_property_index,
                                src_arrmeta,
                                kernreq, ectx);
            } else {
                return make_builtin_type_elwise_property_setter_kernel(
                                ckb, ckb_offset,
                                m_value_tp.get_type_id(),
                                dst_arrmeta, m_property_index,
                                src_arrmeta,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dynd array with type " << m_value_tp;
            throw runtime_error(ss.str());
        }
    }
}

size_t property_type::make_value_to_operand_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const char *dst_arrmeta, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_writable) {
            const ndt::type& ovdt = m_operand_tp.value_type();
            if (!ovdt.is_builtin()) {
                return ovdt.extended()->make_elwise_property_setter_kernel(
                                ckb, ckb_offset,
                                dst_arrmeta, m_property_index,
                                src_arrmeta,
                                kernreq, ectx);
            } else {
                return make_builtin_type_elwise_property_setter_kernel(
                                ckb, ckb_offset,
                                ovdt.get_type_id(),
                                dst_arrmeta, m_property_index,
                                src_arrmeta,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dynd array with type " << m_operand_tp;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_writable) {
            if (!m_value_tp.is_builtin()) {
                return m_value_tp.extended()->make_elwise_property_getter_kernel(
                                ckb, ckb_offset,
                                dst_arrmeta,
                                src_arrmeta, m_property_index,
                                kernreq, ectx);
            } else {
                return make_builtin_type_elwise_property_getter_kernel(
                                ckb, ckb_offset,
                                m_value_tp.get_type_id(),
                                dst_arrmeta,
                                src_arrmeta, m_property_index,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of type " << m_value_tp;
            throw runtime_error(ss.str());
        }
    }
}

ndt::type property_type::with_replaced_storage_type(const ndt::type& replacement_type) const
{
    if (m_operand_tp.get_kind() == expr_kind) {
        return ndt::type(new property_type(
                        m_operand_tp.tcast<base_expr_type>()->with_replaced_storage_type(replacement_type),
                        m_property_name), false);
    } else {
        if (m_operand_tp != replacement_type.value_type()) {
            std::stringstream ss;
            ss << "Cannot chain types, because the property's storage type, " << m_operand_tp;
            ss << ", does not match the replacement's value type, " << replacement_type.value_type();
            throw dynd::type_error(ss.str());
        }
        if (!m_reversed_property) {
            return ndt::type(new property_type(replacement_type, m_property_name), false);
        } else {
            return ndt::type(new property_type(m_value_tp, replacement_type, m_property_name), false);
        }
    }
}

void property_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_tp.get_dtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_properties(out_properties, out_count);
    } else {
        get_builtin_type_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
    }
}

void property_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_tp.get_dtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_functions(out_functions, out_count);
    } else {
        //get_builtin_type_dynamic_array_functions(udt.get_type_id(), out_functions, out_count);
    }
}
