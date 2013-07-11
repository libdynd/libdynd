//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/builtin_dtype_properties.hpp>

using namespace std;
using namespace dynd;


property_dtype::property_dtype(const ndt::type& operand_type, const std::string& property_name,
                size_t property_index)
    : base_expression_type(property_type_id, expression_kind,
                    operand_type.get_data_size(), operand_type.get_data_alignment(), type_flag_none,
                    operand_type.get_metadata_size()),
            m_value_type(), m_operand_type(operand_type),
            m_readable(false), m_writable(false),
            m_reversed_property(false),
            m_property_name(property_name),
            m_property_index(property_index)
{
    if (!operand_type.value_type().is_builtin()) {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = m_operand_type.value_type().extended()->get_elwise_property_index(
                            property_name);
        }
        m_value_type = m_operand_type.value_type().extended()->get_elwise_property_dtype(
                        m_property_index, m_readable, m_writable);
    } else {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = get_builtin_dtype_elwise_property_index(
                            operand_type.value_type().get_type_id(),
                            property_name);
        }
        m_value_type = get_builtin_dtype_elwise_property_dtype(
                        operand_type.value_type().get_type_id(),
                        m_property_index, m_readable, m_writable);
    }
    m_members.flags = inherited_flags(m_value_type.get_flags(), m_operand_type.get_flags());
}

property_dtype::property_dtype(const ndt::type& value_type, const ndt::type& operand_type,
                const std::string& property_name, size_t property_index)
    : base_expression_type(property_type_id, expression_kind,
                    operand_type.get_data_size(), operand_type.get_data_alignment(), type_flag_none,
                    operand_type.get_metadata_size()),
            m_value_type(value_type), m_operand_type(operand_type),
            m_readable(false), m_writable(false),
            m_reversed_property(true),
            m_property_name(property_name),
            m_property_index(property_index)
{
    if (m_value_type.get_kind() == expression_kind) {
        stringstream ss;
        ss << "property_dtype: The destination dtype " << m_value_type;
        ss << " should not be an expression_kind";
        throw std::runtime_error(ss.str());
    }

    ndt::type property_dt;
    if (!value_type.is_builtin()) {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = m_value_type.extended()->get_elwise_property_index(
                            property_name);
        }
        property_dt = m_value_type.extended()->get_elwise_property_dtype(
                        m_property_index, m_writable, m_readable);
    } else {
        if (property_index == numeric_limits<size_t>::max()) {
            m_property_index = get_builtin_dtype_elwise_property_index(
                            value_type.get_type_id(), property_name);
        }
        property_dt = get_builtin_dtype_elwise_property_dtype(
                        value_type.get_type_id(), m_property_index,
                            m_writable, m_readable);
    }
    // If the operand dtype doesn't match the property, add a
    // conversion to the correct dtype
    if (m_operand_type.value_type() != property_dt) {
        m_operand_type = make_convert_dtype(property_dt, m_operand_type);
    }
    m_members.flags = inherited_flags(m_value_type.get_flags(), m_operand_type.get_flags());
}

property_dtype::~property_dtype()
{
}

void property_dtype::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: property_dtype::print_data isn't supposed to be called");
}

void property_dtype::print_dtype(std::ostream& o) const
{
    if (!m_reversed_property) {
        o << "property<name=" << m_property_name << ", operand=" << m_operand_type << ">";
    } else {
        o << "property<reversed, name=" << m_property_name;
        o << ", value=" << m_value_type;
        o << ", operand=" << m_operand_type << ">";
    }
}

void property_dtype::get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *DYND_UNUSED(metadata)) const
{
    if (!m_value_type.is_builtin()) {
        m_value_type.extended()->get_shape(ndim, i, out_shape, NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << m_value_type;
        throw runtime_error(ss.str());
    }
}

bool property_dtype::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dynd::is_lossless_assignment(dst_dt, m_value_type);
    } else {
        return false;
    }
}

bool property_dtype::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != property_type_id) {
        return false;
    } else {
        const property_dtype *dt = static_cast<const property_dtype*>(&rhs);
        return m_value_type == dt->m_value_type &&
            m_operand_type == dt->m_operand_type &&
            m_property_name == dt->m_property_name &&
            m_reversed_property == dt->m_reversed_property;
    }
}

size_t property_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_readable) {
            const ndt::type& ovdt = m_operand_type.value_type();
            if (!ovdt.is_builtin()) {
                return ovdt.extended()->make_elwise_property_getter_kernel(
                                out, offset_out,
                                dst_metadata,
                                src_metadata, m_property_index,
                                kernreq, ectx);
            } else {
                return make_builtin_dtype_elwise_property_getter_kernel(
                                out, offset_out,
                                ovdt.get_type_id(),
                                dst_metadata,
                                src_metadata, m_property_index,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of dtype " << m_operand_type;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_readable) {
            if (!m_value_type.is_builtin()) {
                return m_value_type.extended()->make_elwise_property_setter_kernel(
                                out, offset_out,
                                dst_metadata, m_property_index,
                                src_metadata,
                                kernreq, ectx);
            } else {
                return make_builtin_dtype_elwise_property_setter_kernel(
                                out, offset_out,
                                m_value_type.get_type_id(),
                                dst_metadata, m_property_index,
                                src_metadata,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dynd ndobject with dtype " << m_value_type;
            throw runtime_error(ss.str());
        }
    }
}

size_t property_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_writable) {
            const ndt::type& ovdt = m_operand_type.value_type();
            if (!ovdt.is_builtin()) {
                return ovdt.extended()->make_elwise_property_setter_kernel(
                                out, offset_out,
                                dst_metadata, m_property_index,
                                src_metadata,
                                kernreq, ectx);
            } else {
                return make_builtin_dtype_elwise_property_setter_kernel(
                                out, offset_out,
                                ovdt.get_type_id(),
                                dst_metadata, m_property_index,
                                src_metadata,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dynd ndobject with dtype " << m_operand_type;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_writable) {
            if (!m_value_type.is_builtin()) {
                return m_value_type.extended()->make_elwise_property_getter_kernel(
                                out, offset_out,
                                dst_metadata,
                                src_metadata, m_property_index,
                                kernreq, ectx);
            } else {
                return make_builtin_dtype_elwise_property_getter_kernel(
                                out, offset_out,
                                m_value_type.get_type_id(),
                                dst_metadata,
                                src_metadata, m_property_index,
                                kernreq, ectx);
            }
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of dtype " << m_value_type;
            throw runtime_error(ss.str());
        }
    }
}

ndt::type property_dtype::with_replaced_storage_type(const ndt::type& replacement_type) const
{
    if (m_operand_type.get_kind() == expression_kind) {
        return ndt::type(new property_dtype(
                        static_cast<const base_expression_type *>(m_operand_type.extended())->with_replaced_storage_type(replacement_type),
                        m_property_name), false);
    } else {
        if (m_operand_type != replacement_type.value_type()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the property's storage dtype, " << m_operand_type;
            ss << ", does not match the replacement's value dtype, " << replacement_type.value_type();
            throw std::runtime_error(ss.str());
        }
        if (!m_reversed_property) {
            return ndt::type(new property_dtype(replacement_type, m_property_name), false);
        } else {
            return ndt::type(new property_dtype(m_value_type, replacement_type, m_property_name), false);
        }
    }
}

void property_dtype::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_properties(out_properties, out_count);
    } else {
        get_builtin_dtype_dynamic_array_properties(udt.get_type_id(), out_properties, out_count);
    }
}

void property_dtype::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    const ndt::type& udt = m_value_type.get_udtype();
    if (!udt.is_builtin()) {
        udt.extended()->get_dynamic_array_functions(out_functions, out_count);
    } else {
        //get_builtin_dtype_dynamic_ndobject_functions(udt.get_type_id(), out_functions, out_count);
    }
}
