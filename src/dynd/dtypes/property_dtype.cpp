//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;


property_dtype::property_dtype(const dtype& operand_dtype, const std::string& property_name)
    : base_expression_dtype(date_property_type_id, expression_kind,
                    operand_dtype.get_data_size(), operand_dtype.get_alignment(), dtype_flag_scalar,
                    operand_dtype.get_metadata_size()),
            m_value_dtype(), m_operand_dtype(operand_dtype),
            m_readable(false), m_writable(false),
            m_reversed_property(false),
            m_property_name(property_name),
            m_property_index(0)
{
    if (operand_dtype.value_dtype().is_builtin()) {
        std::stringstream ss;
        ss << "the dtype " << operand_dtype;
        ss << " doesn't have a property \"" << property_name << "\"";
        throw std::runtime_error(ss.str());
    }

    m_property_index = m_operand_dtype.value_dtype().extended()->get_elwise_property_index(
                    property_name, m_readable, m_writable);
    m_value_dtype = m_operand_dtype.value_dtype().extended()->get_elwise_property_dtype(m_property_index);
}

property_dtype::property_dtype(const dtype& value_dtype, const dtype& operand_dtype,
                const std::string& property_name)
    : base_expression_dtype(date_property_type_id, expression_kind,
                    operand_dtype.get_data_size(), operand_dtype.get_alignment(), dtype_flag_scalar,
                    operand_dtype.get_metadata_size()),
            m_value_dtype(value_dtype), m_operand_dtype(operand_dtype),
            m_readable(false), m_writable(false),
            m_reversed_property(true),
            m_property_name(property_name),
            m_property_index(0)
{
    if (value_dtype.is_builtin()) {
        stringstream ss;
        ss << "the dtype " << operand_dtype;
        ss << " doesn't have a property \"" << property_name << "\"";
        throw std::runtime_error(ss.str());
    }
    if (m_value_dtype.get_kind() == expression_kind) {
        stringstream ss;
        ss << "property_dtype: The destination dtype " << m_value_dtype;
        ss << " should not be an expression_kind";
        throw std::runtime_error(ss.str());
    }

    m_property_index = m_value_dtype.extended()->get_elwise_property_index(
                    property_name, m_writable, m_readable);
    // If the operand dtype doesn't match the property, add a
    // convertion to the correct dtype
    dtype property_dtype = m_value_dtype.extended()->get_elwise_property_dtype(
                    m_property_index);
    if (m_operand_dtype.value_dtype() != property_dtype) {
        m_operand_dtype = make_convert_dtype(property_dtype, m_operand_dtype);
    }
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
        o << "property<name=" << m_property_name << ", operand=" << m_operand_dtype << ">";
    } else {
        o << "property<reversed, name=" << m_property_name;
        o << ", value=" << m_value_dtype;
        o << ", operand=" << m_operand_dtype << ">";
    }
}

void property_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
        m_value_dtype.extended()->get_shape(i, out_shape);
    }
}

bool property_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return false;
    }
}

bool property_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != date_property_type_id) {
        return false;
    } else {
        const property_dtype *dt = static_cast<const property_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype &&
            m_operand_dtype == dt->m_operand_dtype &&
            m_property_name == dt->m_property_name;
    }
}

size_t property_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_readable) {
            return m_operand_dtype.value_dtype().extended()->make_elwise_property_getter_kernel(
                            out, offset_out,
                            dst_metadata,
                            src_metadata, m_property_index,
                            ectx);
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of dtype " << m_operand_dtype;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_readable) {
            return m_value_dtype.extended()->make_elwise_property_setter_kernel(
                            out, offset_out,
                            dst_metadata, m_property_index,
                            src_metadata,
                            ectx);
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dtype " << m_value_dtype;
            throw runtime_error(ss.str());
        }
    }
}

size_t property_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                const eval::eval_context *ectx) const
{
    if (!m_reversed_property) {
        if (m_readable) {
            return m_operand_dtype.value_dtype().extended()->make_elwise_property_setter_kernel(
                            out, offset_out,
                            dst_metadata, m_property_index,
                            src_metadata,
                            ectx);
        } else {
            stringstream ss;
            ss << "cannot write to property \"" << m_property_name << "\"";
            ss << " of dtype " << m_operand_dtype;
            throw runtime_error(ss.str());
        }
    } else {
        if (m_readable) {
            return m_value_dtype.extended()->make_elwise_property_getter_kernel(
                            out, offset_out,
                            dst_metadata,
                            src_metadata, m_property_index,
                            ectx);
        } else {
            stringstream ss;
            ss << "cannot read from property \"" << m_property_name << "\"";
            ss << " of dtype " << m_value_dtype;
            throw runtime_error(ss.str());
        }
    }
}

dtype property_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() == expression_kind) {
        return dtype(new property_dtype(
                        static_cast<const base_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype),
                        m_property_name), false);
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the property's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        if (!m_reversed_property) {
            return dtype(new property_dtype(replacement_dtype, m_property_name), false);
        } else {
            return dtype(new property_dtype(m_value_dtype, replacement_dtype, m_property_name), false);
        }
    }
}
