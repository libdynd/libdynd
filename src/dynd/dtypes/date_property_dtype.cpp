//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/date_property_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;


date_property_dtype::date_property_dtype(const dtype& operand_dtype, const std::string& property_name)
    : base_expression_dtype(date_property_type_id, expression_kind,
                    operand_dtype.get_data_size(), operand_dtype.get_alignment(), dtype_flag_scalar,
                    operand_dtype.get_metadata_size()),
            m_value_dtype(), m_operand_dtype(operand_dtype), m_property_name(property_name)
{
    if (operand_dtype.value_dtype().get_type_id() != date_type_id) {
        std::stringstream ss;
        ss << "date_property_dtype: The operand dtype " << operand_dtype << " should be a date dtype";
        throw std::runtime_error(ss.str());
    }

    const date_dtype *dd = static_cast<const date_dtype *>(m_operand_dtype.value_dtype().extended());
    m_property_index = dd->get_property_index(property_name);
    m_value_dtype = dd->get_property_dtype(m_property_index);
}

date_property_dtype::~date_property_dtype()
{
}

void date_property_dtype::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: date_property_dtype::print_data isn't supposed to be called");
}

void date_property_dtype::print_dtype(std::ostream& o) const
{
    o << "property<name=" << m_property_name << ", type=" << m_operand_dtype << ">";
}

void date_property_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
        m_value_dtype.extended()->get_shape(i, out_shape);
    }
}

bool date_property_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return false;
    }
}

bool date_property_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != date_property_type_id) {
        return false;
    } else {
        const date_property_dtype *dt = static_cast<const date_property_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype &&
            m_operand_dtype == dt->m_operand_dtype &&
            m_property_name == dt->m_property_name;
    }
}

size_t date_property_dtype::make_operand_to_value_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const char *dst_metadata, const char *src_metadata,
                const eval::eval_context *ectx) const
{
    const date_dtype *dd = static_cast<const date_dtype *>(m_operand_dtype.value_dtype().extended());
    return dd->make_property_getter_kernel(out, offset_out,
                    dst_metadata,
                    src_metadata, m_property_index,
                    ectx);
}

size_t date_property_dtype::make_value_to_operand_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *DYND_UNUSED(out),
                size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "cannot write to property " << dtype(this, true);
    throw runtime_error(ss.str());
}

dtype date_property_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() == expression_kind) {
        return dtype(new date_property_dtype(
                        static_cast<const base_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype),
                        m_property_name), false);
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the property's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(new date_property_dtype(replacement_dtype, m_property_name), false);
    }
}
