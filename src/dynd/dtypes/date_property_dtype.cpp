//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/date_property_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;


dynd::date_property_dtype::date_property_dtype(const dtype& operand_dtype, const std::string& property_name)
    : base_expression_dtype(date_property_type_id, expression_kind, operand_dtype.get_data_size(), operand_dtype.get_alignment()),
            m_value_dtype(), m_operand_dtype(operand_dtype), m_property_name(property_name)
{
    if (operand_dtype.value_dtype().get_type_id() != date_type_id) {
        std::stringstream ss;
        ss << "date_property_dtype: The operand dtype " << operand_dtype << " should be a date dtype";
        throw std::runtime_error(ss.str());
    }

    const date_dtype *dd = static_cast<const date_dtype *>(m_operand_dtype.value_dtype().extended());
    dd->get_property_getter_kernel(property_name, m_value_dtype, m_to_value_kernel);
}

date_property_dtype::~date_property_dtype()
{
}

void dynd::date_property_dtype::print_data(std::ostream& DYND_UNUSED(o), const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: date_property_dtype::print_data isn't supposed to be called");
}

void dynd::date_property_dtype::print_dtype(std::ostream& o) const
{
    o << "property<name=" << m_property_name << ", type=" << m_operand_dtype << ">";
}

dtype dynd::date_property_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        return m_value_dtype.apply_linear_index(nindices, indices, current_i, root_dt);
    }
}

void dynd::date_property_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
        m_value_dtype.extended()->get_shape(i, out_shape);
    }
}

bool dynd::date_property_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return false;
    }
}

bool dynd::date_property_dtype::operator==(const base_dtype& rhs) const
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

void dynd::date_property_dtype::get_operand_to_value_kernel(const eval::eval_context *ectx,
                        kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const
{
    if (m_to_value_kernel.kernel.single != NULL) {
        out_borrowed_kernel.borrow_from(m_to_value_kernel);
    } else if (ectx != NULL) {
        // If the kernel wasn't set, errmode is assign_error_default, so we must use the eval_context
        ::dynd::get_dtype_assignment_kernel(m_value_dtype, m_operand_dtype.value_dtype(),
                            ectx->default_assign_error_mode, ectx, out_borrowed_kernel);
    } else {
        // An evaluation context is needed to get the kernel, set the output to NULL
        out_borrowed_kernel.kernel = unary_operation_pair_t();
    }
}

void dynd::date_property_dtype::get_value_to_operand_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_borrowed_kernel)) const
{
    stringstream ss;
    ss << "cannot write to property " << dtype(this, true);
    throw runtime_error(ss.str());
}

dtype dynd::date_property_dtype::with_replaced_storage_dtype(const dtype& replacement_dtype) const
{
    if (m_operand_dtype.get_kind() == expression_kind) {
        return dtype(new date_property_dtype(
                        static_cast<const base_expression_dtype *>(m_operand_dtype.extended())->with_replaced_storage_dtype(replacement_dtype),
                        m_property_name));
    } else {
        if (m_operand_dtype != replacement_dtype.value_dtype()) {
            std::stringstream ss;
            ss << "Cannot chain dtypes, because the property's storage dtype, " << m_operand_dtype;
            ss << ", does not match the replacement's value dtype, " << replacement_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        return dtype(new date_property_dtype(replacement_dtype, m_property_name));
    }
}
