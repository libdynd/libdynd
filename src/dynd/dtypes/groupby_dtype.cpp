//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <vector>

#include <dynd/dtypes/groupby_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/array_dtype.hpp>
#include <dynd/dtypes/categorical_dtype.hpp>

using namespace std;
using namespace dynd;

groupby_dtype::groupby_dtype(const dtype& data_values_dtype,
                const dtype& by_values_dtype, const dtype& groups_dtype)
    : base_expression_dtype(groupby_type_id, expression_kind, sizeof(groupby_dtype_data), sizeof(void *))
{
    if (groups_dtype.get_type_id() != categorical_type_id) {
        throw runtime_error("to construct a groupby dtype, its groups dtype must be categorical");
    }
    if (data_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    if (by_values_dtype.get_undim() < 1) {
        throw runtime_error("to construct a groupby dtype, its values dtype must have at least one uniform dimension");
    }
    if (by_values_dtype.at_single(0).value_dtype() !=
                    reinterpret_cast<const categorical_dtype *>(groups_dtype.extended())->get_category_dtype()) {
        stringstream ss;
        ss << "to construct a groupby dtype, the by dtype, " << by_values_dtype.at_single(0);
        ss << ", should match the category dtype, " << reinterpret_cast<const categorical_dtype *>(groups_dtype.extended())->get_category_dtype();
    }
    m_operand_dtype = make_fixedstruct_dtype(make_pointer_dtype(data_values_dtype), "data",
                    make_pointer_dtype(by_values_dtype), "by");
    m_value_dtype = make_strided_array_dtype(make_array_dtype(data_values_dtype.at_single(0)));
    m_groups_dtype = groups_dtype;
}

groupby_dtype::~groupby_dtype()
{
}

void groupby_dtype::print_data(std::ostream& DYND_UNUSED(o),
                const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    throw runtime_error("internal error: groupby_dtype::print_data isn't supposed to be called");
}

void groupby_dtype::print_dtype(std::ostream& o) const
{
    o << "groupby<values=" << m_operand_dtype.at_single(0) << ", by=" << m_operand_dtype.at_single(1);
    o << ", groups=" << m_groups_dtype << ">";
}

dtype groupby_dtype::apply_linear_index(int nindices, const irange *DYND_UNUSED(indices),
                int DYND_UNUSED(current_i), const dtype& DYND_UNUSED(root_dt)) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else {
        throw runtime_error("TODO groupby_dtype::apply_linear_index");
    }
}

void groupby_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_value_dtype.is_builtin()) {
        m_value_dtype.extended()->get_shape(i, out_shape);
    }
}

bool groupby_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    // Treat this dtype as the value dtype for whether assignment is always lossless
    if (src_dt.extended() == this) {
        return ::dynd::is_lossless_assignment(dst_dt, m_value_dtype);
    } else {
        return ::dynd::is_lossless_assignment(m_value_dtype, src_dt);
    }
}

bool groupby_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != view_type_id) {
        return false;
    } else {
        const groupby_dtype *dt = static_cast<const groupby_dtype*>(&rhs);
        return m_value_dtype == dt->m_value_dtype;
    }
}

void groupby_dtype::get_operand_to_value_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_borrowed_kernel)) const
{
    throw runtime_error("TODO: implement groupby_dtype::get_operand_to_value_kernel");
}

void groupby_dtype::get_value_to_operand_kernel(const eval::eval_context *DYND_UNUSED(ectx),
                        kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_borrowed_kernel)) const
{
    throw runtime_error("TODO: implement groupby_dtype::get_value_to_operand_kernel");
}

dtype groupby_dtype::with_replaced_storage_dtype(const dtype& DYND_UNUSED(replacement_dtype)) const
{
    throw runtime_error("TODO: implement groupby_dtype::with_replaced_storage_dtype");
}
