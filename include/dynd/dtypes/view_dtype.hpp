//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The view dtype reinterprets the bytes of
// one dtype as another.
//
#ifndef _DYND__VIEW_DTYPE_HPP_
#define _DYND__VIEW_DTYPE_HPP_

#include <dynd/dtype.hpp>

namespace dynd {

class view_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;

public:
    view_dtype(const dtype& value_dtype, const dtype& operand_dtype);

    virtual ~view_dtype();

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;
    void get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;

    // Propagate properties and functions from the value dtype
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const
    {
        if (!m_value_dtype.is_builtin()) {
            m_value_dtype.extended()->get_dynamic_array_properties(out_properties, out_count);
        }
    }
    void get_dynamic_array_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const
    {
        if (!m_value_dtype.is_builtin()) {
            m_value_dtype.extended()->get_dynamic_array_functions(out_functions, out_count);
        }
    }
};

/**
 * Makes an unaligned dtype to view the given dtype without alignment requirements.
 */
inline dtype make_view_dtype(const dtype& value_dtype, const dtype& operand_dtype) {
    if (value_dtype.get_kind() != expression_kind) {
        return dtype(new view_dtype(value_dtype, operand_dtype), false);
    } else {
        // When the value dtype has an expression_kind, we need to chain things together
        // so that the view operation happens just at the primitive level.
        return static_cast<const base_expression_dtype *>(value_dtype.extended())->with_replaced_storage_dtype(
            dtype(new view_dtype(value_dtype.storage_dtype(), operand_dtype), false));
    }
}

template<typename Tvalue, typename Toperand>
dtype make_view_dtype() {
    return dtype(new view_dtype(make_dtype<Tvalue>()), false);
}

} // namespace dynd

#endif // _DYND__VIEW_DTYPE_HPP_
