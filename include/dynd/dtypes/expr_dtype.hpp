//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt


#ifndef _DYND__EXPR_DTYPE_HPP_
#define _DYND__EXPR_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>

namespace dynd {

/**
 * The expr dtype represents an expression on
 * an arbitrary number of operands.
 */
class expr_dtype : public base_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;

public:
    expr_dtype(const dtype& value_dtype, const dtype& operand_dtype);

    virtual ~expr_dtype();

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;

    size_t make_operand_to_value_assignment_kernel(
                    assignment_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    assignment_kernel *out, size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
};

/**
 * Makes a groupby dtype.
 */
inline dtype make_expr_dtype(const dtype& value_dtype,
                const dtype& operand_dtype)
{
    return dtype(new expr_dtype(value_dtype, operand_dtype), false);
}

} // namespace dynd

#endif // _DYND__EXPR_DTYPE_HPP_
