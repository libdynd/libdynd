//
// Copyright (C) 2011-13, DyND Developers
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
    kernel_instance<unary_operation_pair_t> m_copy_kernel;

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

    // Only support views of POD data for now (TODO: support blockref)
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    void get_shape(size_t i, intptr_t *out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const base_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    void get_operand_to_value_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const;
    void get_value_to_operand_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;

    size_t make_operand_to_value_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
    size_t make_value_to_operand_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t offset_out,
                    const char *dst_metadata, const char *src_metadata,
                    const eval::eval_context *ectx) const;
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
