//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This is being initially written as a "date" property type, but
// with an appropriate extended_dtype interface which generates the
// to_value_kernel, this can be switched to just a property type using
// that interface.

#ifndef _DYND__DATE_PROPERTY_DTYPE_HPP_
#define _DYND__DATE_PROPERTY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

class date_property_dtype : public extended_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    std::string m_property_name;
    kernel_instance<unary_operation_pair_t> m_to_value_kernel;
public:
    date_property_dtype(const dtype& operand_dtype, const std::string& property_name);

    virtual ~date_property_dtype();

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }
    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return m_operand_dtype.get_memory_management();
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(int i, intptr_t *out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    void get_operand_to_value_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const;
    void get_value_to_operand_kernel(const eval::eval_context *ectx,
                            kernel_instance<unary_operation_pair_t>& out_borrowed_kernel) const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;


};

/**
 * Makes a conversion dtype to convert from the operand_dtype to the value_dtype.
 * If the value_dtype has expression_kind, it chains operand_dtype.value_dtype()
 * into value_dtype.storage_dtype().
 */
inline dtype make_date_property_dtype(const dtype& operand_dtype, const std::string& property_name) {
    return dtype(new date_property_dtype(operand_dtype, property_name));
}

} // namespace dynd

#endif // _DYND__DATE_PROPERTY_DTYPE_HPP_
