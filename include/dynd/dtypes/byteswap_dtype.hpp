//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The byteswap dtype represents one of the
// built-in dtypes stored in non-native byte order.
//
// TODO: When needed, a mechanism for non built-in
//       dtypes to expose a byteswap interface should
//       be added, which this dtype would use to
//       do the actual swapping.
//
#ifndef _DYND__BYTESWAP_DTYPE_HPP_
#define _DYND__BYTESWAP_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>

namespace dynd {

class byteswap_dtype : public extended_expression_dtype {
    dtype m_value_dtype, m_operand_dtype;
    unary_specialization_kernel_instance m_byteswap_kernel;

public:
    byteswap_dtype(const dtype& value_dtype);
    byteswap_dtype(const dtype& value_dtype, const dtype& operand_dtype);

    type_id_t get_type_id() const {
        return byteswap_type_id;
    }
    dtype_kind_t get_kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return m_value_dtype.get_alignment();
    }
    size_t get_element_size() const {
        return m_value_dtype.get_element_size();
    }

    const dtype& get_value_dtype() const {
        return m_value_dtype;
    }
    const dtype& get_operand_dtype() const {
        return m_operand_dtype;
    }
    void print_element(std::ostream& o, const char *data, const char *metadata) const;

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

    // Converts to/from the storage's value dtype
    void get_operand_to_value_kernel(const eval::eval_context *ectx,
                            unary_specialization_kernel_instance& out_borrowed_kernel) const;
    void get_value_to_operand_kernel(const eval::eval_context *ectx,
                            unary_specialization_kernel_instance& out_borrowed_kernel) const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;
};

/**
 * Makes a byteswapped dtype to view the given dtype with a swapped byte order.
 */
inline dtype make_byteswap_dtype(const dtype& native_dtype) {
    return dtype(new byteswap_dtype(native_dtype));
}

inline dtype make_byteswap_dtype(const dtype& native_dtype, const dtype& operand_dtype) {
    return dtype(new byteswap_dtype(native_dtype, operand_dtype));
}

template<typename Tnative>
dtype make_byteswap_dtype() {
    return dtype(new byteswap_dtype(make_dtype<Tnative>()));
}

} // namespace dynd

#endif // _DYND__BYTESWAP_DTYPE_HPP_
