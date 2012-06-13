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
#ifndef _DND__BYTESWAP_DTYPE_HPP_
#define _DND__BYTESWAP_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/align_dtype.hpp>

namespace dnd {

class byteswap_dtype : public extended_dtype {
    dtype m_value_dtype, m_operand_dtype;
public:
    byteswap_dtype(const dtype& value_dtype)
        : m_value_dtype(value_dtype), m_operand_dtype(make_bytes_dtype(value_dtype.itemsize(), value_dtype.alignment()))
    {
        if (value_dtype.extended() != 0) {
            throw std::runtime_error("byteswap_dtype: Only built-in dtypes are supported presently");
        }

    }

    byteswap_dtype(const dtype& value_dtype, const dtype& operand_dtype)
        : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype)
    {
        // Only a bytes dtype be the operand to the byteswap
        if (operand_dtype.value_dtype().type_id() != bytes_type_id) {
            std::stringstream ss;
            ss << "byteswap_dtype: The operand to the dtype must have a value dtype of bytes, not " << operand_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
        // Automatically realign if needed
        if (operand_dtype.value_dtype().alignment() < value_dtype.alignment()) {
            m_operand_dtype = make_align_dtype(value_dtype.alignment(), operand_dtype);
        }
    }

    type_id_t type_id() const {
        return byteswap_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_value_dtype.alignment();
    }
    uintptr_t itemsize() const {
        return m_value_dtype.itemsize();
    }

    const dtype& value_dtype(const dtype& self) const {
        return m_value_dtype;
    }
    const dtype& operand_dtype(const dtype& self) const {
        return m_operand_dtype;
    }
    void print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const;

    void print(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    bool is_object_type() const {
        return m_operand_dtype.is_object_type();
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    void get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                        kernel_instance<unary_operation_t>& out_kernel) const;
    void get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                        kernel_instance<unary_operation_t>& out_kernel) const;
    dtype with_replaced_storage_dtype(const dtype& replacement_dtype) const;
};

/**
 * Makes a byteswapped dtype to view the given dtype with a swapped byte order.
 */
inline dtype make_byteswap_dtype(const dtype& native_dtype) {
    return dtype(make_shared<byteswap_dtype>(native_dtype));
}

inline dtype make_byteswap_dtype(const dtype& native_dtype, const dtype& operand_dtype) {
    return dtype(make_shared<byteswap_dtype>(native_dtype, operand_dtype));
}

template<typename Tnative>
dtype make_byteswap_dtype() {
    return dtype(make_shared<byteswap_dtype>(make_dtype<Tnative>()));
}

} // namespace dnd

#endif // _DND__BYTESWAP_DTYPE_HPP_
