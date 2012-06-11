//
// Copyright (C) 2012 Continuum Analytics
//
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

namespace dnd {

class byteswap_dtype : public extended_dtype {
    dtype m_value_dtype;
public:
    byteswap_dtype(const dtype& value_dtype)
        : m_value_dtype(value_dtype)
    {
        if (value_dtype.extended() != 0) {
            throw std::runtime_error("byteswap_dtype: Only built-in dtypes are supported presently");
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
        return self;
    }
    void print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const;

    void print(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    bool is_object_type() const {
        return m_value_dtype.is_object_type();
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;

    // For expression_kind dtypes - converts to/from the storage's value dtype
    void get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                        kernel_instance<unary_operation_t>& out_kernel) const;
    void get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                        kernel_instance<unary_operation_t>& out_kernel) const;
};

/**
 * Makes a byteswapped dtype to view the given dtype with a swapped byte order.
 * If the given dtype is a byteswap_dtype, it gets stripped instead of
 * chaining multiple byteswap operations.
 */
inline dtype make_byteswap_dtype(const dtype& native_dtype) {
    if (native_dtype.type_id() != byteswap_type_id) {
        return dtype(make_shared<byteswap_dtype>(native_dtype));
    } else {
        return native_dtype.extended()->operand_dtype(native_dtype);
    }
}

template<typename Tnative>
dtype make_byteswap_dtype() {
    return dtype(make_shared<byteswap_dtype>(make_dtype<Tnative>()));
}

} // namespace dnd

#endif // _DND__BYTESWAP_DTYPE_HPP_
