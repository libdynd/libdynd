//
// Copyright (C) 2012 Continuum Analytics
//
//
// The conversion dtype represents one dtype viewed
// as another buffering based on the casting mechanism.
//
// This dtype takes on the characteristics of its storage dtype
// through the dtype interface, except for the "kind" which
// is expression_kind to signal that the value_dtype must be examined.
//
#ifndef _DND__CONVERSION_DTYPE_HPP_
#define _DND__CONVERSION_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>

namespace dnd {

class conversion_dtype : public extended_dtype {
    dtype m_value_dtype, m_operand_dtype;
    assign_error_mode m_errmode;
    bool m_no_errors_to_value, m_no_errors_to_storage;
public:
    conversion_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode)
        : m_value_dtype(value_dtype), m_operand_dtype(operand_dtype), m_errmode(errmode),
          m_no_errors_to_value(errmode == assign_error_none || ::dnd::is_lossless_assignment(m_value_dtype, m_operand_dtype)),
          m_no_errors_to_storage(errmode == assign_error_none || ::dnd::is_lossless_assignment(m_operand_dtype, m_value_dtype))
    {
        // An alternative to this error would be to use value_dtype.value_dtype(), cutting
        // away the expression part of the given value_dtype.
        if (m_value_dtype.kind() == expression_kind) {
            throw std::runtime_error("conversion_dtype: The value dtype cannot be an expression_kind");
        }

    }

    type_id_t type_id() const {
        return conversion_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_operand_dtype.alignment();
    }
    uintptr_t itemsize() const {
        return m_operand_dtype.itemsize();
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
    void get_operand_to_value_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const;
    void get_value_to_operand_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride, kernel_instance<unary_operation_t>& out_kernel) const;
};

/**
 * Makes a conversion dtype to convert from the operand_dtype to the value_dtype.
 * This always creates the conversion, if the caller wants to avoid redundant
 * conversions, they should check that (value_dtype != operand_dtype.value_dtype())
 * before calling this function.
 */
inline dtype make_conversion_dtype(const dtype& value_dtype, const dtype& operand_dtype, assign_error_mode errmode = default_error_mode) {
    return dtype(make_shared<conversion_dtype>(value_dtype, operand_dtype, errmode));
}

template<typename Tvalue, typename Tstorage>
dtype make_conversion_dtype(assign_error_mode errmode = default_error_mode) {
    return dtype(make_shared<conversion_dtype>(make_dtype<Tvalue>(), make_dtype<Tstorage>(), errmode));
}

} // namespace dnd

#endif // _DND__CONVERSION_DTYPE_HPP_
