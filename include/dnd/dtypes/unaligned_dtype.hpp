//
// Copyright (C) 2012 Continuum Analytics
//
//
// The unaligned dtype represents a POD dtype
// which has alignment > 1 in a form which doesn't
// require alignment.
//
#ifndef _DND__UNALIGNED_DTYPE_HPP_
#define _DND__UNALIGNED_DTYPE_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

class unaligned_dtype : public extended_dtype {
    dtype m_value_dtype;
public:
    unaligned_dtype(const dtype& value_dtype)
        : m_value_dtype(value_dtype)
    {
        if (value_dtype.is_object_type()) {
            throw std::runtime_error("unaligned_dtype: Only POD dtypes are supported, object dtypes must be aligned");
        }
    }

    type_id_t type_id() const {
        return unaligned_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return 1;
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

    // Don't support unaligned versions of object-semantic data
    bool is_object_type() const {
        return false;
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
 * Makes an unaligned dtype to view the given dtype without alignment requirements.
 */
inline dtype make_unaligned_dtype(const dtype& value_dtype) {
    if (value_dtype.alignment() > 1) {
        return dtype(make_shared<unaligned_dtype>(value_dtype));
    } else {
        return value_dtype;
    }
}

template<typename Tnative>
dtype make_unaligned_dtype() {
    return dtype(make_shared<unaligned_dtype>(make_dtype<Tnative>()));
}

} // namespace dnd

#endif // _DND__UNALIGNED_DTYPE_HPP_
