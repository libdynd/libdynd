//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The realign dtype applies a more
// stringent alignment to a bytes dtype.
//
#ifndef _DND__REALIGN_DTYPE_HPP_
#define _DND__REALIGN_DTYPE_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

class align_dtype : public extended_dtype {
    dtype m_value_dtype, m_operand_dtype;
public:
    align_dtype(intptr_t alignment, const dtype& operand_dtype)
        : m_value_dtype(make_bytes_dtype(operand_dtype.value_dtype().itemsize(), alignment)),
          m_operand_dtype(operand_dtype)
    {
        if (operand_dtype.value_dtype().type_id() != bytes_type_id) {
            std::stringstream ss;
            ss << "align_dtype: Can only apply alignment operation to a bytes dtype, not to " << operand_dtype.value_dtype();
            throw std::runtime_error(ss.str());
        }
    }

    type_id_t type_id() const {
        return align_type_id;
    }
    dtype_kind_t kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_operand_dtype.alignment();
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
 * Makes an unaligned dtype to view the given dtype without alignment requirements.
 */
inline dtype make_align_dtype(intptr_t alignment, const dtype& value_dtype)
{
    if (alignment > value_dtype.alignment()) {
        return dtype(make_shared<align_dtype>(alignment, value_dtype));
    } else {
        return value_dtype;
    }
}

/**
 * Reduces a dtype's alignment requirements to 1.
 */
dtype make_unaligned_dtype(const dtype& value_dtype);

/**
 * Reduces a dtype's alignment requirements to 1.
 */
template<typename T>
dtype make_unaligned_dtype()
{
    return make_unaligned_dtype(make_dtype<T>());
}

} // namespace dnd

#endif // _DND__REALIGN_DTYPE_HPP_
