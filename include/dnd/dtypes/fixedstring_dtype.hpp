//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The fixedstring dtype represents a string with
// a particular encoding, stored in a fixed-size
// buffer.
//
#ifndef _DND__FIXEDSTRING_DTYPE_HPP_
#define _DND__FIXEDSTRING_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/view_dtype.hpp>
#include <dnd/string_encodings.hpp>

namespace dnd {

class fixedstring_dtype : public extended_string_dtype {
    intptr_t m_element_size, m_alignment, m_stringsize;
    string_encoding_t m_encoding;

public:
    fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize);

    type_id_t type_id() const {
        return fixedstring_type_id;
    }
    dtype_kind_t kind() const {
        return string_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_alignment;
    }
    uintptr_t element_size() const {
        return m_element_size;
    }

    string_encoding_t encoding() const {
        return m_encoding;
    }

    const dtype& value_dtype(const dtype& self) const {
        return self;
    }
    const dtype& operand_dtype(const dtype& self) const {
        return self;
    }

    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    bool is_object_type() const {
        return false;
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;
};

inline dtype make_fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize) {
    return dtype(make_shared<fixedstring_dtype>(encoding, stringsize));
}

} // namespace dnd

#endif // _DND__FIXEDSTRING_DTYPE_HPP_
