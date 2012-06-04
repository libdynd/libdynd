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
    dtype m_value_dtype, m_storage_dtype;
    assign_error_mode m_errmode;
public:
    conversion_dtype(const dtype& value_dtype, const dtype& storage_dtype, assign_error_mode errmode)
        : m_value_dtype(value_dtype.value_dtype()), m_storage_dtype(storage_dtype), m_errmode(errmode)
    {}

    int type_id() const {
        return conversion_type_id;
    }
    unsigned char kind() const {
        return expression_kind;
    }
    // Expose the storage traits here
    unsigned char alignment() const {
        return m_storage_dtype.alignment();
    }
    uintptr_t itemsize() const {
        return m_storage_dtype.itemsize();
    }

    const dtype& value_dtype(const dtype& self) const {
        return m_value_dtype;
    }
    const dtype& storage_dtype(const dtype& self) const {
        return m_storage_dtype;
    }
    void print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const;

    void print(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    bool is_object_type() const {
        return m_storage_dtype.is_object_type();
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;
};

inline dtype make_conversion_dtype(const dtype& value_dtype, const dtype& storage_dtype, assign_error_mode errmode = default_error_mode) {
    return dtype(make_shared<conversion_dtype>(value_dtype, storage_dtype, errmode));
}

template<typename Tvalue, typename Tstorage>
dtype make_conversion_dtype(assign_error_mode errmode = default_error_mode) {
    return dtype(make_shared<conversion_dtype>(make_dtype<Tvalue>(), make_dtype<Tstorage>(), errmode));
}

} // namespace dnd

#endif // _DND__CONVERSION_DTYPE_HPP_
