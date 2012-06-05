//
// Copyright (C) 2012 Continuum Analytics
//
//
// The array dtype always represents linearly-strided arrays of
// dtyped elements.
//
#ifndef _DND__ARRAY_DTYPE_HPP_
#define _DND__ARRAY_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/shortvector.hpp>

namespace dnd {

class array_dtype : public extended_dtype {
    // The number of bytes required for this dtype
    uintptr_t m_itemsize;
    // The data type of the array elements
    dtype m_element_dtype;
    // The shape and strides of the array
    multi_shortvector<intptr_t, 2> m_shape_and_strides;
    // The number of dimensions in the array
    int m_ndim;

public:
    // The general constructor with arbitrarily linear-strided data. The caller should ensure correctness,
    // i.e. that the strides are all divisible by the element_dtype's alignment.
    array_dtype(intptr_t dtype_size, int ndim, intptr_t *shape, intptr_t *strides, const dtype& element_dtype);
    // A default C-order constructor
    array_dtype(int ndim, intptr_t *shape, const dtype& element_dtype);

    int type_id() const {
        return array_type_id;
    }
    unsigned char kind() const {
        return composite_kind;
    }
    unsigned char alignment() const {
        return m_element_dtype.alignment();
    }
    uintptr_t itemsize() const {
        return m_itemsize;
    }

    const dtype& value_dtype(const dtype& self) const {
        return self;
    }
    void print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const;

    void print(std::ostream& o) const;

    bool is_object_type() const {
        return m_element_dtype.is_object_type();
    }

    const intptr_t *get_shape() const {
        return m_shape_and_strides.get(0);
    }

    const intptr_t *get_strides() const {
        return m_shape_and_strides.get(1);
    }

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    bool operator==(const extended_dtype& rhs) const;
};

template<class T>
dtype make_array_dtype(int ndim, intptr_t *shape) {
    return dtype(make_shared<array_dtype>(ndim, shape, make_dtype<T>()));
}

} // namespace dnd

#endif // _DND__ARRAY_DTYPE_HPP_
