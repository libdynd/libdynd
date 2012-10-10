//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The ndarray dtype always represents linearly-strided arrays of
// dtyped elements.
//
#ifndef _DND__NDARRAY_DTYPE_HPP_
#define _DND__NDARRAY_DTYPE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/shortvector.hpp>

namespace dnd {

class ndarray_dtype : public extended_dtype {
    // The number of bytes required for this dtype
    uintptr_t m_element_size;
    // The data type of the array elements
    dtype m_element_dtype;
    // The shape and strides of the array
    multi_shortvector<intptr_t, 2> m_shape_and_strides;
    // The number of dimensions in the array
    int m_ndim;

public:
    // The general constructor with arbitrarily linear-strided data. The caller should ensure correctness,
    // i.e. that the strides are all divisible by the element_dtype's alignment.
    ndarray_dtype(intptr_t dtype_size, int ndim, intptr_t *shape, intptr_t *strides, const dtype& element_dtype);
    // A default C-order constructor
    ndarray_dtype(int ndim, intptr_t *shape, const dtype& element_dtype);

    type_id_t type_id() const {
        return (type_id_t)ndarray_type_id;
    }
    dtype_kind_t kind() const {
        return composite_kind;
    }
    size_t alignment() const {
        return m_element_dtype.alignment();
    }
    uintptr_t element_size() const {
        return m_element_size;
    }

    void print_element(std::ostream& o, const char *data) const;

    void print_dtype(std::ostream& o) const;

    dtype_memory_management_t get_memory_management() const {
        return m_element_dtype.get_memory_management();
    }

    dtype apply_linear_index(int ndim, const irange *indices, int dtype_ndim) const;

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
dtype make_ndarray_dtype(int ndim, intptr_t *shape) {
    return dtype(new ndarray_dtype(ndim, shape, make_dtype<T>()));
}

} // namespace dnd

#endif // _DND__NDARRAY_DTYPE_HPP_
