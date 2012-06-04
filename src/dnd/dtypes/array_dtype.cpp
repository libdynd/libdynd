//
// Copyright (C) 2012 Continuum Analytics
//

#include <dnd/dtypes/array_dtype.hpp>
#include <dnd/raw_iteration.hpp>

using namespace dnd;

array_dtype::array_dtype(intptr_t itemsize, int ndim,
                                    intptr_t *shape, intptr_t *strides,
                                    const dtype& element_dtype)
    : extended_dtype(), m_itemsize(itemsize), m_element_dtype(element_dtype),
      m_shape_and_strides(ndim), m_ndim(ndim)
{
    memcpy(m_shape_and_strides.get(0), shape, m_ndim * sizeof(intptr_t));
    memcpy(m_shape_and_strides.get(1), strides, m_ndim * sizeof(intptr_t));
}

intptr_t product(int n, intptr_t *values)
{
    intptr_t result = 1;
    for (int i = 0; i < n; ++i) {
        result *= values[i];
    }

    return result;
}

array_dtype::array_dtype(int ndim, intptr_t *shape, const dtype& element_dtype)
    : extended_dtype(), m_itemsize(element_dtype.itemsize() * product(ndim, shape)),
      m_element_dtype(element_dtype), m_shape_and_strides(ndim), m_ndim(ndim)
{
    intptr_t *this_shape = m_shape_and_strides.get(0), *this_strides = m_shape_and_strides.get(1);

    memcpy(this_shape, shape, m_ndim * sizeof(intptr_t));
    intptr_t stride_value = element_dtype.itemsize();
    for (int idim = ndim-1; idim >= 0; --idim) {
        this_strides[idim] = stride_value;
        stride_value *= this_shape[idim];
    }
}

static void nested_array_print(std::ostream& o, const dtype& d, const char *data, int ndim, const intptr_t *shape, const intptr_t *strides)
{
    o << "[";
    if (ndim == 1) {
        d.print_data(o, data, strides[0], shape[0], ", ");
    } else {
        intptr_t size = *shape;
        intptr_t stride = *strides;
        for (intptr_t k = 0; k < size; ++k) {
            nested_array_print(o, d, data, ndim - 1, shape + 1, strides + 1);
            if (k + 1 != size) {
                o << ", ";
            }
            data += stride;
        }
    }
    o << "]";
}

void array_dtype::print_data(std::ostream& o, const dtype& dt, const char *data, intptr_t stride, intptr_t size,
                        const char *separator) const
{
    for (int i = 0; i < size; ++i) {
        nested_array_print(o, m_element_dtype, data, m_ndim, get_shape(), get_strides());
        if (i != size-1) {
            o << separator;
            data += size;
        }
    }
}

void array_dtype::print(std::ostream& o) const
{
    o << "array<" << m_element_dtype << ", (";
    const intptr_t *shape = get_shape();
    for (int i = 0; i < m_ndim; ++i) {
        o << shape[i];
        if (i != m_ndim - 1) {
            o << ",";
        }
    }
    o << ")>";
}


bool array_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (src_dt.extended() == this) {
        if (dst_dt.extended() == this) {
            // Casting from identical types
            return true;
        } else if (dst_dt.type_id() == array_type_id) {
            // Casting array to array, check that it can broadcast, and that the
            // element dtype can cast losslessly
            const array_dtype *dst_adt = static_cast<const array_dtype *>(dst_dt.extended());
            return shape_can_broadcast(dst_adt->m_ndim, dst_adt->get_shape(), m_ndim, get_shape()) &&
                ::is_lossless_assignment(dst_adt->m_element_dtype, m_element_dtype);
        } else {
            return false;
        }

    } else {
        // If the src element can losslessly cast to the element, then
        // can broadcast it to everywhere.
        return ::is_lossless_assignment(m_element_dtype, src_dt);
    }
}

bool array_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.type_id() != array_type_id) {
        return false;
    } else {
        const array_dtype *adt = static_cast<const array_dtype*>(&rhs);

        if (m_ndim != adt->m_ndim || m_element_dtype != adt->m_element_dtype)
            return false;

        const intptr_t *this_shape = get_shape(), *this_strides = get_strides();
        const intptr_t *adt_shape = adt->get_shape(), *adt_strides = get_strides();
        for (int i = 0; i < m_ndim; ++i) {
            if (this_shape[i] != adt_shape[i] || this_strides[i] != adt_strides[i]) {
                return false;
            }
        }

        return true;
    }
}
