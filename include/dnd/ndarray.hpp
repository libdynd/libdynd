//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_HPP_
#define _DND__NDARRAY_HPP_

#include <iostream> // FOR DEBUG
#include <stdexcept>
#include <initializer_list>

#include <boost/utility/enable_if.hpp>

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/membuffer.hpp>
#include <dnd/shortvector.hpp>
#include <dnd/irange.hpp>

namespace dnd {

/** Typedef for vector of dimensions or strides */
typedef shortvector<intptr_t> dimvector;

class ndarray;

/**
 * This function constructs an array with the same shape and memory layout
 * of the one given, but with a different dtype.
 */
ndarray empty_like(const ndarray& rhs, const dtype& dt);
/** Stream printing function */
std::ostream& operator<<(std::ostream& o, const ndarray& rhs);

/**
 * This is the primary multi-dimensional array class.
 */
class ndarray {
    dtype m_dtype;
    int m_ndim;
    intptr_t m_num_elements;
    dimvector m_shape;
    dimvector m_strides;
    char *m_originptr;
    std::shared_ptr<membuffer> m_buffer;

    /**
     * Private method which constructs an array from all the members. This
     * function does not validate that the originptr/strides stay within
     * the buffer's bounds.
     */
    ndarray(const dtype& dt, int ndim, intptr_t size, const dimvector& shape,
            const dimvector& strides, char *originptr,
            const std::shared_ptr<membuffer>& buffer);

    /**
     * Private method for general indexing based on a raw array of irange
     * objects. Maybe this method should be public?
     */
    ndarray index(int nindex, const irange *indices) const;


public:
    /** Constructs an array with no buffer (NULL state) */
    ndarray();
    /** Constructs a zero-dimensional scalar array */
    explicit ndarray(const dtype& dt);
    /** Constructs a one-dimensional array */
    ndarray(intptr_t dim0, const dtype& dt);
    /** Constructs a two-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt);
    /** Constructs a three-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt);

    /**
     * Constructs an array from an initializer list.
     *
     * NOTE: These constructors are rather fragile, every initializer value
     *       has to be of the same type, otherwise the compiler thinks the
     *       initializer list is ambiguous and fails to bind to one of these
     *       functions.
     */
    template<class T>
    ndarray(std::initializer_list<T> il);
    /** Constructs an array from a two-level nested initializer list */
    template<class T>
    ndarray(std::initializer_list<std::initializer_list<T> > il);
    /** Constructs an array from a three-level nested initializer list */
    template<class T>
    ndarray(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il);

    /**
     * Constructs an array from a multi-dimensional C-style array.
     */
    template<class T, int N>
    ndarray(const T (&rhs)[N]);

    /** Copy constructor */
    ndarray(const ndarray& rhs)
        : m_dtype(rhs.m_dtype), m_ndim(rhs.m_ndim), m_num_elements(rhs.m_num_elements),
          m_shape(rhs.m_ndim, rhs.m_shape),
          m_strides(rhs.m_ndim, rhs.m_strides),
          m_originptr(rhs.m_originptr), m_buffer(rhs.m_buffer) {}
    /** Move constructor (should just be "= default" in C++11) */
    ndarray(ndarray&& rhs)
        : m_dtype(std::move(rhs.m_dtype)), m_ndim(rhs.m_ndim), m_num_elements(rhs.m_num_elements),
          m_shape(std::move(rhs.m_shape)), m_strides(std::move(rhs.m_strides)),
          m_originptr(rhs.m_originptr), m_buffer(std::move(rhs.m_buffer)) {}

    /** Swap operation (should be "noexcept" in C++11) */
    void swap(ndarray& rhs);

    /**
     * Assignment operator (should be just "= default" in C++11).
     *
     * TODO: This assignment operation copies 'rhs' with reference-like
     *       semantics. Should it instead copy the values of 'rhs' into
     *       'this'? The current way seems nicer for passing around arguments
     *       and making copies of arrays.
     */
    ndarray& operator=(const ndarray& rhs);
    /** Move assignment operator (should be just "= default" in C++11) */
    ndarray& operator=(ndarray&& rhs) {
        if (this != &rhs) {
            m_dtype = std::move(rhs.m_dtype);
            m_ndim = rhs.m_ndim;
            m_num_elements = rhs.m_num_elements;
            m_shape = std::move(rhs.m_shape);
            m_strides = std::move(rhs.m_strides);
            m_originptr = rhs.m_originptr;
            m_buffer = std::move(rhs.m_buffer);
        }

        return *this;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int ndim() const {
        return m_ndim;
    }

    const intptr_t *shape() const {
        return m_shape.get();
    }

    intptr_t shape(int i) const {
        return m_shape.get()[i];
    }

    const intptr_t *strides() const {
        return m_strides.get();
    }

    intptr_t strides(int i) const {
        return m_strides.get()[i];
    }

    intptr_t num_elements() const {
        return m_num_elements;
    }

    char *originptr() {
        return m_originptr;
    }

    // TODO: Should this return the non-const pointer?
    const char *originptr() const {
        return m_originptr;
    }

    /**
     * The ndarray uses the function call operator to do indexing. The [] operator
     * only supports one index object at a time, and while there are tricks that can be
     * done by overloading the comma operator, this doesn't produce a fool-proof result.
     * The function call operator behaves more consistently.
     */
    ndarray operator()(const irange& i0) const {
        return index(1, &i0);
    }
    /** Indexing with two index values */
    ndarray operator()(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return index(2, i);
    }
    /** Indexing with three index values */
    ndarray operator()(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return index(2, i);
    }
    /** Indexing with four index values */
    ndarray operator()(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return index(4, i);
    }
    /** Indexing with one integer index */
    ndarray operator()(intptr_t idx) const;

    /** Does a value-assignment from the rhs array. */
    void vassign(const ndarray& rhs, assign_error_mode errmode = assign_error_fractional);
    /** Does a value-assignment from the rhs raw scalar */
    void vassign(const dtype& dt, const void *data, assign_error_mode errmode = assign_error_fractional);
    /** Does a value-assignment from the rhs C++ scalar. */
    template<class T>
    typename boost::enable_if<is_dtype_scalar<T>, void>::type vassign(const T& rhs,
                                                assign_error_mode errmode = assign_error_fractional) {
        //DEBUG_COUT << "vassign C++ scalar\n";
        vassign(make_dtype<T>(), &rhs, errmode);
    }
    void vassign(const bool& rhs, assign_error_mode errmode = assign_error_fractional) {
        //DEBUG_COUT << "vassign bool\n";
        vassign(dnd_bool(rhs), errmode);
    }

    /**
     * Converts the array into the specified dtype. This function always makes a copy
     * even if the dtype is the same as the one for 'this'
     */
    ndarray as_dtype(const dtype& dt, assign_error_mode errmode = assign_error_fractional) const;

    /**
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type.
     *
     * TODO: Support ndarray.as_scalar<bool>()
     *
     * @param errmode  The assignment error mode to use.
     */
    template<class T>
    typename boost::enable_if<is_dtype_scalar<T>, T>::type as_scalar(
                                        assign_error_mode errmode = assign_error_fractional) const {
        T result;
        if (ndim() != 0) {
            throw std::runtime_error("can only convert ndarrays with 0 dimensions to scalars");
        }
        dtype_assign(make_dtype<T>(), &result, m_dtype, m_buffer->data(), errmode);
        return result;
    }

    friend ndarray empty_like(const ndarray& rhs, const dtype& dt);
    friend std::ostream& operator<<(std::ostream& o, const ndarray& rhs);
};

///////////// Initializer list constructor implementation /////////////////////////
namespace detail {
    // Computes the number of dimensions in a nested initializer list constructor
    template<class T>
    struct initializer_list_ndim {static const int value = 0;};
    template<class T>
    struct initializer_list_ndim<std::initializer_list<T> > {
        static const int value = initializer_list_ndim<T>::value + 1;
    };

    // Computes the array dtype of a nested initializer list constructor
    template<class T>
    struct initializer_list_dtype {typedef T type;};
    template<class T>
    struct initializer_list_dtype<std::initializer_list<T> > {
        typedef typename initializer_list_dtype<T>::type type;
    };

    // Gets the shape of the nested initializer list constructor, and validates that
    // it isn't ragged
    template <class T>
    struct initializer_list_shape;
    // Base case, an initializer list parameterized by a non-initializer list
    template<class T>
    struct initializer_list_shape<std::initializer_list<T> > {
        static void compute(intptr_t *out_shape, const std::initializer_list<T>& il) {
            out_shape[0] = il.size();
        }
        static void validate(const intptr_t *shape, const std::initializer_list<T>& il) {
            if (il.size() != shape[0]) {
                throw std::runtime_error("initializer list for ndarray is ragged, must be "
                                        "nested in a regular fashion");
            }
        }
        static void copy_data(T **dataptr, const std::initializer_list<T>& il) {
            memcpy(*dataptr, il.begin(), il.size() * sizeof(T));
            *dataptr += il.size();
        }
    };
    // Recursive case, an initializer list parameterized by an initializer list
    template<class T>
    struct initializer_list_shape<std::initializer_list<std::initializer_list<T> > > {
        static void compute(intptr_t *out_shape,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            out_shape[0] = il.size();
            if (out_shape[0] > 0) {
                // Recursively compute the rest of the shape
                initializer_list_shape<std::initializer_list<T> >::
                                                compute(out_shape + 1, *il.begin());
                // Validate the shape for the nested initializer lists
                for (auto i = il.begin() + 1; i != il.end(); ++i) {
                    initializer_list_shape<std::initializer_list<T> >::
                                                        validate(out_shape + 1, *i);
                }
            }
        }
        static void validate(const intptr_t *shape,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            if (il.size() != shape[0]) {
                throw std::runtime_error("initializer list for ndarray is ragged, must be "
                                        "nested in a regular fashion");
            }
            // Validate the shape for the nested initializer lists
            for (auto i = il.begin(); i != il.end(); ++i) {
                initializer_list_shape<std::initializer_list<T> >::validate(shape + 1, *i);
            }
        }
        static void copy_data(typename initializer_list_dtype<T>::type **dataptr,
                        const std::initializer_list<std::initializer_list<T> >& il) {
            for (auto i = il.begin(); i != il.end(); ++i) {
                initializer_list_shape<std::initializer_list<T> >::copy_data(dataptr, *i);
            }
        }
    };
} // namespace detail

// Implementation of initializer list construction
template<class T>
dnd::ndarray::ndarray(std::initializer_list<T> il)
    : m_dtype(type_id_of<T>::value), m_ndim(1), m_shape(1), m_strides(1)
{
    intptr_t size = il.size();
    m_shape[0] = size;
    m_num_elements = size;
    m_strides[0] = (size == 1) ? 0 : sizeof(T);
    if (size > 0) {
        // Allocate the storage buffer and copy the data
        m_buffer.reset(new membuffer(m_dtype, size));
        m_originptr = m_buffer->data();
        memcpy(m_originptr, il.begin(), sizeof(T)*size);
    } else {
        m_originptr = NULL;
    }
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<T> > il)
    : m_dtype(type_id_of<T>::value), m_ndim(2), m_shape(2), m_strides(2)
{
    typedef std::initializer_list<std::initializer_list<T> > S;

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(m_shape.get(), il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = m_ndim-1; i >= 0; --i) {
        m_strides[i] = (m_shape[i] == 1) ? 0 : stride;
        num_elements *= m_shape[i];
        stride *= m_shape[i];
    }
    m_num_elements = num_elements;
    if (num_elements > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, num_elements));
        m_originptr = m_buffer->data();
        // Populate the storage buffer from the nested initializer list
        T *dataptr = reinterpret_cast<T *>(m_originptr);
        detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    } else {
        m_originptr = NULL;
    }
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il)
    : m_dtype(type_id_of<T>::value), m_ndim(3), m_shape(3), m_strides(3)
{
    typedef std::initializer_list<std::initializer_list<std::initializer_list<T> > > S;

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(m_shape.get(), il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = m_ndim-1; i >= 0; --i) {
        m_strides[i] = (m_shape[i] == 1) ? 0 : stride;
        num_elements *= m_shape[i];
        stride *= m_shape[i];
    }
    m_num_elements = num_elements;
    if (num_elements > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, num_elements));
        m_originptr = m_buffer->data();
        // Populate the storage buffer from the nested initializer list
        T *dataptr = reinterpret_cast<T *>(m_originptr);
        detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    } else {
        m_originptr = NULL;
    }
}

///////////// C-style array constructor implementation /////////////////////////
namespace detail {
    template<class T> struct type_from_array {
        typedef T type;
        static const int itemsize = sizeof(T);
        static const int type_id = type_id_of<T>::value;
    };
    template<class T, int N> struct type_from_array<T[N]> {
        typedef typename type_from_array<T>::type type;
        static const int itemsize = type_from_array<T>::itemsize;
        static const int type_id = type_from_array<T>::type_id;
    };

    template<class T> struct ndim_from_array {static const int value = 0;};
    template<class T, int N> struct ndim_from_array<T[N]> {
        static const int value = ndim_from_array<T>::value + 1;
    };

    template<class T> struct fill_shape_and_strides_from_array {
        static intptr_t fill(intptr_t *, intptr_t *) {
            return sizeof(T);
        }
    };
    template<class T, int N> struct fill_shape_and_strides_from_array<T[N]> {
        static intptr_t fill(intptr_t *out_shape, intptr_t *out_strides) {
            intptr_t stride = fill_shape_and_strides_from_array<T>::
                                            fill(out_shape + 1, out_strides + 1);
            out_strides[0] = stride;
            out_shape[0] = N;
            return N * stride;
        }
    };
};

template<class T, int N>
dnd::ndarray::ndarray(const T (&rhs)[N])
    : m_dtype(detail::type_from_array<T>::type_id), m_ndim(detail::ndim_from_array<T[N]>::value),
      m_shape(m_ndim), m_strides(m_ndim)
{
    intptr_t num_bytes = detail::fill_shape_and_strides_from_array<T[N]>::
                                            fill(m_shape.get(), m_strides.get());
    m_num_elements = num_bytes / detail::type_from_array<T>::itemsize;
    if (m_num_elements > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, m_num_elements));
        m_originptr = m_buffer->data();
        // Populate the storage buffer from the nested initializer list
        memcpy(m_originptr, &rhs[0], num_bytes);
    } else {
        m_originptr = NULL;
    }
}

} // namespace dnd

#endif // _DND__NDARRAY_HPP_
