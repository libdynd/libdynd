//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <iostream> // FOR DEBUG
#include <stdexcept>
#include <initializer_list>

#include <boost/utility/enable_if.hpp>

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/membuffer.hpp>
#include <dnd/shortvector.hpp>

namespace dnd {

/** Typedef for vector of dimensions or strides */
typedef shortvector<intptr_t, 3> dimvector;

/**
 * This is the primary multi-dimensional array class.
 */
class ndarray {
    dtype m_dtype;
    int m_ndim;
    dimvector m_shape;
    dimvector m_strides;
    intptr_t m_baseoffset;
    std::shared_ptr<membuffer> m_buffer;

    /**
     * Private method which constructs an array from all the members. This
     * function does not validate that the strides/baseoffset stay within
     * the buffer's bounds.
     */
    ndarray(const dtype& dt, int ndim, const dimvector& shape,
            const dimvector& strides, intptr_t baseoffset,
            const std::shared_ptr<membuffer>& buffer);

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
        : m_dtype(rhs.m_dtype), m_ndim(rhs.m_ndim),
          m_shape(rhs.m_ndim, rhs.m_shape),
          m_strides(rhs.m_ndim, rhs.m_strides),
          m_baseoffset(rhs.m_baseoffset), m_buffer(rhs.m_buffer) {}
    /** Move constructor (should just be "= default" in C++11) */
    ndarray(ndarray&& rhs)
        : m_dtype(std::move(rhs.m_dtype)), m_ndim(rhs.m_ndim),
          m_shape(std::move(rhs.m_shape)), m_strides(std::move(rhs.m_strides)),
          m_baseoffset(rhs.m_baseoffset), m_buffer(std::move(rhs.m_buffer)) {}

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
            m_shape = std::move(rhs.m_shape);
            m_strides = std::move(rhs.m_strides);
            m_baseoffset = rhs.m_baseoffset;
            m_buffer = std::move(rhs.m_buffer);
        }

        return *this;
    }
    /** Initializer list assignment operator */
    template <class T>
    ndarray& operator=(const std::initializer_list<T>& il) {
        *this = ndarray(il);
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

    const intptr_t *strides() const {
        return m_strides.get();
    }

    char *data() {
        return m_buffer->data() + m_baseoffset;
    }

    const char *data() const {
        return m_buffer->data() + m_baseoffset;
    }

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
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type.
     *
     * @param errmode  The assignment error mode to use.
     */
    template<class T>
    typename boost::enable_if<is_dtype_scalar<T>, T>::type as_scalar(
                                        assign_error_mode errmode = assign_error_fractional) {
        T result;
        if (ndim() != 0) {
            throw std::runtime_error("can only convert ndarrays with 0 dimensions to scalars");
        }
        dtype_assign(make_dtype<T>(), &result, m_dtype, m_buffer->data(), errmode);
        return result;
    }
    
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
    : m_dtype(type_id_of<T>::value), m_ndim(1), m_shape(1), m_strides(1), m_baseoffset(0)
{
    intptr_t size = il.size();
    m_shape[0] = size;
    m_strides[0] = (size == 1) ? 0 : sizeof(T);
    if (size > 0) {
        // Allocate the storage buffer and copy the data
        m_buffer.reset(new membuffer(m_dtype, size));
        memcpy(m_buffer->data(), il.begin(), sizeof(T)*size);
    }
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<T> > il)
    : m_dtype(type_id_of<T>::value), m_ndim(2), m_shape(2), m_strides(2), m_baseoffset(0)
{
    typedef std::initializer_list<std::initializer_list<T> > S;

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(m_shape.get(), il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t size = 1, stride = sizeof(T);
    for (int i = m_ndim-1; i >= 0; --i) {
        m_strides[i] = (m_shape[i] == 1) ? 0 : stride;
        size *= m_shape[i];
        stride *= m_shape[i];
    }
    if (size > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, size));
        // Populate the storage buffer from the nested initializer list
        T *dataptr = reinterpret_cast<T *>(m_buffer->data());
        detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    }
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il)
    : m_dtype(type_id_of<T>::value), m_ndim(3), m_shape(3), m_strides(3), m_baseoffset(0)
{
    typedef std::initializer_list<std::initializer_list<std::initializer_list<T> > > S;

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(m_shape.get(), il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t size = 1, stride = sizeof(T);
    for (int i = m_ndim-1; i >= 0; --i) {
        m_strides[i] = (m_shape[i] == 1) ? 0 : stride;
        size *= m_shape[i];
        stride *= m_shape[i];
    }
    if (size > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, size));
        // Populate the storage buffer from the nested initializer list
        T *dataptr = reinterpret_cast<T *>(m_buffer->data());
        detail::initializer_list_shape<S>::copy_data(&dataptr, il);
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
      m_shape(m_ndim), m_strides(m_ndim), m_baseoffset(0)
{
    intptr_t size = detail::fill_shape_and_strides_from_array<T[N]>::
                                            fill(m_shape.get(), m_strides.get());
    size /= detail::type_from_array<T>::itemsize;
    if (size > 0) {
        // Allocate the storage buffer
        m_buffer.reset(new membuffer(m_dtype, size));
        // Populate the storage buffer from the nested initializer list
        memcpy(m_buffer->data(), &rhs[0], detail::type_from_array<T>::itemsize * size);
    }
}

} // namespace dnd

#endif//_NDARRAY_HPP_
