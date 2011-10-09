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
#include <dnd/shortvector.hpp>
#include <dnd/irange.hpp>
#include <dnd/ndarray_expr_node.hpp>

namespace dnd {

class ndarray;

/** Stream printing function */
std::ostream& operator<<(std::ostream& o, const ndarray& rhs);

namespace detail {
    void *ndarray_buffer_allocator(intptr_t size);
    void ndarray_buffer_deleter(void *ptr);
} // namespace detail

/**
 * This is the primary multi-dimensional array class.
 */
class ndarray {
    /**
     * The ndarray is fully contained in an expression tree, this is the root node, a
     * boost_intrusive_ptr, to make the ndarray size equivalent to a single pointer.
     */
    ndarray_expr_node_ptr m_expr_tree;

    /**
     * Private method for general indexing based on a raw array of irange
     * objects. Maybe this method should be public?
     */
    ndarray index(int nindex, const irange *indices) const;


public:
    /** Constructs an array with no buffer (NULL state) */
    ndarray();
    /**
     * Constructs a zero-dimensional scalar array from a C++ scalar.
     *
     * TODO: Figure out why enable_if with is_dtype_scalar didn't work for this constructor
     *       in g++ 4.6.0.
     */
    ndarray(int8_t value);
    ndarray(int16_t value);
    ndarray(int32_t value);
    ndarray(int64_t value);
    ndarray(uint8_t value);
    ndarray(uint16_t value);
    ndarray(uint32_t value);
    ndarray(uint64_t value);
    ndarray(float value);
    ndarray(double value);
    /** Constructs a zero-dimensional scalar array */
    explicit ndarray(const dtype& dt);
    /** Constructs an array with the given dtype, shape, and axis_perm (for memory layout) */
    ndarray(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm)
        : m_expr_tree(new strided_array_expr_node(dt, ndim, shape, axis_perm)) {
    }

    /** Constructs an ndaray from an expr node */
    explicit ndarray(const ndarray_expr_node_ptr& expr_tree);
    /** Constructs an ndaray from an expr node */
    explicit ndarray(ndarray_expr_node_ptr&& expr_tree);

    /** Constructs a one-dimensional array */
    ndarray(intptr_t dim0, const dtype& dt);
    /** Constructs a two-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, const dtype& dt);
    /** Constructs a three-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, const dtype& dt);
    /** Constructs a four-dimensional array */
    ndarray(intptr_t dim0, intptr_t dim1, intptr_t dim2, intptr_t dim3, const dtype& dt);

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
        : m_expr_tree(rhs.m_expr_tree) {}
    /** Move constructor (should just be "= default" in C++11) */
    ndarray(ndarray&& rhs)
        : m_expr_tree(std::move(rhs.m_expr_tree)) {}

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
        m_expr_tree = std::move(rhs.m_expr_tree);

        return *this;
    }

    const dtype& get_dtype() const {
        return m_expr_tree->get_dtype();
    }

    int get_ndim() const {
        return m_expr_tree->get_ndim();
    }

    const intptr_t *get_shape() const {
        return m_expr_tree->get_shape();
    }

    intptr_t get_shape(int i) const {
        return m_expr_tree->get_shape()[i];
    }

    const intptr_t *get_strides() const {
        if (m_expr_tree->get_node_type() == strided_array_node_type) {
            return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_strides();
        } else {
            throw std::runtime_error("cannot get the strides of an expression view ndarray");
        }
    }

    intptr_t get_strides(int i) const {
        if (m_expr_tree->get_node_type() == strided_array_node_type) {
            return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_strides()[i];
        } else {
            throw std::runtime_error("cannot get the strides of an expression view ndarray");
        }
    }

    intptr_t get_num_elements() const {
        intptr_t nelem = 1, ndim = get_ndim();
        const intptr_t *shape = get_shape();
        for (int i = 0; i < ndim; ++i) {
            nelem *= shape[i];
        }
        return nelem;
    }

    char *get_originptr() const {
        if (m_expr_tree->get_node_type() == strided_array_node_type) {
            return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_originptr();
        } else {
            throw std::runtime_error("cannot get the origin pointer of an expression view ndarray");
        }
    }

    std::shared_ptr<void> get_buffer_owner() const {
        if (m_expr_tree->get_node_type() == strided_array_node_type) {
            return static_cast<const strided_array_expr_node *>(m_expr_tree.get())->get_buffer_owner();
        } else {
            throw std::runtime_error("cannot get the buffer owner of an expression view ndarray");
        }
    }

    ndarray_expr_node *get_expr_tree() const {
        return m_expr_tree.get();
    }

    /**
     * Returns true if this is an array which owns its own data,
     * and all the data elements are aligned according to the needs
     * of the data type.
     *
     * This operation loops through all the strides to compute whether it is aligned.
     */
    bool is_aligned() const {
        if (m_expr_tree->get_node_type() == strided_array_node_type) {
            const strided_array_expr_node *node =
                            static_cast<const strided_array_expr_node *>(m_expr_tree.get());
            return node->is_aligned();
        } else {
            return false;
        }
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
        return index(3, i);
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
     * When the array is an expression-based view, creates a new ndarray and evaluates the
     * expression into that array. When the array is a strided array, simply returns the array.
     */
    ndarray as_strided() const;

    /**
     * Converts the array into the specified dtype. This function always makes a copy
     * even if the dtype is the same as the one for 'this'
     */
    ndarray as_dtype(const dtype& dt, assign_error_mode errmode = assign_error_fractional) const;

    /**
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type. This function may be extended in the future for
     * 1D vectors (as<std::vector<T>>), matrices, etc.
     *
     * @param errmode  The assignment error mode to use.
     */
    template<class T>
    T as(assign_error_mode errmode = assign_error_fractional) const;

    void debug_dump(std::ostream& o) const;

    friend std::ostream& operator<<(std::ostream& o, const ndarray& rhs);
};

ndarray operator+(const ndarray& op0, const ndarray& op1);
ndarray operator-(const ndarray& op0, const ndarray& op1);
ndarray operator/(const ndarray& op0, const ndarray& op1);
ndarray operator*(const ndarray& op0, const ndarray& op1);

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
            if ((intptr_t)il.size() != shape[0]) {
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
            if ((intptr_t)il.size() != shape[0]) {
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
{
    intptr_t dim0 = il.size();
    intptr_t stride = (dim0 == 1) ? 0 : sizeof(T);
    std::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * dim0),
                    ::dnd::detail::ndarray_buffer_deleter);
    memcpy(buffer_owner.get(), il.begin(), sizeof(T) * dim0);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 1, &dim0, &stride,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<T> > il)
{
    typedef std::initializer_list<std::initializer_list<T> > S;
    intptr_t shape[2], strides[2];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = 1; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        num_elements *= shape[i];
        stride *= shape[i];
    }
    std::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * num_elements),
                    ::dnd::detail::ndarray_buffer_deleter);
    T *dataptr = reinterpret_cast<T *>(buffer_owner.get());
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 2, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il)
{
    typedef std::initializer_list<std::initializer_list<std::initializer_list<T> > > S;
    intptr_t shape[3], strides[3];

    // Get and validate that the shape is regular
    detail::initializer_list_shape<S>::compute(shape, il);
    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = sizeof(T);
    for (int i = 2; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        num_elements *= shape[i];
        stride *= shape[i];
    }
    std::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * num_elements),
                    ::dnd::detail::ndarray_buffer_deleter);
    T *dataptr = reinterpret_cast<T *>(buffer_owner.get());
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 3, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
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
{
    intptr_t shape[detail::ndim_from_array<T[N]>::value], strides[detail::ndim_from_array<T[N]>::value];
    const int ndim = detail::ndim_from_array<T[N]>::value;
    intptr_t num_bytes = detail::fill_shape_and_strides_from_array<T[N]>::fill(shape, strides);

    // Compute the number of elements in the array, and the strides at the same time
    intptr_t num_elements = 1, stride = detail::type_from_array<T>::itemsize;
    for (int i = ndim-1; i >= 0; --i) {
        strides[i] = (shape[i] == 1) ? 0 : stride;
        num_elements *= shape[i];
        stride *= shape[i];
    }
    std::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(num_bytes),
                    ::dnd::detail::ndarray_buffer_deleter);
    memcpy(buffer_owner.get(), &rhs[0], num_bytes);
    m_expr_tree.reset(new strided_array_expr_node(dtype(detail::type_from_array<T>::type_id),
                            ndim, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}

///////////// The ndarray.as<type>() templated function /////////////////////////
namespace detail {
    template <class T>
    struct ndarray_as_helper {
        static typename boost::enable_if<is_dtype_scalar<T>, T>::type as(const ndarray& lhs,
                                                                    assign_error_mode errmode) {
            T result;
            if (lhs.get_ndim() != 0) {
                throw std::runtime_error("can only convert ndarrays with 0 dimensions to scalars");
            }
            if (lhs.get_expr_tree()->get_node_type() == strided_array_node_type) {
                const strided_array_expr_node *node =
                        static_cast<const strided_array_expr_node *>(lhs.get_expr_tree());
                dtype_assign(make_dtype<T>(), &result, node->get_dtype(), node->get_originptr(), errmode);
            } else {
                ndarray tmp = lhs.as_strided();
                const strided_array_expr_node *node =
                        static_cast<const strided_array_expr_node *>(tmp.get_expr_tree());
                dtype_assign(make_dtype<T>(), &result, node->get_dtype(), node->get_originptr(), errmode);
            }
            return result;
        }
    };

    template <>
    struct ndarray_as_helper<bool> {
        static bool as(const ndarray& lhs, assign_error_mode errmode) {
            return ndarray_as_helper<dnd_bool>::as(lhs, errmode);
        }
    };

    // Could do as<std::vector<T>> for 1D arrays, and other similiar conversions
} // namespace detail;

template<class T>
T dnd::ndarray::as(assign_error_mode errmode) const {
    return detail::ndarray_as_helper<T>::as(*this, errmode);
}

/**
 * Constructs an array with the same shape and memory layout
 * of the one given, but with a different dtype.
 */
ndarray empty_like(const ndarray& rhs, const dtype& dt);

/**
 * Constructs an empty array matching the parameters of 'rhs'
 */
inline ndarray empty_like(const ndarray& rhs) {
    return empty_like(rhs, rhs.get_dtype());
}

} // namespace dnd

#endif // _DND__NDARRAY_HPP_
