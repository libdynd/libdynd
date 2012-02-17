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

class ndarray_vals;

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
    ndarray(signed char value);
    ndarray(short value);
    ndarray(int value);
    ndarray(long value);
    ndarray(long long value);
    ndarray(unsigned char value);
    ndarray(unsigned short value);
    ndarray(unsigned int value);
    ndarray(unsigned long value);
    ndarray(unsigned long long value);
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

#ifdef DND_INIT_LIST
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
#endif // DND_INIT_LIST

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

    /**
     * Assignment operator from an ndarray_vals object. This converts the
     * array 'rhs' has a reference to into a strided array, evaluating
     * it from the expression tree if necessary. If 'rhs' contains a
     * strided array, this copies it by reference, use the function 'copy'
     * when a copy is required.
     */
    ndarray& operator=(const ndarray_vals& rhs);

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

    dnd::shared_ptr<void> get_buffer_owner() const {
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
     * Returns a value-exposing helper object, which allows one to assign to
     * the values of the ndarray, or collapse the expression tree of the
     * ndarray into a strided array.
     */
    ndarray_vals vals() const;

    /**
     * The ndarray uses the function call operator to do indexing. The [] operator
     * only supports one index object at a time, and while there are tricks that can be
     * done by overloading the comma operator, this doesn't produce a fool-proof result.
     * The function call operator behaves more consistently.
     */
    const ndarray operator()(const irange& i0) const {
        return index(1, &i0);
    }
    /** Indexing with two index values */
    const ndarray operator()(const irange& i0, const irange& i1) const {
        irange i[2] = {i0, i1};
        return index(2, i);
    }
    /** Indexing with three index values */
    const ndarray operator()(const irange& i0, const irange& i1, const irange& i2) const {
        irange i[3] = {i0, i1, i2};
        return index(3, i);
    }
    /** Indexing with four index values */
    const ndarray operator()(const irange& i0, const irange& i1, const irange& i2, const irange& i3) const {
        irange i[4] = {i0, i1, i2, i3};
        return index(4, i);
    }
    /** Indexing with one integer index */
    const ndarray operator()(intptr_t idx) const;

    /** Does a value-assignment from the rhs array. */
    void val_assign(const ndarray& rhs, assign_error_mode errmode = assign_error_fractional) const;
    /** Does a value-assignment from the rhs raw scalar */
    void val_assign(const dtype& dt, const void *data, assign_error_mode errmode = assign_error_fractional) const;

    /**
     * Converts the array into the specified dtype.
     */
    ndarray as_dtype(const dtype& dt, assign_error_mode errmode = assign_error_fractional) const;

    /**
     * Converts the array into the specified explicit template dtype.
     */
    template<class T>
    ndarray as_dtype(assign_error_mode errmode = assign_error_fractional) const {
        return as_dtype(make_dtype<T>(), errmode);
    }

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
    friend class ndarray_vals;
};

ndarray operator+(const ndarray& op0, const ndarray& op1);
ndarray operator-(const ndarray& op0, const ndarray& op1);
ndarray operator/(const ndarray& op0, const ndarray& op1);
ndarray operator*(const ndarray& op0, const ndarray& op1);

/**
 * This is a helper class for dealing with value assignment and collapsing
 * a view-based ndarray into a strided array. Only the ndarray class itself
 * is permitted to construct this helper object, and it is non-copyable.
 *
 * All that can be done is assigning the values of the referenced array
 * to another array, or assigning values from another array into the elements
 * the referenced array.
 */
class ndarray_vals {
    const ndarray& m_arr;
    ndarray_vals(const ndarray& arr)
        : m_arr(arr) {
    }

    // Non-copyable, not default-constructable
    ndarray_vals(const ndarray_vals&);
    ndarray_vals& operator=(const ndarray_vals&);
public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    ndarray_vals& operator=(const ndarray& rhs) {
        m_arr.val_assign(rhs);
        return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template<class T>
    typename boost::enable_if<is_dtype_scalar<T>, ndarray_vals&>::type operator=(const T& rhs) {
        m_arr.val_assign(make_dtype<T>(), &rhs);
        return *this;
    }
    /** Does a value-assignment from the rhs C++ boolean scalar. */
    ndarray_vals& operator=(const bool& rhs) {
        dnd_bool v = rhs;
        m_arr.val_assign(make_dtype<dnd_bool>(), &v);
        return *this;
    }

    // TODO: Could also do +=, -=, *=, etc.

    // Can implicitly convert to an ndarray, by collapsing to a strided array
    operator ndarray() const {
        return ndarray((m_arr.m_expr_tree->get_node_type() == strided_array_node_type)
                        ? m_arr.m_expr_tree
                        : m_arr.m_expr_tree->evaluate());
    }

    friend class ndarray;
};

inline ndarray_vals ndarray::vals() const {
    return ndarray_vals(*this);
}

inline ndarray& ndarray::operator=(const ndarray_vals& rhs) {
    // No copy if the rhs is already a strided array
    m_expr_tree = (rhs.m_arr.m_expr_tree->get_node_type() == strided_array_node_type)
                        ? rhs.m_arr.m_expr_tree
                        : rhs.m_arr.m_expr_tree->evaluate();
    return *this;
}

///////////// Initializer list constructor implementation /////////////////////////
#ifdef DND_INIT_LIST
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
            DND_MEMCPY(*dataptr, il.begin(), il.size() * sizeof(T));
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
    : m_expr_tree()
{
    intptr_t dim0 = il.size();
    intptr_t stride = (dim0 == 1) ? 0 : sizeof(T);
    dnd::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * dim0),
                    ::dnd::detail::ndarray_buffer_deleter);
    DND_MEMCPY(buffer_owner.get(), il.begin(), sizeof(T) * dim0);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 1, &dim0, &stride,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<T> > il)
    : m_expr_tree()
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
    dnd::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * num_elements),
                    ::dnd::detail::ndarray_buffer_deleter);
    T *dataptr = reinterpret_cast<T *>(buffer_owner.get());
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 2, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
template<class T>
dnd::ndarray::ndarray(std::initializer_list<std::initializer_list<std::initializer_list<T> > > il)
    : m_expr_tree()
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
    dnd::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(sizeof(T) * num_elements),
                    ::dnd::detail::ndarray_buffer_deleter);
    T *dataptr = reinterpret_cast<T *>(buffer_owner.get());
    detail::initializer_list_shape<S>::copy_data(&dataptr, il);
    m_expr_tree.reset(new strided_array_expr_node(make_dtype<T>(), 3, shape, strides,
                            reinterpret_cast<char *>(buffer_owner.get()), std::move(buffer_owner)));
}
#endif // DND_INIT_LIST

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
    : m_expr_tree()
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
    dnd::shared_ptr<void> buffer_owner(
                    ::dnd::detail::ndarray_buffer_allocator(num_bytes),
                    ::dnd::detail::ndarray_buffer_deleter);
    DND_MEMCPY(buffer_owner.get(), &rhs[0], num_bytes);
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
                ndarray tmp = lhs.vals();
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
