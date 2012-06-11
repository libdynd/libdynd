//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__AUXILIARY_DATA_HPP_
#define _DND__AUXILIARY_DATA_HPP_

#include <new>
#include <algorithm>

namespace dnd {

// AuxDataBase is the same as NpyAuxData, see the numpy doc link
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#auxiliary-data-with-object-semantics
struct AuxDataBase;

// Function pointers for freeing or cloning auxiliary data
// IMPORTANT NOTE: These are C-ABI functions, they should
//                 not throw exceptions!
typedef void (AuxData_FreeFunc) (AuxDataBase *);
typedef AuxDataBase *(AuxData_CloneFunc) (const AuxDataBase *);

struct AuxDataBase {
    AuxData_FreeFunc *free;
    AuxData_CloneFunc *clone;
    // To allow for a bit of expansion without breaking ABI
    void *reserved[2];
};

class auxiliary_data;

template<typename T>
void make_auxiliary_data(auxiliary_data& out_created);

namespace detail {

    template<typename T>
    void auxiliary_data_holder_free(AuxDataBase *auxdata);

    template<typename T>
    AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata);

    /**
     * The auxiliary data class encapsulates data lifetime-managed
     * data which pairs with kernel functions. This object is oriented
     * to be C-ABI compatible, in particular it is binary compatible with
     * the NpyAuxData structure in Numpy. The difference with Numpy's version
     * is that we use C++ templates to wrap it in an easier-to-use class
     * object instead of having some macros.
     *
     */
    template<typename Taux>
    class auxiliary_data_holder {
        AuxDataBase m_base;
        Taux m_auxdata;

        // Only make_auxiliary_data<T> can default-construct one of these
        auxiliary_data_holder() {
        }

        // Only auxiliary_data_holder_free<T> can delete one of these
        ~auxiliary_data_holder() {
        }

        // Only auxiliary_data_holder_clone<T> can copy one of these
        auxiliary_data_holder(const auxiliary_data_holder& rhs)
            : m_base(rhs.m_base), m_auxdata(rhs.m_auxdata)
        {}

        // Non-copyable
        auxiliary_data_holder& operator=(const auxiliary_data_holder&);

        template<typename T>
        friend void ::dnd::make_auxiliary_data(auxiliary_data& out_created);
        template<typename T>
        friend void auxiliary_data_holder_free(AuxDataBase *auxdata);
        template<typename T>
        friend AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata);

        friend class ::dnd::auxiliary_data;
    public:
        const Taux& get() const {
            return m_auxdata;
        }
    };

    template<typename T>
    void auxiliary_data_holder_free(AuxDataBase *auxdata)
    {
        auxiliary_data_holder<T> *adh = reinterpret_cast<auxiliary_data_holder<T> *>(auxdata);
        delete adh;
    }

    template<typename T>
    AuxDataBase *auxiliary_data_holder_clone(const AuxDataBase *auxdata)
    {
        const auxiliary_data_holder<T> *adh = reinterpret_cast<const auxiliary_data_holder<T> *>(auxdata);
        try {
            return reinterpret_cast<AuxDataBase *>(new auxiliary_data_holder<T>(*adh));
        } catch(const std::exception&) {
            return 0;
        }
    }

} // namespace detail

/**
 * Holds an auxiliary data object, and manages its lifetime.
 * It's non-copyable, as no reference counting has been added
 * for performance reasons, and we want to support pre-C++11.
 * With C++11, we would follow the example of unique_ptr<T>.
 */
class auxiliary_data {
    AuxDataBase *m_auxdata;

    // Non-copyable
    auxiliary_data(const auxiliary_data&);
    auxiliary_data& operator=(const auxiliary_data&);
    // Would be nice if we could rely on it being movable, like std::unique_ptr,
    // but we probably need to support pre C++11...
public:
    auxiliary_data() {
        m_auxdata = 0;
    }
    ~auxiliary_data() {
        free();
    }

    bool empty() {
        return m_auxdata != 0;
    }

    // Frees the auxdata memory, makes this auxdata NULL
    void free() {
        if (m_auxdata != 0) {
            m_auxdata->free(m_auxdata);
            m_auxdata = 0;
        }
    }

    void clone_into(auxiliary_data& out_cloned) const {
        out_cloned.free();
        if (m_auxdata != 0) {
            out_cloned.m_auxdata = m_auxdata->clone(m_auxdata);
            if (out_cloned.m_auxdata == 0) {
                throw std::bad_alloc();
            }
        }
    }

    void swap(auxiliary_data& rhs) {
        std::swap(m_auxdata, rhs.m_auxdata);
    }

    // When the auxiliary_data was created with make_auxiliary_data<T>, this
    // returns a reference to the T member. This should only be called when
    // this->empty() returns false.
    template<typename T>
    T& get() {
        return reinterpret_cast<detail::auxiliary_data_holder<T> *>(m_auxdata)->m_auxdata;
    }

    // Allow implicit conversion to const AuxDataBase *, so that this
    // can be passed as a parameter to kernel functions.
    operator const AuxDataBase *() const {
        return m_auxdata;
    }

    template<typename T>
    friend void make_auxiliary_data(auxiliary_data& out_created);
};

/**
 * Default-constructs an auxiliary data object, with a payload of the requested type.
 * This will typically be paired with a kernel function which uses a
 * const AuxDataBase* for auxiliary data.
 */
template<typename T>
void make_auxiliary_data(auxiliary_data& out_created)
{
    out_created.free();
    out_created.m_auxdata = reinterpret_cast<AuxDataBase *>(new detail::auxiliary_data_holder<T>());
    out_created.m_auxdata->free = detail::auxiliary_data_holder_free<T>;
    out_created.m_auxdata->clone = detail::auxiliary_data_holder_clone<T>;
}

/**
 * Default-constructs an auxiliary data object, with a payload of the requested type.
 * This will typically be paired with a kernel function which uses a
 * const AuxDataBase* for auxiliary data.
 *
 * After construction, assigns the given value into the auxdata.
 */
template<typename T>
void make_auxiliary_data(auxiliary_data& out_created, const T& value)
{
    make_auxiliary_data<T>(out_created);
    out_created.get<T>() = value;
}

/**
 * When auxdata points to an object created with make_auxiliary_data<T>,
 * this returns a reference to the T object it contains. Should not be
 * called when auxdata is NULL.
 */
template<typename T>
const T& get_auxiliary_data(const AuxDataBase *auxdata)
{
    return reinterpret_cast<const detail::auxiliary_data_holder<T>*>(auxdata)->get();
}


} // namespace dnd

#endif // _DND__AUXILIARY_DATA_HPP_