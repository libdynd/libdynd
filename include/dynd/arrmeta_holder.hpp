//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRMETA_HOLDER_HPP_
#define _DYND__ARRMETA_HOLDER_HPP_

#include <dynd/type.hpp>

namespace dynd {

/**
 * A helper class which owns an ndt::type and memory
 * for corresponding array metadata, but does not itself
 * hold any data.
 *
 * An example usage for this is within
 * a ckernel to produce subset views of an array that
 * a child ckernel can use without building a full-blown
 * reference counted nd::array object.
 */
class arrmeta_holder {
    /** This memory holds one ndt::type, followed by its array metadata */
    void *m_arrmeta;

    // Non-copyable
    arrmeta_holder(const arrmeta_holder&);
    arrmeta_holder& operator=(const arrmeta_holder&);
public:
    arrmeta_holder() : m_arrmeta(NULL) {}
    arrmeta_holder(const ndt::type &tp)
        : m_arrmeta(malloc(sizeof(ndt::type) + tp.get_metadata_size()))
    {
        if (!m_arrmeta) {
            throw std::bad_alloc();
        }
        memset(reinterpret_cast<char *>(m_arrmeta) + sizeof(ndt::type), 0,
               tp.get_metadata_size());
        try {
            new (m_arrmeta) ndt::type(tp);
        } catch(...) {
            free(m_arrmeta);
            throw;
        }
    }

    ~arrmeta_holder()
    {
        if (m_arrmeta != NULL) {
            ndt::type &tp = *reinterpret_cast<ndt::type *>(m_arrmeta);
            if (tp.get_metadata_size() > 0) {
                tp.extended()->metadata_destruct(
                    reinterpret_cast<char *>(m_arrmeta) + sizeof(ndt::type));
            }
            tp.~type();
            free(m_arrmeta);
        }
    }

    void swap(arrmeta_holder& rhs) {
        std::swap(m_arrmeta, rhs.m_arrmeta);
    }

    const ndt::type& get_type() const {
        return *reinterpret_cast<const ndt::type *>(m_arrmeta);
    }

    char *get()
    {
        return reinterpret_cast<char *>(m_arrmeta) + sizeof(ndt::type);
    }

    template<class MetaType>
    MetaType *get_at(intptr_t offset)
    {
        return reinterpret_cast<MetaType *>(
            reinterpret_cast<char *>(m_arrmeta) + sizeof(ndt::type) + offset);
    }

    void metadata_default_construct(intptr_t ndim, const intptr_t *shape) {
        const ndt::type& tp = get_type();
        if (!tp.is_builtin()) {
            tp.extended()->metadata_default_construct(get(), ndim, shape);
        }
    }
};

} // namespace dynd

#endif // _DYND__ARRMETA_HOLDER_HPP_
