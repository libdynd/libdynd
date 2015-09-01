//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {

/**
 * A helper class which owns an ndt::type and memory
 * for corresponding array arrmeta, but does not itself
 * hold any data.
 *
 * An example usage for this is within
 * a ckernel to produce subset views of an array that
 * a child ckernel can use without building a full-blown
 * reference counted nd::array object.
 */
class DYND_API arrmeta_holder {
    /** This memory holds one ndt::type, followed by its array arrmeta */
    void *m_arrmeta;

    // Non-copyable
    arrmeta_holder(const arrmeta_holder&);
    arrmeta_holder& operator=(const arrmeta_holder&);
public:
    arrmeta_holder() : m_arrmeta(NULL) {}
    arrmeta_holder(const ndt::type &tp)
        : m_arrmeta(malloc(sizeof(ndt::type) + tp.get_arrmeta_size()))
    {
        if (!m_arrmeta) {
            throw std::bad_alloc();
        }
        memset(reinterpret_cast<char *>(m_arrmeta) + sizeof(ndt::type), 0,
               tp.get_arrmeta_size());
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
            if (tp.get_arrmeta_size() > 0) {
                tp.extended()->arrmeta_destruct(
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

    void arrmeta_default_construct(bool blockref_alloc)
    {
      const ndt::type &tp = get_type();
      if (!tp.is_builtin()) {
        tp.extended()->arrmeta_default_construct(get(), blockref_alloc);
      }
    }
};

} // namespace dynd
