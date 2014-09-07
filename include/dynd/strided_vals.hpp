//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_VALS_HPP_
#define _DYND__STRIDED_VALS_HPP_

#include <dynd/types/base_type.hpp>

namespace dynd { namespace nd {

template <typename T, int N>
class strided_vals;

template <typename T, int N, int K = N - 1>
struct iterator_helper {
    typedef typename strided_vals<T, N>::iterator iterator_type;

    static iterator_type &incr(iterator_type &it) {
        if (++it.m_index[K] != it.m_src.m_ss[K].dim_size) {
            it.m_data += it.m_src.m_ss[K].stride;
            return it;
        }

        it.m_index[K] = 0;
        it.m_data -= (it.m_src.m_ss[K].dim_size - 1) * it.m_src.m_ss[K].stride;
        return iterator_helper<T, N, K - 1>::incr(it);
    }
};

template <typename T, int N>
struct iterator_helper<T, N, 0> {
    typedef typename strided_vals<T, N>::iterator iterator_type;

    static iterator_type &incr(iterator_type &it) {
        ++it.m_index[0];
        it.m_data += it.m_src.m_ss[0].stride;
        return it;
    }
};

template <typename T, int N>
class strided_vals {
public:
    size_stride_t m_ss[N];
    const char *m_data_pointer;

public:
    class iterator {
    public:
        const strided_vals<T, N> &m_src;
        const char *m_data;
        intptr_t m_index[N];

    public:
        iterator(const strided_vals<T, N> &src) : m_src(src), m_data(src.m_data_pointer) {
            memset(m_index, 0, N * sizeof(intptr_t));
        }

        iterator& operator++() {
            return iterator_helper<T, N>::incr(*this);
        }

        iterator operator++(int) {
            iterator tmp(*this);
            operator++();
            return tmp;
        }

        bool operator!=(const iterator &other) const {
            return m_data != other.m_data;
        }

        const T *operator *() const {
            return reinterpret_cast<const T *>(m_data);
        }
    };

    strided_vals() : m_data_pointer(NULL) {
    }

    void init(const size_stride_t *ss) {
        DYND_MEMCPY(m_ss, ss, N * sizeof(size_stride_t));
    }

    intptr_t get_ndim() const {
        return N;
    }

    intptr_t get_dim_size(intptr_t i) const {
        return m_ss[i].dim_size;
    }

    const T *get_readonly_originptr() const {
        return reinterpret_cast<const T *>(m_data_pointer);
    }

    void set_readonly_originptr(const char *data_pointer) {
        m_data_pointer = data_pointer;
    }

    const T *operator()(intptr_t i0, intptr_t i1) const {
        return reinterpret_cast<const T *>(m_data_pointer + i0 * m_ss[0].stride + i1 * m_ss[1].stride);
    }

    const T *operator()(intptr_t i0, intptr_t i1, intptr_t i2) const {
        return reinterpret_cast<const T *>(m_data_pointer + i0 * m_ss[0].stride + i1 * m_ss[1].stride + i2 * m_ss[2].stride);
    }

    iterator begin() const {
        return iterator(*this);
    }

    iterator end() const {
        iterator it = iterator(*this);
        it.m_data = m_data_pointer + m_ss[0].dim_size * m_ss[0].stride;
        return it;
    }
};

}} // namespace dynd::nd

#endif
