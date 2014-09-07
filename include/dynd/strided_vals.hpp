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

template <int K>
struct iterator_helper {
    template <typename T, int N>
    static void incr(typename strided_vals<T, N>::iterator &it) {
        size_stride_t ss = it.m_vals.get_size_stride(K);
        if (++it.m_index[K] != ss.dim_size) {
            it.m_data += ss.stride;
        } else {
            it.m_index[K] = 0;
            it.m_data -= (ss.dim_size - 1) * ss.stride;
            iterator_helper<K - 1>::template incr<T, N>(it);
        }
    }
};

template <>
struct iterator_helper<0> {
    template <typename T, int N>
    static void incr(typename strided_vals<T, N>::iterator &it) {
        ++it.m_index[0];
        it.m_data += it.m_vals.get_stride(0);
    }
};

template <typename T, int N>
class strided_vals {
    const char *m_data;
    size_stride_t m_size_stride[N];

public:
    class iterator {
        const strided_vals<T, N> &m_vals;
        const char *m_data;
        intptr_t m_index[N];

    public:
        iterator(const strided_vals<T, N> &vals, intptr_t offset = 0)
          : m_vals(vals), m_data(vals.m_data + offset) {
            for (intptr_t i = 0; i < N; ++i) {
                m_index[i] = 0;
            }
        }

        iterator& operator++() {
            iterator_helper<N - 1>::template incr<T, N>(*this);
            return *this;
        }

        iterator operator++(int) {
            iterator tmp(*this);
            operator++();
            return tmp;
        }

        bool operator==(const iterator &other) const {
            return &m_vals == &other.m_vals && m_data == other.m_data;
        }

        bool operator!=(const iterator &other) const {
            return !(*this == other);
        }

        const T *operator *() const {
            return reinterpret_cast<const T *>(m_data);
        }

        template <int K>
        friend class iterator_helper;
    };

    strided_vals() : m_data(NULL) {
    }

    void init(const size_stride_t *ss) {
        DYND_MEMCPY(m_size_stride, ss, N * sizeof(size_stride_t));
    }

    intptr_t get_ndim() const {
        return N;
    }

    size_stride_t get_size_stride(intptr_t i) const {
        return m_size_stride[i];
    }

    intptr_t get_dim_size(intptr_t i) const {
        return m_size_stride[i].dim_size;
    }

    intptr_t get_stride(intptr_t i) const {
        return m_size_stride[i].stride;
    }

    const T *get_readonly_originptr() const {
        return reinterpret_cast<const T *>(m_data);
    }

    void set_readonly_originptr(const char *data) {
        m_data = data;
    }

    const T *operator()(intptr_t i0) const {
        return reinterpret_cast<const T *>(m_data + i0 * m_size_stride[0].stride);
    }

    const T *operator()(intptr_t i0, intptr_t i1) const {
        return reinterpret_cast<const T *>(m_data + i0 * m_size_stride[0].stride + i1 * m_size_stride[1].stride);
    }

    const T *operator()(intptr_t i0, intptr_t i1, intptr_t i2) const {
        return reinterpret_cast<const T *>(m_data + i0 * m_size_stride[0].stride + i1 * m_size_stride[1].stride + i2 * m_size_stride[2].stride);
    }

    iterator begin() const {
        return iterator(*this);
    }

    iterator end() const {
        return iterator(*this, m_size_stride[0].dim_size * m_size_stride[0].stride);
    }
};

}} // namespace dynd::nd

#endif
