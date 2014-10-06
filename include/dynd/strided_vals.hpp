//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_VALS_HPP_
#define _DYND__STRIDED_VALS_HPP_

#include <dynd/type.hpp>

namespace dynd {

struct start_stop_t {
    intptr_t start;
    intptr_t stop;
};

namespace nd {

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

namespace detail {

template <int N>
struct strided {
    static const char *get(const char *pointer, const intptr_t *stride, const intptr_t *index) {
        return strided<N - 1>::get(pointer, stride, index) + index[N - 1] * stride[N - 1];
    }

    static const char *get(const char *pointer, const size_stride_t *size_stride, const intptr_t *index) {
        return strided<N - 1>::get(pointer, size_stride, index) + index[N - 1] * size_stride[N - 1].stride;
    }
};

template <>
struct strided<1> {
    static const char *get(const char *pointer, const intptr_t stride, const intptr_t i0) {
        return pointer + i0 * stride;
    }

    static const char *get(const char *pointer, const intptr_t *stride, const intptr_t *i) {
        return pointer + i[0] * stride[0];
    }

    static const char *get(const char *pointer, const size_stride_t *size_stride, const intptr_t *i) {
        return pointer + i[0] * size_stride[0].stride;
    }
};

}

template <typename T, int N>
class strided_vals {
    struct {
        const char *pointer;
        size_stride_t size_stride[N];
        const start_stop_t *start_stop;
    } m_data;
    struct {
        const char *pointer;
        intptr_t stride[N];
    } m_mask;

public:
    strided_vals() : m_data({NULL}), m_mask({NULL}) {
    }

    intptr_t get_ndim() const {
        return N;
    }

    size_stride_t get_size_stride(intptr_t i) const {
        return m_data.size_stride[i];
    }

    intptr_t get_dim_size(intptr_t i) const {
        return m_data.size_stride[i].dim_size;
    }

    intptr_t get_size() const {
        if (m_mask.pointer != NULL) {
            return -1;
        }

        return 0;
    }

    intptr_t get_stride(intptr_t i) const {
        return m_data.size_stride[i].stride;
    }

    const T *get_readonly_originptr() const {
        return reinterpret_cast<const T *>(m_data.pointer);
    }

    void set_readonly_originptr(const char *data) {
        m_data.pointer = data;
    }

    void set_data(const char *data) {
        m_data.pointer = data; // is this legal?
    }

    void set_data(const char *data, const size_stride_t *size_stride, const start_stop_t *start_stop = NULL) {
        m_data.pointer = data;
        for (intptr_t i = 0; i < N; ++i) {
            m_data.size_stride[i] = size_stride[i];
        }
        m_data.start_stop = start_stop;
    }

    const char *get_mask() const {
        return m_mask.pointer;
    }

    void set_mask(const char *mask) {
        m_mask.pointer = mask;
    }

    void set_mask(const char *mask, const intptr_t *stride) {
        m_mask.pointer = mask;
        for (intptr_t i = 0; i < N; ++i) {
            m_mask.stride[i] = stride[i];
        }
    }

    void set_mask(const char *mask, const size_stride_t *size_stride) {
        m_mask.pointer = mask;
        for (intptr_t i = 0; i < N; ++i) {
            m_mask.stride[i] = size_stride[i].stride;
        }
    }

    const T &operator()(intptr_t i0) const {
        return *reinterpret_cast<const T *>(detail::strided<1>::get(m_data.pointer, m_data.size_stride[0].stride, i0));
    }

    const T &operator()(intptr_t i0, intptr_t i1) const {
//        std::cout << "start = (" << m_start[0] << ", " << m_start[1] << ")" << std::endl;
  //      std::cout << "stop = (" << m_stop[0] << ", " << m_stop[1] << ")" << std::endl;
        return *reinterpret_cast<const T *>(m_data.pointer + i0 * m_data.size_stride[0].stride + i1 * m_data.size_stride[1].stride);
    }

    const T &operator()(intptr_t i0, intptr_t i1, intptr_t i2) const {
        return *reinterpret_cast<const T *>(m_data.pointer + i0 * m_data.size_stride[0].stride + i1 * m_data.size_stride[1].stride + i2 *m_data.size_stride[2].stride);
    }

    bool is_null() const {
        return m_data.pointer == NULL;
    }

    bool is_valid(intptr_t i0, intptr_t i1) const {
        return (i0 >= m_data.start_stop[0].start) && (i0 < m_data.start_stop[0].stop) && (i1 >= m_data.start_stop[1].start) && (i1 < m_data.start_stop[1].stop);
    }

    bool is_valid(intptr_t *index) const {
        bool res = true;
        for (int i = 0; i < N; ++i) {
            res = res && (index[i] >= m_data.start_stop[i].start) && (index[i] < m_data.start_stop[i].stop);
        }

        return res;
    }

    bool is_masked(intptr_t i0) const {
        return m_mask.pointer == NULL
            || *reinterpret_cast<const dynd_bool *>(m_mask.pointer + i0 * m_mask.stride[0]);
    }

    bool is_masked(const intptr_t *i) const {
        return m_mask.pointer == NULL
            || *reinterpret_cast<const dynd_bool *>(detail::strided<N>::get(m_mask.pointer, m_mask.stride, i));
    }

    class iterator {
        const strided_vals<T, N> &m_vals;
        const char *m_data;
        intptr_t m_index[N];

    public:
        iterator(const strided_vals<T, N> &vals, intptr_t offset = 0)
          : m_vals(vals), m_data(vals.m_data.pointer + offset) {
            for (intptr_t i = 0; i < N; ++i) {
                m_index[i] = 0;
            }
        }

        iterator& operator++() {
            do {
                iterator_helper<N - 1>::template incr<T, N>(*this);
            } while (*this != m_vals.end() && (!m_vals.is_valid(m_index) || !m_vals.is_masked(m_index)));
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

        const T &operator *() const {
            return *reinterpret_cast<const T *>(m_data);
        }

        template <int K>
        friend struct iterator_helper;
    };

    iterator begin() const {
        intptr_t index[N];
        for (int i = 0; i < N; ++i) {
            index[i] = 0;
        }

        if (is_valid(index) && is_masked(index)) {
            return iterator(*this);
        } else {
            return ++iterator(*this);
        }
    }

    iterator end() const {
        return iterator(*this, m_data.size_stride[0].dim_size * m_data.size_stride[0].stride);
    }
};

}} // namespace dynd::nd

#endif
