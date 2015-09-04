//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {

struct DYND_API start_stop_t {
    intptr_t start;
    intptr_t stop;
};

namespace nd {

namespace detail {

template <int N>
struct strided_utils {
    static const char *get(const char *pointer, const intptr_t *index, const intptr_t *strides) {
        return strided_utils<N - 1>::get(pointer, strides, index) + index[N - 1] * strides[N - 1];
    }

    static bool is_valid(const intptr_t *index, const start_stop_t *start_stop) {
        return strided_utils<N - 1>::is_valid(index, start_stop) && (index[N - 1] >= start_stop[N - 1].start) && (index[N - 1] < start_stop[N - 1].stop);
    }

    static void incr(const char *&pointer, intptr_t *index, const intptr_t *sizes, const intptr_t *strides) {
        if (++index[N - 1] != sizes[N - 1]) {
            pointer += strides[N - 1];
        } else {
            index[N - 1] = 0;
            pointer -= (sizes[N - 1] - 1) * strides[N - 1];
            strided_utils<N - 1>::incr(pointer, index, sizes, strides);
        }
    }
};

template <>
struct strided_utils<1> {
    static const char *get(const char *pointer, const intptr_t *index, const intptr_t *strides) {
        return pointer + index[0] * strides[0];
    }

    static bool is_valid(const intptr_t *index, const start_stop_t *start_stop) {
        return (index[0] >= start_stop[0].start) && (index[0] < start_stop[0].stop);
    }

    static void incr(const char *&pointer, intptr_t *index, const intptr_t *DYND_UNUSED(sizes), const intptr_t *strides) {
        ++index[0];
        pointer += strides[0];
    }
};

template <typename T, int N>
class strided_vals {
    struct {
        const char *pointer;
        intptr_t strides[N];
    } m_data;
    struct {
        const char *pointer;
        intptr_t strides[N];
    } m_mask;
    intptr_t m_sizes[N];
    const start_stop_t *m_start_stop;
    intptr_t m_center_index[N];

public:
    const char *get_data() const {
        return m_data.pointer;
    }

    const char *get_mask() const {
        return m_mask.pointer;
    }

    intptr_t get_ndim() const {
        return N;
    }

    intptr_t get_size(intptr_t i) const {
        return m_sizes[i];
    }

    const intptr_t *get_sizes() const {
        return m_sizes;
    }

    intptr_t get_stride(intptr_t i) const {
        return m_data.strides[i];
    }

    const intptr_t *get_strides() const {
        return m_data.strides;
    }

    const intptr_t *get_center_index() const {
        return m_center_index;
    }

    void set_data(const char *data) {
        m_data.pointer = data;
    }

    void set_data(const char *data, const size_stride_t *size_stride, const start_stop_t *start_stop = NULL) {
        m_data.pointer = data;
        for (intptr_t i = 0; i < N; ++i) {
            m_data.strides[i] = size_stride[i].stride;
            m_sizes[i] = size_stride[i].dim_size;
            m_center_index[i] = (m_sizes[i] - 1) / 2;
        }
        m_start_stop = start_stop;
    }

    void set_mask(const char *mask) {
        m_mask.pointer = mask;
    }

    void set_mask(const char *mask, const size_stride_t *size_stride) {
        m_mask.pointer = mask;
        for (intptr_t i = 0; i < N; ++i) {
            m_mask.strides[i] = size_stride[i].stride;
        }
    }

    const T &operator ()(const intptr_t *index) const {
        return *reinterpret_cast<const T *>(detail::strided_utils<N>::get(m_data.pointer, index, m_data.strides));
    }

    bool is_masked(const intptr_t *index) const {
        return m_mask.pointer == NULL
            || *reinterpret_cast<const bool1 *>(detail::strided_utils<N>::get(m_mask.pointer, index, m_mask.strides));
    }

    bool is_valid(const intptr_t *index) const {
        return strided_utils<N>::is_valid(index, m_start_stop);
    }

    class iterator {
        const strided_vals<T, N> &m_vals;
        const char *m_data;
        intptr_t m_index[N];

    public:
        iterator(const strided_vals<T, N> &vals, intptr_t offset = 0) : m_vals(vals), m_data(vals.get_data() + offset) {
            memset(m_index, 0, N * sizeof(intptr_t));
        }

        const intptr_t *get_index() const {
            return m_index;
        }

        iterator& operator++() {
            do {
                strided_utils<N>::incr(m_data, m_index, m_vals.get_sizes(), m_vals.get_strides());
            } while (*this != m_vals.end() && !(m_vals.is_masked(m_index) && m_vals.is_valid(m_index)));
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
    };

    iterator begin() const {
        iterator it(*this);
        if (is_masked(it.get_index()) && is_valid(it.get_index())) {
            return it;
        }

        return ++it;
    }

    iterator end() const {
        return iterator(*this, m_sizes[0] * m_data.strides[0]);
    }
};

}

template <typename T, int N>
class strided_vals : public detail::strided_vals<T, N> {
};

template <typename T>
class strided_vals<T, 1> : public detail::strided_vals<T, 1> {
public:
    const T &operator ()(const intptr_t *index) const {
        return detail::strided_vals<T, 1>::operator ()(index);
    }

    const T &operator ()(intptr_t i0) const {
        return operator ()(&i0);
    }

    bool is_masked(const intptr_t *index) const {
        return detail::strided_vals<T, 1>::is_masked(index);
    }

    bool is_masked(intptr_t i0) const {
        return is_masked(&i0);
    }

    bool is_valid(const intptr_t *index) const {
        return detail::strided_vals<T, 1>::is_valid(index);
    }

    bool is_valid(intptr_t i0) const {
        return is_valid(&i0);
    }
};

template <typename T>
class strided_vals<T, 2> : public detail::strided_vals<T, 2> {
public:
    const T &operator ()(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::operator ()(index);
    }

    const T &operator ()(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return operator ()(index);
    }

    bool is_masked(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::is_masked(index);
    }

    bool is_masked(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return is_masked(index);
    }

    bool is_valid(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::is_valid(index);
    }

    bool is_valid(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return is_valid(index);
    }
};

template <typename T>
class strided_vals<T, 3> : public detail::strided_vals<T, 3> {
public:
    const T &operator ()(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::operator ()(index);
    }

    const T &operator ()(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return operator ()(index);
    }

    bool is_masked(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::is_masked(index);
    }

    bool is_masked(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return is_masked(index);
    }

    bool is_valid(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::is_valid(index);
    }

    bool is_valid(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return is_valid(index);
    }
};

}} // namespace dynd::nd
