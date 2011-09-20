//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _RAW_ITERATION_HPP_
#define _RAW_ITERATION_HPP_

#include <algorithm>

#include <dnd/ndarray.hpp>

namespace dnd {

/** Use raw_ndarray_iter<N> to iterate over N arrays simultaneously */
template<int N>
class raw_ndarray_iter;

template<>
class raw_ndarray_iter<1> {
    int m_ndim;
    dimvector m_iterindex;
    dimvector m_itershape;
    char *m_data;
    dimvector m_strides;

    void init(int ndim, const intptr_t *shape, char *data, const intptr_t *strides)
    {
        m_ndim = ndim;

        // Special case 0 and 1 dimensional arrays
        if (m_ndim == 0) {
            m_ndim = 1;
            m_data = data;
            // NOTE: This is ok, because shortvectors always have
            //       at least 1 element even if initialized with size = 0.
            m_itershape[0] = 1;
            m_strides[0] = 0;
            return;
        } else if (m_ndim == 1) {
            m_itershape[0] = shape[0];
            // Always make the stride positive
            if (strides[0] >= 0) {
                m_data = data;
                m_strides[0] = strides[0];
            } else {
                m_data = data + strides[0] * (shape[0] - 1);
                m_strides[0] = -strides[0];
            }
            return;
        }

        // Sort the strides in C order, and copy into 'this'
        shortvector<int,3> strideperm(m_ndim);
        for (int i = 0; i < m_ndim; ++i) {
            strideperm[i] = i;
        }
        std::sort(strideperm.get(), strideperm.get() + ndim,
                        [&m_strides](int i, int j) -> bool {
            intptr_t astride = m_strides[i], bstride = m_strides[j];
            // Take the absolute value
            if (astride < 0) astride = -astride;
            if (bstride < 0) bstride = -bstride;

            return astride < bstride;
        });
        for (int i = 0; i < m_ndim; ++i) {
            int p = strideperm[i];
            m_itershape[i] = shape[p];
            m_strides[i] = strides[p];
        }

        // Reverse any negative strides
        for (int i = 0; i < m_ndim; ++i) {
            intptr_t stride = m_strides[i], size = m_itershape[i];

            if (stride < 0) {
                data += stride * (size - 1);
                m_strides[i] = -stride;
            }

            // Detect and handle a zero-size array
            if (size == 0) {
                m_ndim = 1;
                m_data = data;
                m_itershape[0] = 0;
                m_strides[0] = 0;
                return;
            }
        }

        // Coalesce axes where possible
        int i = 0;
        for (int j = 1; j < m_ndim; ++j) {
            if (m_itershape[i] == 1) {
                // Remove axis i
                m_itershape[i] = m_itershape[j];
                m_strides[i] = m_strides[j];
            } else if (m_itershape[j] == 1) {
                // Remove axis j
            } else if (m_strides[i] * m_itershape[i] == m_strides[j]) {
                // Coalesce axes i and j
                m_itershape[i] *= m_itershape[j];
            } else {
                // Can't coalesce, go to the next i
                ++i;
                m_itershape[i] = m_itershape[j];
                m_strides[i] = m_strides[j];
            }
        }
        m_ndim = i+1;
        m_data = data;
        memset(m_iterindex.get(), 0, m_ndim * sizeof(intptr_t));
    }
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape, char *data, const intptr_t *strides)
        : m_iterindex(ndim), m_itershape(ndim), m_strides(ndim)
    {
        init(ndim, shape, data, strides);
    }

    raw_ndarray_iter(ndarray& arr)
        : m_iterindex(arr.ndim()), m_itershape(arr.ndim()), m_strides(arr.ndim())
    {
        init(arr.ndim(), arr.shape(), arr.data(), arr.strides());
    }

    intptr_t innersize() const {
        return m_itershape[0];
    }

    intptr_t innerstride() const {
        return m_strides[0];
    }

    char *data() {
        return m_data;
    }

    bool iternext() {
        int i = 1;
        for (; i < m_ndim; ++i) {
            intptr_t size = m_itershape[i];
            if (++m_iterindex[i] == size) {
                m_iterindex[i] = 0;
                m_data -= (size - 1) * m_strides[i];
            } else {
                m_data += m_strides[i];
                break;
            }
        }

        return i < m_ndim;
    }
};

} // namespace dnd

#endif // _RAW_ITERATION_HPP_
