//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _RAW_ITERATION_HPP_
#define _RAW_ITERATION_HPP_

#include <algorithm>

#include <boost/utility/enable_if.hpp>

#include <dnd/ndarray.hpp>

namespace dnd {

namespace detail {
    template<int N, int staticNDIM = 3>
    class raw_ndarray_iter_base {
        intptr_t *m_strides_alloc_data;
        intptr_t m_strides_static_data[staticNDIM*N];
    protected:
        int m_ndim;
        dimvector m_iterindex;
        dimvector m_itershape;
        char *m_data[N];
        intptr_t *m_strides[N];

        raw_ndarray_iter_base(int ndim)
          : m_ndim(ndim), m_iterindex(ndim), m_itershape(ndim)
        {
            if (ndim <= staticNDIM) {
                m_strides_alloc_data = NULL;
                for (int i = 0; i < N; ++i) {
                    m_strides[i] = &m_strides_static_data[ndim*i];
                }
            } else {
                m_strides_alloc_data = new intptr_t[ndim*N];
                for (int i = 0; i < N; ++i) {
                    m_strides[i] = &m_strides_alloc_data[ndim*i];
                }
            }
        }

        ~raw_ndarray_iter_base() {
            if (m_strides_alloc_data != NULL) {
                delete[] m_strides_alloc_data;
            }
        }

        inline bool strides_can_coalesce(int i, int j) {
            intptr_t size = m_itershape[i];
            for (int k = 0; k < N; ++k) {
                if (m_strides[k][i] * size != m_strides[k][j]) {
                    return false;
                }
            }
            return true;
        }

        void init(const intptr_t *shape, char **data, const intptr_t **strides)
        {
            for (int k = 0; k < N; ++k) {
                m_data[k] = data[k];
            }

            // Special case 0 and 1 dimensional arrays
            if (m_ndim == 0) {
                m_ndim = 1;
                // NOTE: This is ok, because shortvectors always have
                //       at least 1 element even if initialized with size = 0.
                m_iterindex[0] = 0;
                m_itershape[0] = 1;
                for (int k = 0; k < N; ++k) {
                    m_strides[k][0] = 0;
                }
                return;
            } else if (m_ndim == 1) {
                intptr_t size = shape[0];
                m_iterindex[0] = 0;
                m_itershape[0] = size;
                // Always make the stride positive
                if (strides[0][0] >= 0) {
                    for (int k = 0; k < N; ++k) {
                        m_strides[k][0] = strides[k][0];
                    }
                } else {
                    for (int k = 0; k < N; ++k) {
                        m_data[k][0] += strides[k][0] * (size - 1);
                        m_strides[k][0] = -strides[k][0];
                    }
                }
                return;
            }

            // Sort the strides in C order according to the first operand, and copy into 'this'
            shortvector<int,staticNDIM> strideperm(m_ndim);
            for (int i = 0; i < m_ndim; ++i) {
                strideperm[i] = i;
            }
            const intptr_t *strides0 = strides[0];
            std::sort(strideperm.get(), strideperm.get() + m_ndim,
                            [&strides0](int i, int j) -> bool {
                intptr_t astride = strides0[i], bstride = strides0[j];
                // Take the absolute value
                if (astride < 0) astride = -astride;
                if (bstride < 0) bstride = -bstride;

                return astride < bstride;
            });
            for (int i = 0; i < m_ndim; ++i) {
                int p = strideperm[i];
                m_itershape[i] = shape[p];
                for (int k = 0; k < N; ++k) {
                    m_strides[k][i] = strides[k][p];
                }
            }

            // Reverse any axes where the first operand has a negative stride
            for (int i = 0; i < m_ndim; ++i) {
                intptr_t stride = m_strides[0][i], size = m_itershape[i];

                if (stride < 0) {
                    m_data[0] += stride * (size - 1);
                    m_strides[0][i] = -stride;
                    for (int k = 1; k < N; ++k) {
                        m_data[k] += m_strides[k][i] * (size - 1);
                        m_strides[k][i] = -m_strides[k][i];
                    }
                }

                // Detect and handle a zero-size array
                if (size == 0) {
                    m_ndim = 1;
                    m_itershape[0] = 0;
                    for (int k = 0; k < N; ++k) {
                        m_strides[k][0] = 0;
                    }
                    return;
                }
            }

            // Coalesce axes where possible
            int i = 0;
            for (int j = 1; j < m_ndim; ++j) {
                if (m_itershape[i] == 1) {
                    // Remove axis i
                    m_itershape[i] = m_itershape[j];
                    for (int k = 0; k < N; ++k) {
                        m_strides[k][i] = m_strides[k][j];
                    }
                } else if (m_itershape[j] == 1) {
                    // Remove axis j
                } else if (strides_can_coalesce(i, j)) {
                    // Coalesce axes i and j
                    m_itershape[i] *= m_itershape[j];
                } else {
                    // Can't coalesce, go to the next i
                    ++i;
                    m_itershape[i] = m_itershape[j];
                    for (int k = 0; k < N; ++k) {
                        m_strides[k][i] = m_strides[k][j];
                    }
                }
            }
            m_ndim = i+1;
            memset(m_iterindex.get(), 0, m_ndim * sizeof(intptr_t));
        }

    public:
        bool iternext() {
            int i = 1;
            for (; i < m_ndim; ++i) {
                intptr_t size = m_itershape[i];
                if (++m_iterindex[i] == size) {
                    m_iterindex[i] = 0;
                    for (int k = 0; k < N; ++k) {
                        m_data[k] -= (size - 1) * m_strides[k][i];
                    }
                } else {
                    for (int k = 0; k < N; ++k) {
                        m_data[k] += m_strides[k][i];
                    }
                    break;
                }
            }

            return i < m_ndim;
        }

        intptr_t innersize() const {
            return m_itershape[0];
        }

        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < N), intptr_t>::type innerstride() const {
            return m_strides[K][0];
        }

        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < N), char *>::type data() {
            return m_data[K];
        }

        /**
         * Gets a byte suitable for the 'align_test' argument in dtype's
         * is_data_aligned function.
         */
        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < N), char>::type get_align_test() const {
            char result = static_cast<char>(reinterpret_cast<intptr_t>(m_data[K]));
            for (int i = 0; i < m_ndim; ++i) {
                result |= m_strides[K][i];
            }
            return result;
        }

        /*
        int m_ndim;
        dimvector m_iterindex;
        dimvector m_itershape;
        char *m_data[N];
        intptr_t *m_strides[N];
        */

        /**
         * Prints out a debug dump of the object.
         */
        void debug_dump(std::ostream& o) {
            o << "------ raw_ndarray_iter<" << N << ">\n";
            o << " ndim: " << m_ndim << "\n";
            o << " iterindex: ";
            for (int i = 0; i < m_ndim; ++i) o << m_iterindex[i] << " ";
            o << "\n";
            o << " itershape: ";
            for (int i = 0; i < m_ndim; ++i) o << m_itershape[i] << " ";
            o << "\n";
            o << " data: ";
            for (int k = 0; k < N; ++k) o << (void *)m_data[k] << " ";
            o << "\n";
            for (int k = 0; k < N; ++k) {
                o << " strides[" << k << "]: ";
                for (int i = 0; i < m_ndim; ++i) o << m_strides[k][i] << " ";
                o << "\n";
            }
            o << "------\n";
        }
    };
} // namespace detail

/** Use raw_ndarray_iter<N> to iterate over N arrays simultaneously */
template<int N>
class raw_ndarray_iter;

template<>
class raw_ndarray_iter<1> : public detail::raw_ndarray_iter_base<1> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape, char *data, const intptr_t *strides)
        : detail::raw_ndarray_iter_base<1>(ndim)
    {
        init(shape, &data, &strides);
    }

    raw_ndarray_iter(ndarray& arr)
        : detail::raw_ndarray_iter_base<1>(arr.ndim())
    {
        char *data = arr.originptr();
        const intptr_t *strides = arr.strides();
        init(arr.shape(), &data, &strides);
    }

};

template<>
class raw_ndarray_iter<2> : public detail::raw_ndarray_iter_base<2> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape,
                                char *dataA, const intptr_t *stridesA,
                                char *dataB, const intptr_t *stridesB)
        : detail::raw_ndarray_iter_base<2>(ndim)
    {
        char *data[2] = {dataA, dataB};
        const intptr_t *strides[2] = {stridesA, stridesB};
        init(shape, data, strides);
    }
};

} // namespace dnd

#endif // _RAW_ITERATION_HPP_
