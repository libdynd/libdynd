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
    template<int Nwrite, int Nread, int staticNDIM = 3>
    class raw_ndarray_iter_base {
        // contains all the ndim-sized vectors
        // 0: iterindex
        // 1: itershape
        // 2 ... (Nwrite + Nread) + 1: all the operand strides
        multi_shortvector<intptr_t, Nwrite + Nread + 2, staticNDIM> m_vectors;
    protected:
        int m_ndim;
        char *m_data[Nwrite + Nread];

        raw_ndarray_iter_base(int ndim)
          : m_ndim(ndim), m_vectors(ndim) {
        }

        intptr_t& iterindex(int i) {
            return m_vectors.get(0, i);
        }

        const intptr_t& iterindex(int i) const {
            return m_vectors.get(0, i);
        }

        intptr_t& itershape(int i) {
            return m_vectors.get(1, i);
        }

        const intptr_t& itershape(int i) const {
            return m_vectors.get(1, i);
        }

        intptr_t* strides(int k) {
            return m_vectors.get_all()[k+2];
        }

        const intptr_t* strides(int k) const {
            return m_vectors.get_all()[k+2];
        }

        inline bool strides_can_coalesce(int i, int j) {
            intptr_t size = itershape(i);
            for (int k = 0; k < Nwrite + Nread; ++k) {
                if (strides(k)[i] * size != strides(k)[j]) {
                    return false;
                }
            }
            return true;
        }

        void init(const intptr_t *shape, char **data, const intptr_t **in_strides)
        {
            for (int k = 0; k < Nwrite + Nread; ++k) {
                m_data[k] = data[k];
            }

            // Special case 0 and 1 dimensional arrays
            if (m_ndim == 0) {
                m_ndim = 1;
                // NOTE: This is ok, because shortvectors always have
                //       at least 1 element even if initialized with size = 0.
                iterindex(0) = 0;
                itershape(0) = 1;
                for (int k = 0; k < Nwrite + Nread; ++k) {
                    strides(k)[0] = 0;
                }
                return;
            } else if (m_ndim == 1) {
                intptr_t size = shape[0];
                iterindex(0) = 0;
                itershape(0) = size;
                // Always make the stride positive
                if (in_strides[0][0] >= 0) {
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        strides(k)[0] = in_strides[k][0];
                    }
                } else {
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        m_data[k][0] += in_strides[k][0] * (size - 1);
                        strides(k)[0] = -in_strides[k][0];
                    }
                }
                return;
            }

            // Sort the strides in C order according to the first operand, and copy into 'this'
            shortvector<int,staticNDIM> strideperm(m_ndim);
            for (int i = 0; i < m_ndim; ++i) {
                strideperm[i] = i;
            }
            const intptr_t *strides0 = in_strides[0];
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
                itershape(i) = shape[p];
                for (int k = 0; k < Nwrite + Nread; ++k) {
                    strides(k)[i] = in_strides[k][p];
                }
            }

            // Reverse any axes where the first operand has a negative stride
            for (int i = 0; i < m_ndim; ++i) {
                intptr_t stride = strides(0)[i], size = itershape(i);

                if (stride < 0) {
                    m_data[0] += stride * (size - 1);
                    strides(0)[i] = -stride;
                    for (int k = 1; k < Nwrite + Nread; ++k) {
                        m_data[k] += strides(k)[i] * (size - 1);
                        strides(k)[i] = -strides(k)[i];
                    }
                }

                // Detect and handle a zero-size array
                if (size == 0) {
                    m_ndim = 1;
                    itershape(0) = 0;
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        strides(k)[0] = 0;
                    }
                    return;
                }
            }

            // Coalesce axes where possible
            int i = 0;
            for (int j = 1; j < m_ndim; ++j) {
                if (itershape(i) == 1) {
                    // Remove axis i
                    itershape(i) = itershape(j);
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        strides(k)[i] = strides(k)[j];
                    }
                } else if (itershape(j) == 1) {
                    // Remove axis j
                } else if (strides_can_coalesce(i, j)) {
                    // Coalesce axes i and j
                    itershape(i) *= itershape(j);
                } else {
                    // Can't coalesce, go to the next i
                    ++i;
                    itershape(i) = itershape(j);
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        strides(k)[i] = strides(k)[j];
                    }
                }
            }
            m_ndim = i+1;
            memset(&iterindex(0), 0, m_ndim * sizeof(intptr_t));
        }

    public:
        bool iternext() {
            int i = 1;
            for (; i < m_ndim; ++i) {
                intptr_t size = itershape(i);
                if (++iterindex(i) == size) {
                    iterindex(i) = 0;
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        m_data[k] -= (size - 1) * strides(k)[i];
                    }
                } else {
                    for (int k = 0; k < Nwrite + Nread; ++k) {
                        m_data[k] += strides(k)[i];
                    }
                    break;
                }
            }

            return i < m_ndim;
        }

        intptr_t innersize() const {
            return itershape(0);
        }

        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < Nwrite + Nread), intptr_t>::type
                                                                        innerstride() const {
            return strides(K)[0];
        }

        /**
         * Provide non-const access to the 'write' operands.
         */
        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < Nwrite), char *>::type data() {
            return m_data[K];
        }

        /**
         * Provide const access to all the operands.
         */
        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < Nwrite + Nread), const char *>::type
                                                                        data() const {
            return m_data[K];
        }

        /**
         * Gets a byte suitable for the 'align_test' argument in dtype's
         * is_data_aligned function.
         */
        template<int K>
        typename boost::enable_if_c<(K >= 0 && K < Nwrite + Nread), char>::type
                                                                        get_align_test() const {
            char result = static_cast<char>(reinterpret_cast<intptr_t>(m_data[K]));
            for (int i = 0; i < m_ndim; ++i) {
                result |= strides(K)[i];
            }
            return result;
        }

        /**
         * Prints out a debug dump of the object.
         */
        void debug_dump(std::ostream& o) {
            o << "------ raw_ndarray_iter<" << Nwrite << ", " << Nread << ">\n";
            o << " ndim: " << m_ndim << "\n";
            o << " iterindex: ";
            for (int i = 0; i < m_ndim; ++i) o << iterindex(i) << " ";
            o << "\n";
            o << " itershape: ";
            for (int i = 0; i < m_ndim; ++i) o << itershape(i) << " ";
            o << "\n";
            o << " data: ";
            for (int k = 0; k < Nwrite + Nread; ++k) o << (void *)m_data[k] << " ";
            o << "\n";
            for (int k = 0; k < Nwrite + Nread; ++k) {
                o << " strides[" << k << "]: ";
                for (int i = 0; i < m_ndim; ++i) o << strides(k)[i] << " ";
                o << "\n";
            }
            o << "------\n";
        }
    };
} // namespace detail

/**
 * Use raw_ndarray_iter<Nwrite, int Nread> to iterate over Nwrite output and Nread
 * input arrays simultaneously.
 */
template<int Nwrite, int Nread>
class raw_ndarray_iter;

template<>
class raw_ndarray_iter<0,1> : public detail::raw_ndarray_iter_base<0,1> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape, const char *data, const intptr_t *strides)
        : detail::raw_ndarray_iter_base<0,1>(ndim)
    {
        init(shape, const_cast<char **>(&data), &strides);
    }

    raw_ndarray_iter(ndarray& arr)
        : detail::raw_ndarray_iter_base<0,1>(arr.ndim())
    {
        char *data = arr.originptr();
        const intptr_t *strides = arr.strides();
        init(arr.shape(), const_cast<char **>(&data), &strides);
    }

};

template<>
class raw_ndarray_iter<1,0> : public detail::raw_ndarray_iter_base<1,0> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape, const char *data, const intptr_t *strides)
        : detail::raw_ndarray_iter_base<1,0>(ndim)
    {
        init(shape, const_cast<char **>(&data), &strides);
    }

    raw_ndarray_iter(ndarray& arr)
        : detail::raw_ndarray_iter_base<1,0>(arr.ndim())
    {
        char *data = arr.originptr();
        const intptr_t *strides = arr.strides();
        init(arr.shape(), const_cast<char **>(&data), &strides);
    }

};

template<>
class raw_ndarray_iter<1,1> : public detail::raw_ndarray_iter_base<1,1> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape,
                                char *dataA, const intptr_t *stridesA,
                                const char *dataB, const intptr_t *stridesB)
        : detail::raw_ndarray_iter_base<1,1>(ndim)
    {
        char *data[2] = {dataA, const_cast<char *>(dataB)};
        const intptr_t *strides[2] = {stridesA, stridesB};
        init(shape, data, strides);
    }
};

template<>
class raw_ndarray_iter<1,2> : public detail::raw_ndarray_iter_base<1,2> {
public:
    raw_ndarray_iter(int ndim, const intptr_t *shape,
                                char *dataA, const intptr_t *stridesA,
                                const char *dataB, const intptr_t *stridesB,
                                const char *dataC, const intptr_t *stridesC)
        : detail::raw_ndarray_iter_base<1,2>(ndim)
    {
        char *data[3] = {dataA, const_cast<char *>(dataB), const_cast<char *>(dataC)};
        const intptr_t *strides[3] = {stridesA, stridesB, stridesC};
        init(shape, data, strides);
    }
};

} // namespace dnd

#endif // _RAW_ITERATION_HPP_
