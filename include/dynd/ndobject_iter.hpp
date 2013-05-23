//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _NDOBJECT_ITER_HPP_
#define _NDOBJECT_ITER_HPP_

#include <algorithm>

#include <dynd/ndobject.hpp>
#include <dynd/shape_tools.hpp>

namespace dynd {

namespace detail {
    /** A simple metaprogram to indicate whether a value is within the bounds or not */
    template<int V, int V_START, int V_END>
    struct is_value_within_bounds {
        enum {value = (V >= V_START) && V < V_END};
    };
}

template<int Nwrite, int Nread>
class ndobject_iter;

template<>
class ndobject_iter<1, 0> {
    intptr_t m_itersize;
    size_t m_iter_ndim;
    dimvector m_iterindex;
    dimvector m_itershape;
    char *m_data;
    const char *m_metadata;
    iterdata_common *m_iterdata;
    dtype m_array_dtype, m_uniform_dtype;

    inline void init(const dtype& dt0, const char *metadata0, char *data0)
    {
        m_array_dtype = dt0;
        m_iter_ndim = m_array_dtype.get_undim();
        m_itersize = 1;
        if (m_iter_ndim != 0) {
            m_iterindex.init(m_iter_ndim);
            memset(m_iterindex.get(), 0, sizeof(intptr_t) * m_iter_ndim);
            m_itershape.init(m_iter_ndim);
            m_array_dtype.extended()->get_shape(0, m_itershape.get(), metadata0);

            size_t iterdata_size = m_array_dtype.extended()->get_iterdata_size(m_iter_ndim);
            m_iterdata = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata) {
                throw std::bad_alloc();
            }
            m_metadata = metadata0;
            m_array_dtype.iterdata_construct(m_iterdata,
                            &m_metadata, m_iter_ndim, m_itershape.get(), m_uniform_dtype);
            m_data = m_iterdata->reset(m_iterdata, data0, m_iter_ndim);

            for (size_t i = 0, i_end = m_iter_ndim; i != i_end; ++i) {
                m_itersize *= m_itershape[i];
            }
        } else {
            m_iterdata = NULL;
            m_uniform_dtype = m_array_dtype;
            m_data = data0;
            m_metadata = metadata0;
        }
    }
public:
    ndobject_iter(const ndobject& op0) {
        init(op0.get_dtype(), op0.get_ndo_meta(), op0.get_readwrite_originptr());
    }

    ~ndobject_iter() {
        if (m_iterdata) {
            m_array_dtype.extended()->iterdata_destruct(m_iterdata, m_iter_ndim);
            free(m_iterdata);
        }
    }

    size_t itersize() const {
        return m_itersize;
    }

    bool empty() const {
        return m_itersize == 0;
    }

    bool next() {
        size_t i = m_iter_ndim;
        if (i != 0) {
            do {
                --i;
                if (++m_iterindex[i] != m_itershape[i]) {
                    m_data = m_iterdata->incr(m_iterdata, m_iter_ndim - i - 1);
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (i != 0);
        }

        return false;
    }

    char *data() {
        return m_data;
    }

    const char *metadata() {
        return m_metadata;
    }

    const dtype& get_uniform_dtype() const {
        return m_uniform_dtype;
    }
};

template<>
class ndobject_iter<0, 1> {
    intptr_t m_itersize;
    size_t m_iter_ndim;
    dimvector m_iterindex;
    dimvector m_itershape;
    const char *m_data;
    const char *m_metadata;
    iterdata_common *m_iterdata;
    dtype m_array_dtype, m_uniform_dtype;

    inline void init(const dtype& dt0, const char *metadata0, const char *data0)
    {
        m_array_dtype = dt0;
        m_iter_ndim = m_array_dtype.get_undim();
        m_itersize = 1;
        if (m_iter_ndim != 0) {
            m_iterindex.init(m_iter_ndim);
            memset(m_iterindex.get(), 0, sizeof(intptr_t) * m_iter_ndim);
            m_itershape.init(m_iter_ndim);
            m_array_dtype.extended()->get_shape(0, m_itershape.get(), metadata0);

            size_t iterdata_size = m_array_dtype.extended()->get_iterdata_size(m_iter_ndim);
            m_iterdata = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata) {
                throw std::bad_alloc();
            }
            m_metadata = metadata0;
            m_array_dtype.iterdata_construct(m_iterdata,
                            &m_metadata, m_iter_ndim, m_itershape.get(), m_uniform_dtype);
            m_data = m_iterdata->reset(m_iterdata, const_cast<char *>(data0), m_iter_ndim);

            for (size_t i = 0, i_end = m_iter_ndim; i != i_end; ++i) {
                m_itersize *= m_itershape[i];
            }
        } else {
            m_iterdata = NULL;
            m_uniform_dtype = m_array_dtype;
            m_data = data0;
            m_metadata = metadata0;
        }
    }
public:
    ndobject_iter(const dtype& dt0, const char *metadata0, const char *data0) {
        init(dt0, metadata0, data0);
    }

    ndobject_iter(const ndobject& op0) {
        init(op0.get_dtype(), op0.get_ndo_meta(), op0.get_readonly_originptr());
    }

    ~ndobject_iter() {
        if (m_iterdata) {
            m_array_dtype.extended()->iterdata_destruct(m_iterdata, m_iter_ndim);
            free(m_iterdata);
        }
    }

    size_t itersize() const {
        return m_itersize;
    }

    bool empty() const {
        return m_itersize == 0;
    }

    bool next() {
        size_t i = m_iter_ndim;
        if (i != 0) {
            do {
                --i;
                if (++m_iterindex[i] != m_itershape[i]) {
                    m_data = m_iterdata->incr(m_iterdata, m_iter_ndim - i - 1);
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (i != 0);
        }

        return false;
    }

    const char *data() {
        return m_data;
    }

    const char *metadata() {
        return m_metadata;
    }

    const dtype& get_uniform_dtype() const {
        return m_uniform_dtype;
    }
};

template<>
class ndobject_iter<1, 1> {
    intptr_t m_itersize;
    size_t m_iter_ndim[2];
    dimvector m_iterindex;
    dimvector m_itershape;
    char *m_data[2];
    const char *m_metadata[2];
    iterdata_common *m_iterdata[2];
    dtype m_array_dtype[2], m_uniform_dtype[2];

    inline void init(const dtype& dt0, const char *metadata0, char *data0,
                    const dtype& dt1, const char *metadata1, const char *data1)
    {
        m_array_dtype[0] = dt0;
        m_array_dtype[1] = dt1;
        m_itersize = 1;
        // The destination shape
        m_iter_ndim[0] = m_array_dtype[0].get_undim();
        m_itershape.init(m_iter_ndim[0]);
        if (m_iter_ndim[0] > 0) {
            m_array_dtype[0].extended()->get_shape(0, m_itershape.get(), metadata0);
        }
        // The source shape
        dimvector src_shape;
        m_iter_ndim[1] = m_array_dtype[1].get_undim();
        src_shape.init(m_iter_ndim[1]);
        if (m_iter_ndim[1] > 0) {
            m_array_dtype[1].extended()->get_shape(0, src_shape.get(), metadata1);
        }
        // Check that the source shape broadcasts ok
        if (!shape_can_broadcast(m_iter_ndim[0], m_itershape.get(),
                        m_iter_ndim[1], src_shape.get())) {
            throw broadcast_error(m_iter_ndim[0], m_itershape.get(),
                            m_iter_ndim[1], src_shape.get());
        }
        // Allocate and initialize the iterdata
        if (m_iter_ndim[0] != 0) {
            m_iterindex.init(m_iter_ndim[0]);
            memset(m_iterindex.get(), 0, sizeof(intptr_t) * m_iter_ndim[0]);
            // The destination iterdata
            size_t iterdata_size = m_array_dtype[0].get_iterdata_size(m_iter_ndim[0]);
            m_iterdata[0] = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata[0]) {
                throw std::bad_alloc();
            }
            m_metadata[0] = metadata0;
            m_array_dtype[0].iterdata_construct(m_iterdata[0],
                            &m_metadata[0], m_iter_ndim[0], m_itershape.get(), m_uniform_dtype[0]);
            m_data[0] = m_iterdata[0]->reset(m_iterdata[0], data0, m_iter_ndim[0]);
            // The source iterdata
            iterdata_size = m_array_dtype[1].get_broadcasted_iterdata_size(m_iter_ndim[1]);
            m_iterdata[1] = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata[1]) {
                throw std::bad_alloc();
            }
            m_metadata[1] = metadata1;
            m_array_dtype[1].broadcasted_iterdata_construct(m_iterdata[1],
                            &m_metadata[1], m_iter_ndim[1],
                            m_itershape.get() + (m_iter_ndim[0] - m_iter_ndim[1]), m_uniform_dtype[1]);
            m_data[1] = m_iterdata[1]->reset(m_iterdata[1], const_cast<char *>(data1), m_iter_ndim[0]);

            for (size_t i = 0, i_end = m_iter_ndim[0]; i != i_end; ++i) {
                m_itersize *= m_itershape[i];
            }
        } else {
            m_iterdata[0] = NULL;
            m_iterdata[1] = NULL;
            m_uniform_dtype[0] = m_array_dtype[0];
            m_uniform_dtype[1] = m_array_dtype[1];
            m_data[0] = data0;
            m_data[1] = const_cast<char *>(data1);
            m_metadata[0] = metadata0;
            m_metadata[1] = metadata1;
        }
    }
public:
    ndobject_iter(const dtype& dt0, const char *metadata0, char *data0,
                    const dtype& dt1, const char *metadata1, const char *data1) {
        init(dt0, metadata0, data0, dt1, metadata1, data1);
    }
    ndobject_iter(const ndobject& op0, const ndobject& op1) {
        init(op0.get_dtype(), op0.get_ndo_meta(), op0.get_readwrite_originptr(),
                        op1.get_dtype(), op1.get_ndo_meta(), op1.get_readonly_originptr());
    }

    ~ndobject_iter() {
        if (m_iterdata[0]) {
            m_array_dtype[0].iterdata_destruct(m_iterdata[0], m_iter_ndim[0]);
            free(m_iterdata[0]);
        }
        if (m_iterdata[1]) {
            m_array_dtype[1].iterdata_destruct(m_iterdata[1], m_iter_ndim[1]);
            free(m_iterdata[1]);
        }
    }

    size_t itersize() const {
        return m_itersize;
    }

    bool empty() const {
        return m_itersize == 0;
    }

    bool next() {
        size_t i = m_iter_ndim[0];
        if (i != 0) {
            do {
                --i;
                if (++m_iterindex[i] != m_itershape[i]) {
                    m_data[0] = m_iterdata[0]->incr(m_iterdata[0], m_iter_ndim[0] - i - 1);
                    m_data[1] = m_iterdata[1]->incr(m_iterdata[1], m_iter_ndim[0] - i - 1);
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (i != 0);
        }

        return false;
    }

    /**
     * Provide non-const access to the 'write' operands.
     */
    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 1>::value, char *>::type data() {
        return m_data[K];
    }

    /**
     * Provide const access to all the operands.
     */
    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const char *>::type data() const {
        return m_data[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const char *>::type metadata() const {
        return m_metadata[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const dtype&>::type get_uniform_dtype() const {
        return m_uniform_dtype[K];
    }
};

template<>
class ndobject_iter<0, 2> {
    intptr_t m_itersize;
    size_t m_iter_ndim;
    dimvector m_iterindex;
    dimvector m_itershape;
    char *m_data[2];
    const char *m_metadata[2];
    iterdata_common *m_iterdata[2];
    dtype m_array_dtype[2], m_uniform_dtype[2];
public:
    ndobject_iter(const ndobject& op0, const ndobject& op1) {
        ndobject ops[2] = {op0, op1};
        m_array_dtype[0] = op0.get_dtype();
        m_array_dtype[1] = op1.get_dtype();
        m_itersize = 1;
        shortvector<int> axis_perm; // TODO: Use this to affect the iteration order
        broadcast_input_shapes(2, ops, m_iter_ndim, m_itershape, axis_perm);
        // Allocate and initialize the iterdata
        if (m_iter_ndim != 0) {
            m_iterindex.init(m_iter_ndim);
            memset(m_iterindex.get(), 0, sizeof(intptr_t) * m_iter_ndim);
            // The op iterdata
            for (int i = 0; i < 2; ++i) {
                size_t iter_ndim_i = m_array_dtype[i].get_undim();
                size_t iterdata_size = m_array_dtype[i].get_broadcasted_iterdata_size(iter_ndim_i);
                m_iterdata[i] = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
                if (!m_iterdata[i]) {
                    throw std::bad_alloc();
                }
                m_metadata[i] = ops[i].get_ndo_meta();
                m_array_dtype[i].broadcasted_iterdata_construct(m_iterdata[i],
                                &m_metadata[i], iter_ndim_i,
                                m_itershape.get() + (m_iter_ndim - iter_ndim_i), m_uniform_dtype[i]);
                m_data[i] = m_iterdata[i]->reset(m_iterdata[i], ops[i].get_ndo()->m_data_pointer, m_iter_ndim);
            }

            for (size_t i = 0, i_end = m_iter_ndim; i != i_end; ++i) {
                m_itersize *= m_itershape[i];
            }
        } else {
            for (size_t i = 0; i < 2; ++i) {
                m_iterdata[i] = NULL;
                m_uniform_dtype[i] = m_array_dtype[i];
                m_data[i] = ops[i].get_ndo()->m_data_pointer;
                m_metadata[i] = ops[i].get_ndo_meta();
            }
        }
    }

    ~ndobject_iter() {
        for (size_t i = 0; i < 2; ++i) {
            if (m_iterdata[i]) {
                m_array_dtype[i].iterdata_destruct(m_iterdata[i], m_array_dtype[i].get_undim());
                free(m_iterdata[i]);
            }
        }
    }

    size_t itersize() const {
        return m_itersize;
    }

    bool empty() const {
        return m_itersize == 0;
    }

    bool next() {
        size_t i = m_iter_ndim;
        if (i != 0) {
            do {
                --i;
                if (++m_iterindex[i] != m_itershape[i]) {
                    for (size_t j = 0; j < 2; ++j) {
                        m_data[j] = m_iterdata[j]->incr(m_iterdata[j], m_iter_ndim - i - 1);
                    }
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (i != 0);
        }

        return false;
    }

    /**
     * Provide const access to all the operands.
     */
    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const char *>::type data() const {
        return m_data[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const char *>::type metadata() const {
        return m_metadata[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 2>::value, const dtype&>::type get_uniform_dtype() const {
        return m_uniform_dtype[K];
    }
};

template<>
class ndobject_iter<1, 3> {
    intptr_t m_itersize;
    size_t m_iter_ndim[4];
    dimvector m_iterindex;
    dimvector m_itershape;
    char *m_data[4];
    const char *m_metadata[4];
    iterdata_common *m_iterdata[4];
    dtype m_array_dtype[4], m_uniform_dtype[4];
public:
    // Constructor which creates the output based on the input's broadcast shape
    ndobject_iter(const dtype& op0_dtype, ndobject& out_op0, const ndobject& op1, const ndobject& op2, const ndobject& op3) {
        create_broadcast_result(op0_dtype, op1, op2, op3, out_op0, m_iter_ndim[0], m_itershape);
        ndobject ops[4] = {out_op0, op1, op2, op3};
        m_array_dtype[0] = out_op0.get_dtype();
        m_array_dtype[1] = op1.get_dtype();
        m_array_dtype[2] = op2.get_dtype();
        m_array_dtype[3] = op3.get_dtype();
        m_itersize = 1;
        m_iter_ndim[1] = m_array_dtype[1].get_undim();
        m_iter_ndim[2] = m_array_dtype[2].get_undim();
        m_iter_ndim[3] = m_array_dtype[3].get_undim();
        // Allocate and initialize the iterdata
        if (m_iter_ndim[0] != 0) {
            m_iterindex.init(m_iter_ndim[0]);
            memset(m_iterindex.get(), 0, sizeof(intptr_t) * m_iter_ndim[0]);
            // The destination iterdata
            size_t iterdata_size = m_array_dtype[0].get_iterdata_size(m_iter_ndim[0]);
            m_iterdata[0] = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata[0]) {
                throw std::bad_alloc();
            }
            m_metadata[0] = out_op0.get_ndo_meta();
            m_array_dtype[0].iterdata_construct(m_iterdata[0],
                            &m_metadata[0], m_iter_ndim[0], m_itershape.get(), m_uniform_dtype[0]);
            m_data[0] = m_iterdata[0]->reset(m_iterdata[0], out_op0.get_readwrite_originptr(), m_iter_ndim[0]);
            // The op iterdata
            for (int i = 1; i < 4; ++i) {
                iterdata_size = m_array_dtype[i].get_broadcasted_iterdata_size(m_iter_ndim[i]);
                m_iterdata[i] = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
                if (!m_iterdata[i]) {
                    throw std::bad_alloc();
                }
                m_metadata[i] = ops[i].get_ndo_meta();
                m_array_dtype[i].broadcasted_iterdata_construct(m_iterdata[i],
                                &m_metadata[i], m_iter_ndim[i],
                                m_itershape.get() + (m_iter_ndim[0] - m_iter_ndim[i]), m_uniform_dtype[i]);
                m_data[i] = m_iterdata[i]->reset(m_iterdata[i], ops[i].get_ndo()->m_data_pointer, m_iter_ndim[0]);
            }

            for (size_t i = 0, i_end = m_iter_ndim[0]; i != i_end; ++i) {
                m_itersize *= m_itershape[i];
            }
        } else {
            for (size_t i = 0; i < 4; ++i) {
                m_iterdata[i] = NULL;
                m_uniform_dtype[i] = m_array_dtype[i];
                m_data[i] = ops[i].get_ndo()->m_data_pointer;
                m_metadata[i] = ops[i].get_ndo_meta();
            }
        }
    }

    ~ndobject_iter() {
        for (size_t i = 0; i < 4; ++i) {
            if (m_iterdata[i]) {
                m_array_dtype[i].iterdata_destruct(m_iterdata[i], m_iter_ndim[i]);
                free(m_iterdata[i]);
            }
        }
    }

    size_t itersize() const {
        return m_itersize;
    }

    bool empty() const {
        return m_itersize == 0;
    }

    bool next() {
        size_t i = m_iter_ndim[0];
        if (i != 0) {
            do {
                --i;
                if (++m_iterindex[i] != m_itershape[i]) {
                    for (size_t j = 0; j < 4; ++j) {
                        m_data[j] = m_iterdata[j]->incr(m_iterdata[j], m_iter_ndim[0] - i - 1);
                    }
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (i != 0);
        }

        return false;
    }

    /**
     * Provide non-const access to the 'write' operands.
     */
    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 1>::value, char *>::type data() {
        return m_data[K];
    }

    /**
     * Provide const access to all the operands.
     */
    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 4>::value, const char *>::type data() const {
        return m_data[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 4>::value, const char *>::type metadata() const {
        return m_metadata[K];
    }

    template<int K>
    inline typename enable_if<detail::is_value_within_bounds<K, 0, 4>::value, const dtype&>::type get_uniform_dtype() const {
        return m_uniform_dtype[K];
    }
};

} // namespace dynd

#endif // _NDOBJECT_ITER_HPP_
