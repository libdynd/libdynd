//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _NDOBJECT_ITER_HPP_
#define _NDOBJECT_ITER_HPP_

#include <algorithm>

#include <dynd/ndobject.hpp>
#include <dynd/shape_tools.hpp>

namespace dynd {

template<int Nwrite, int Nread>
class ndobject_iter;

template<>
class ndobject_iter<1, 0> {
    intptr_t m_itersize;
    // TODO: More efficient representation of the shape/iteration
    std::vector<intptr_t> m_iterindex;
    std::vector<intptr_t> m_itershape;
    char *m_data;
    iterdata_common *m_iterdata;
    dtype m_array_dtype, m_uniform_dtype;
public:
    ndobject_iter(const ndobject& op0) {
        m_array_dtype = op0.get_dtype();
        if (m_array_dtype.extended()) {
            m_array_dtype.extended()->get_shape(0, m_itershape, op0.get_ndo()->m_data_pointer, op0.get_ndo_meta());
        }
        if (m_itershape.empty()) {
            m_iterdata = NULL;
            m_uniform_dtype = m_array_dtype;
            m_data = op0.get_ndo()->m_data_pointer;
        } else {
            size_t iterdata_size = m_array_dtype.extended()->get_iterdata_size();
            m_iterdata = reinterpret_cast<iterdata_common *>(malloc(iterdata_size));
            if (!m_iterdata) {
                throw std::bad_alloc("memory allocation error creating dynd ndobject iterator");
            }
            m_array_dtype.extended()->iterdata_construct(m_iterdata,
                            op0.get_ndo_meta(), m_itershape.size(), &m_itershape[0], m_uniform_dtype);
            m_data = m_iterdata->reset(m_iterdata, op0.get_ndo()->m_data_pointer, m_itershape.size());
        }
        m_iterindex.resize(m_itershape.size());
        m_itersize = 1;
        for (size_t i = 0, i_end = m_itershape.size(); i != i_end; ++i) {
            m_itersize *= m_itershape[i];
        }
    }

    ~ndobject_iter() {
        if (m_iterdata) {
            m_array_dtype.extended()->iterdata_destruct(m_iterdata, m_itershape.size());
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
        size_t i = 0, i_end = m_itershape.size();
        if (i_end != 0) {
            do {
                if (++m_iterindex[i] != m_itershape[i]) {
                    m_data = m_iterdata->incr(m_iterdata, i);
                    return true;
                } else {
                    m_iterindex[i] = 0;
                }
            } while (++i != i_end);
        }

        return false;
    }

    char *data() {
        return m_data;
    }
};

} // namespace dynd

#endif // _NDOBJECT_ITER_HPP_
