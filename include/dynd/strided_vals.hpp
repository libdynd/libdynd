//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRIDED_VALS_HPP_
#define _DYND__STRIDED_VALS_HPP_

#include <dynd/types/base_type.hpp>

namespace dynd { namespace nd {

template <typename T, int N>
class strided_vals {
    size_stride_t m_ss[N];
    const char *m_data_pointer;

public:
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

    void set_readonly_originptr(const char *data_pointer) {
        m_data_pointer = data_pointer;
    }

    T operator()(intptr_t i0, intptr_t i1) const {
        return *reinterpret_cast<const T *>(m_data_pointer + i0 * m_ss[0].stride + i1 * m_ss[1].stride);
    }
};

}} // namespace dynd::nd

#endif
