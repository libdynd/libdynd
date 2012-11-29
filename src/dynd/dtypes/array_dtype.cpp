//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/array_dtype.hpp>
#include <dynd/kernels/array_assignment_kernels.hpp>
#include <dynd/shape_tools.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

dynd::array_dtype::array_dtype(const dtype& element_dtype)
    : m_element_dtype(element_dtype)
{
}

void dynd::array_dtype::print_element(std::ostream& o, const char *data, const char *metadata) const
{
    const char *begin = reinterpret_cast<const char * const *>(data)[0];
    const char *end = reinterpret_cast<const char * const *>(data)[1];

    o << "[";
    while (begin < end) {
        m_element_dtype.print_element(o, begin, metadata);
        begin += m_element_dtype.get_element_size();
        if (begin < end) {
            o << ", ";
        }
    }
    o << "]";
}

void dynd::array_dtype::print_dtype(std::ostream& o) const {

    o << "array<" << m_element_dtype << ">";

}

dtype dynd::array_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            return m_element_dtype;
        } else {
            return dtype(this, true);
        }
    } else {
        if (indices->step() == 0) {
            return m_element_dtype.apply_linear_index(nindices-1, indices+1, current_i+1, root_dt);
        } else {
            return dtype(new array_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1, current_i+1, root_dt)));
        }
    }
}

void dynd::array_dtype::get_shape(int i, intptr_t *out_shape) const
{
    // Adjust the current shape
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (m_element_dtype.extended()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape);
    }
}

bool dynd::array_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (dst_dt.get_type_id() == array_type_id) {
            const array_dtype *src_esd = static_cast<const array_dtype*>(src_dt.extended());
            return ::dynd::is_lossless_assignment(m_element_dtype, src_esd->m_element_dtype);
        } else {
            return ::dynd::is_lossless_assignment(m_element_dtype, src_dt);
        }
    } else {
        return false;
    }
}

void dynd::array_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const {
    throw std::runtime_error("array_dtype::get_single_compare_kernel not supported yet");
}

void dynd::array_dtype::get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                assign_error_mode errmode,
                unary_specialization_kernel_instance& out_kernel) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case array_type_id: {
                const array_dtype *src_fs = static_cast<const array_dtype *>(src_dt.extended());
                get_blockref_array_assignment_kernel(m_element_dtype, src_fs->m_element_dtype,
                                        errmode, out_kernel);
                break;
            }
            default: {
                src_dt.extended()->get_dtype_assignment_kernel(dst_dt, src_dt, errmode, out_kernel);
                break;
            }
        }
    } else {
        throw runtime_error("conversions from array to non-array are not implemented");
    }
}


bool dynd::array_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != array_type_id) {
        return false;
    } else {
        const array_dtype *dt = static_cast<const array_dtype*>(&rhs);
        return m_element_dtype == dt->m_element_dtype;
    }
}
