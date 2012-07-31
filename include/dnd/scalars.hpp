//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _SCALARS_HPP_
#define _SCALARS_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>

#ifdef __GNUC__
#pragma GCC diagnostic push
// The -Weffc++ flag warns about member variables not being initialized by
// the member initialization list. In this case, I didn't see a nice way
// to do this (maybe making {m_data, m_allocated} into a separate class...).
//
// NOTE: The documentation says this is only for g++ 4.6.0 and up.
#pragma GCC diagnostic ignored "-Weffc++"
#endif

namespace dnd {

/**
 * This class makes a copy of an input scalar if necessary, otherwise
 * passes the pointer through. The data provided is aligned and
 * with the dtype requested.
 */
class scalar_copied_if_necessary {
    int64_t m_shortbuffer[2]; // TODO: What about 16-byte alignment like for SSE vectors?
    const char *m_data;
    bool m_allocated;


    char *allocate_data(uintptr_t size) {
        char *result;
        if (size <= sizeof(m_shortbuffer)) {
            m_data = result = reinterpret_cast<char *>(&m_shortbuffer[0]);
            m_allocated = false;
        } else {
            m_data = result = new char[size];
            m_allocated = true;
        }

        return result;
    }

    // Non-copyable
    scalar_copied_if_necessary(const scalar_copied_if_necessary&);
    scalar_copied_if_necessary& operator=(const scalar_copied_if_necessary&);
public:
    /**
     * Constructs the temporary scalar object.
     *
     * @param dst_dtype   The dtype the scalar should have.
     * @param src_dtype   The dtype the input scalar has.
     * @param src_data    The data for the input scalar.
     * @param errmode     What kind of error checking to do when converting data types.
     */
    scalar_copied_if_necessary(const dtype& dst_dtype, const dtype& src_dtype, const char *src_data,
                                    assign_error_mode errmode = assign_error_fractional,
                                    const eval::eval_context *ectx = &eval::default_eval_context) {

        if (dst_dtype == src_dtype) {
            // Pass through the aligned data pointer
            m_data = reinterpret_cast<const char *>(src_data);
            m_allocated = false;
        } else {
            // Make a converted copy into an aligned buffer
            char *tmp = allocate_data(dst_dtype.element_size());
            dtype_assign(dst_dtype, tmp, src_dtype, src_data, errmode, ectx);
        }
    }

    ~scalar_copied_if_necessary() {
        if (m_allocated) {
            delete[] m_data;
        }
    }

    /** Gets the scalar data pointer */
    const char *data() const {
        return m_data;
    }
};

} // namespace dnd

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // _SCALARS_HPP_
