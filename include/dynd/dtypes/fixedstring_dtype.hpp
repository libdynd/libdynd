//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The fixedstring dtype represents a string with
// a particular encoding, stored in a fixed-size
// buffer.
//
#ifndef _DYND__FIXEDSTRING_DTYPE_HPP_
#define _DYND__FIXEDSTRING_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

class fixedstring_dtype : public base_string_dtype {
    intptr_t m_stringsize;
    string_encoding_t m_encoding;

public:
    fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize);

    virtual ~fixedstring_dtype();

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *dst, assign_error_mode errmode, const std::string& utf8_str) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    dtype get_canonical_dtype() const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_default_construct(char *DYND_UNUSED(metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }
};

inline dtype make_fixedstring_dtype(string_encoding_t encoding, intptr_t stringsize) {
    return dtype(new fixedstring_dtype(encoding, stringsize), false);
}

} // namespace dynd

#endif // _DYND__FIXEDSTRING_DTYPE_HPP_
