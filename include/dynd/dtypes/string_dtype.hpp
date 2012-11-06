//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string dtype uses memory_block references to store
// arbitrarily sized strings.
//
#ifndef _DYND__STRING_DTYPE_HPP_
#define _DYND__STRING_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

class string_dtype : public extended_string_dtype {
    string_encoding_t m_encoding;

public:
    string_dtype(string_encoding_t encoding);

    type_id_t type_id() const {
        return string_type_id;
    }
    dtype_kind_t kind() const {
        return string_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return sizeof(const char *);
    }
    size_t get_element_size() const {
        return 2 * sizeof(const char *);
    }

    string_encoding_t encoding() const {
        return m_encoding;
    }

    void print_element(std::ostream& o, const char *data, const char *metadata) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    void get_shape(int i, std::vector<intptr_t>& out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const {
        return 0;
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_dump(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }
};

inline dtype make_string_dtype(string_encoding_t encoding) {
    return dtype(new string_dtype(encoding));
}

} // namespace dynd

#endif // _DYND__STRING_DTYPE_HPP_
