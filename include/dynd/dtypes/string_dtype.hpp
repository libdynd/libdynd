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

struct string_dtype_metadata {
    /** A reference to the memory block which contains the string's data */
    memory_block_data *blockref;
};

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

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *data, const char *metadata) const;

    void print_element(std::ostream& o, const char *data, const char *metadata) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    dtype get_canonical_dtype() const;

    void get_shape(int i, std::vector<intptr_t>& out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    void prepare_kernel_auxdata(const char *metadata, AuxDataBase *auxdata) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_dump(const char *metadata, std::ostream& o, const std::string& indent) const;
};

inline dtype make_string_dtype(string_encoding_t encoding) {
    return dtype(new string_dtype(encoding));
}

} // namespace dynd

#endif // _DYND__STRING_DTYPE_HPP_
