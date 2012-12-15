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
    /**
     * A reference to the memory block which contains the string's data.
     * NOTE: This is identical to bytes_dtype_metadata, by design. Maybe
     *       both should become a typedef to a common class?
     */
    memory_block_data *blockref;
};

struct string_dtype_data {
    char *begin;
    char *end;
};

class string_dtype : public extended_string_dtype {
    string_encoding_t m_encoding;

public:
    string_dtype(string_encoding_t encoding);

    type_id_t get_type_id() const {
        return string_type_id;
    }
    dtype_kind_t get_kind() const {
        return string_kind;
    }
    // Expose the storage traits here
    size_t get_alignment() const {
        return sizeof(const char *);
    }
    size_t get_data_size() const {
        return sizeof(string_dtype_data);
    }

    string_encoding_t get_encoding() const {
        return m_encoding;
    }

    /** Alignment of the string data being pointed to. */
    size_t get_data_alignment() const {
        return string_encoding_char_size_table[m_encoding];
    }

    void get_string_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode, const std::string& utf8_str) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;
    intptr_t apply_linear_index(int nindices, const irange *indices, char *data, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt) const;

    dtype get_canonical_dtype() const;

    void get_shape(int i, std::vector<intptr_t>& out_shape) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;
};

inline dtype make_string_dtype(string_encoding_t encoding) {
    return dtype(new string_dtype(encoding));
}

} // namespace dynd

#endif // _DYND__STRING_DTYPE_HPP_
