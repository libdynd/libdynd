//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDBYTES_DTYPE_HPP_
#define _DYND__FIXEDBYTES_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

class fixedbytes_dtype : public extended_dtype {
    intptr_t m_element_size, m_alignment;

public:
    fixedbytes_dtype(intptr_t element_size, intptr_t alignment);

    type_id_t get_type_id() const {
        return fixedbytes_type_id;
    }
    dtype_kind_t kind() const {
        return bytes_kind;
    }
    // Expose the storage traits here
    size_t alignment() const {
        return m_alignment;
    }
    size_t get_element_size() const {
        return m_element_size;
    }

    void print_element(std::ostream& o, const char *data, const char *metadata) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return pod_memory_management;
    }

    dtype apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(single_compare_kernel_instance& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    unary_specialization_kernel_instance& out_kernel) const;

    bool operator==(const extended_dtype& rhs) const;

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

/**
 * Creates a bytes<size, alignment> dtype, for representing
 * raw, uninterpreted bytes.
 */
inline dtype make_fixedbytes_dtype(intptr_t element_size, intptr_t alignment) {
    return dtype(new fixedbytes_dtype(element_size, alignment));
}

} // namespace dynd

#endif // _DYND__FIXEDBYTES_DTYPE_HPP_
