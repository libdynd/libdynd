//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FIXEDBYTES_TYPE_HPP_
#define _DYND__FIXEDBYTES_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/base_bytes_type.hpp>
#include <dynd/dtypes/view_type.hpp>

namespace dynd {

class fixedbytes_type : public base_bytes_type {
public:
    fixedbytes_type(intptr_t element_size, intptr_t alignment);

    virtual ~fixedbytes_type();

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    void get_bytes_range(const char **out_begin, const char**out_end, const char *metadata, const char *data) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *DYND_UNUSED(metadata), size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const ndt::type& dst_dt, const char *dst_metadata,
                    const ndt::type& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

/**
 * Creates a bytes<size, alignment> dtype, for representing
 * raw, uninterpreted bytes.
 */
inline ndt::type make_fixedbytes_type(intptr_t element_size, intptr_t alignment) {
    return ndt::type(new fixedbytes_type(element_size, alignment), false);
}

} // namespace dynd

#endif // _DYND__FIXEDBYTES_TYPE_HPP_
