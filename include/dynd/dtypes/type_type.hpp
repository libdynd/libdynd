//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_TYPE_HPP_
#define _DYND__DTYPE_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtype_assign.hpp>

namespace dynd {

struct type_type_data {
    const base_type *dt;
};

/**
 * A dynd type whose ndobject instances themselves contain
 * dynd types.
 */
class type_type : public base_type {
public:
    type_type();

    virtual ~type_type();

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const ndt::type& dst_dt, const char *dst_metadata,
                    const ndt::type& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

namespace ndt {
    inline ndt::type make_type() {
        return ndt::type(new type_type(), false);
    }
} // namespace ndt

} // namespace dynd

#endif // _DYND__DTYPE_TYPE_HPP_
