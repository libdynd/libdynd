//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TYPE_TYPE_HPP_
#define _DYND__TYPE_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>

namespace dynd {

struct type_type_data {
    const base_type *tp;
};

/**
 * A dynd type whose nd::array instances themselves contain
 * dynd types.
 */
class type_type : public base_type {
public:
    type_type();

    virtual ~type_type();

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_type(std::ostream& o) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;

    void data_destruct(const char *metadata, char *data) const;
    void data_destruct_strided(const char *metadata, char *data,
                    intptr_t stride, size_t count) const;

    size_t make_assignment_kernel(
                    ckernel_builder *out, size_t offset_out,
                    const ndt::type& dst_tp, const char *dst_metadata,
                    const ndt::type& src_tp, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;
};

namespace ndt {
    ndt::type make_type();
} // namespace ndt

} // namespace dynd

#endif // _DYND__TYPE_TYPE_HPP_
