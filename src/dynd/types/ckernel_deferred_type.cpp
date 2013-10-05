//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/types/var_dim_type.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

ckernel_deferred_type::ckernel_deferred_type()
    : base_type(ckernel_deferred_type_id, custom_kind, sizeof(ckernel_deferred_type_data),
                    sizeof(void *),
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0)
{
}

ckernel_deferred_type::~ckernel_deferred_type()
{
}

void ckernel_deferred_type::print_data(std::ostream& o,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    const ckernel_deferred_type_data *ddd = reinterpret_cast<const ckernel_deferred_type_data *>(data);
    o << "<ckernel_deferred at " << (const void *)data;
    o << ", types [";
    for (size_t i = 0; i != ddd->data_types_size; ++i) {
        o << ndt::type(ddd->data_dynd_types[i], true);
        if (i != ddd->data_types_size - 1) {
            o << ", ";
        }
    }
    o << "]>";
}

void ckernel_deferred_type::print_type(std::ostream& o) const
{
    o << "ckernel_deferred";
}

bool ckernel_deferred_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == ckernel_deferred_type_id;
}

void ckernel_deferred_type::metadata_default_construct(char *DYND_UNUSED(metadata),
                intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void ckernel_deferred_type::metadata_copy_construct(char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void ckernel_deferred_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::metadata_finalize_buffers(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
}

void ckernel_deferred_type::data_destruct(const char *DYND_UNUSED(metadata), char *data) const
{
    const ckernel_deferred_type_data *d = reinterpret_cast<ckernel_deferred_type_data *>(data);
    if (d->data_ptr != NULL && d->free_func != NULL) {
        d->free_func(d->data_ptr);
    }
}

void ckernel_deferred_type::data_destruct_strided(const char *DYND_UNUSED(metadata), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const ckernel_deferred_type_data *d = reinterpret_cast<ckernel_deferred_type_data *>(data);
        if (d->data_ptr != NULL && d->free_func != NULL) {
            d->free_func(d->data_ptr);
        }
    }
}

size_t ckernel_deferred_type::make_assignment_kernel(
                ckernel_builder *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const ndt::type& dst_tp, const char *DYND_UNUSED(dst_metadata),
                const ndt::type& src_tp, const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    // Nothing can be assigned to/from ckernel_deferred
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw runtime_error(ss.str());
}
