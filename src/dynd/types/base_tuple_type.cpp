//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shortvector.hpp>

using namespace std;
using namespace dynd;


base_tuple_type::~base_tuple_type() {
}

size_t base_tuple_type::get_default_data_size(intptr_t ndim, const intptr_t *shape) const
{
    size_t field_count = get_field_count();
    const ndt::type *field_types = get_field_types();
    // Default layout is to match the field order - could reorder the elements for more efficient packing
    size_t s = 0;
    for (size_t i = 0; i != field_count; ++i) {
        s = inc_to_alignment(s, field_types[i].get_data_alignment());
        if (!field_types[i].is_builtin()) {
            s += field_types[i].extended()->get_default_data_size(ndim, shape);
        } else {
            s += field_types[i].get_data_size();
        }
    }
    s = inc_to_alignment(s, m_members.data_alignment);
    return s;
}

void base_tuple_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *metadata, const char *DYND_UNUSED(data)) const
{
    out_shape[i] = m_field_count;
    if (i < ndim-1) {
        const ndt::type *field_types = get_field_types();
        const size_t *metadata_offsets = get_metadata_offsets();
        dimvector tmpshape(ndim);
        // Accumulate the shape from all the field shapes
        for (size_t fi = 0; fi != m_field_count; ++fi) {
            const ndt::type& ft = field_types[i];
            if (!ft.is_builtin()) {
                ft.extended()->get_shape(ndim, i+1, tmpshape.get(),
                                metadata ? (metadata + metadata_offsets[fi]) : NULL,
                                NULL);
            } else {
                stringstream ss;
                ss << "requested too many dimensions from type " << ft;
                throw runtime_error(ss.str());
            }
            if (fi == 0) {
                // Copy the shape from the first field
                memcpy(out_shape + i + 1, tmpshape.get() + i + 1, (ndim - i - 1) * sizeof(intptr_t));
            } else {
                // Merge the shape from the rest
                for (intptr_t k = i + 1; k < ndim; ++k) {
                    // If we see different sizes, make the output -1
                    if (out_shape[k] != -1 && out_shape[k] != tmpshape[k]) {
                        out_shape[k] = -1;
                    }
                }
            }
        }
    }
}

void base_tuple_type::data_destruct(const char *metadata, char *data) const
{
    const ndt::type *field_types = get_field_types();
    const size_t *metadata_offsets = get_metadata_offsets();
    const size_t *data_offsets = get_data_offsets(metadata);
    size_t field_count = get_field_count();
    for (size_t i = 0; i != field_count; ++i) {
        const ndt::type& dt = field_types[i];
        if (dt.get_flags()&type_flag_destructor) {
            dt.extended()->data_destruct(
                            metadata + metadata_offsets[i],
                            data + data_offsets[i]);
        }
    }
}

void base_tuple_type::data_destruct_strided(const char *metadata, char *data,
                intptr_t stride, size_t count) const
{
    const ndt::type *field_types = get_field_types();
    const size_t *metadata_offsets = get_metadata_offsets();
    const size_t *data_offsets = get_data_offsets(metadata);
    size_t field_count = get_field_count();
    // Destruct all the fields a chunk at a time, in an
    // attempt to have some kind of locality
    while (count > 0) {
        size_t chunk_size = min(count, DYND_BUFFER_CHUNK_SIZE);
        for (size_t i = 0; i != field_count; ++i) {
            const ndt::type& dt = field_types[i];
            if (dt.get_flags()&type_flag_destructor) {
                dt.extended()->data_destruct_strided(
                                metadata + metadata_offsets[i],
                                data + data_offsets[i],
                                stride, chunk_size);
            }
        }
        data += stride * chunk_size;
        count -= chunk_size;
    }
}


 
