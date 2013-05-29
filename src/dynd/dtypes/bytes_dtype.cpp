//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/bytes_assignment_kernels.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

bytes_dtype::bytes_dtype(size_t alignment)
    : base_bytes_dtype(bytes_type_id, bytes_kind, sizeof(bytes_dtype_data),
                    sizeof(const char *), dtype_flag_scalar|dtype_flag_zeroinit|dtype_flag_blockref,
                    sizeof(bytes_dtype_metadata)), m_alignment(alignment)
{
    if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8 && alignment != 16) {
        std::stringstream ss;
        ss << "Cannot make a bytes dtype with alignment " << alignment << ", it must be a small power of two";
        throw std::runtime_error(ss.str());
    }
}

bytes_dtype::~bytes_dtype()
{
}

void bytes_dtype::get_bytes_range(const char **out_begin, const char**out_end,
                const char *DYND_UNUSED(metadata), const char *data) const
{
    *out_begin = reinterpret_cast<const bytes_dtype_data *>(data)->begin;
    *out_end = reinterpret_cast<const bytes_dtype_data *>(data)->end;
}

void bytes_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    const char *begin = reinterpret_cast<const char * const *>(data)[0];
    const char *end = reinterpret_cast<const char * const *>(data)[1];

    // Print as hexadecimal
    o << "0x";
    hexadecimal_print(o, begin, end - begin);
}

void bytes_dtype::print_dtype(std::ostream& o) const
{
    o << "bytes";
    if (m_alignment != 1) {
        o << "<align=" << m_alignment << ">";
    }
}

bool bytes_dtype::is_unique_data_owner(const char *metadata) const
{
    const bytes_dtype_metadata *md = reinterpret_cast<const bytes_dtype_metadata *>(*metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             md->blockref->m_type != pod_memory_block_type)) {
        return false;
    }
    return true;
}

dtype bytes_dtype::get_canonical_dtype() const
{
    return dtype(this, true);
}


void bytes_dtype::get_shape(size_t ndim, size_t i,
                intptr_t *out_shape, const char *DYND_UNUSED(metadata)) const
{
    out_shape[i] = -1;
    if (i+1 < ndim) {
        stringstream ss;
        ss << "requested too many dimensions from type " << dtype(this, true);
        throw runtime_error(ss.str());
    }
}

bool bytes_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.get_kind() == bytes_kind) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

size_t bytes_dtype::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        switch (src_dt.get_type_id()) {
            case bytes_type_id: {
                return make_blockref_bytes_assignment_kernel(out, offset_out,
                                get_alignment(), dst_metadata,
                                src_dt.get_alignment(), src_metadata,
                                kernreq, ectx);
            }
            case fixedbytes_type_id: {
                return make_fixedbytes_to_blockref_bytes_assignment_kernel(out, offset_out,
                                get_alignment(), dst_metadata,
                                src_dt.get_data_size(), src_dt.get_alignment(),
                                kernreq, ectx);
            }
            default: {
                if (!src_dt.is_builtin()) {
                    src_dt.extended()->make_assignment_kernel(out, offset_out,
                                    dst_dt, dst_metadata,
                                    src_dt, src_metadata,
                                    kernreq, errmode, ectx);
                }
                break;
            }
        }
    }

    stringstream ss;
    ss << "Cannot assign from " << src_dt << " to " << dst_dt;
    throw runtime_error(ss.str());
}


bool bytes_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != bytes_type_id) {
        return false;
    } else {
        const bytes_dtype *dt = static_cast<const bytes_dtype*>(&rhs);
        return m_alignment == dt->m_alignment;
    }
}

void bytes_dtype::metadata_default_construct(char *metadata, size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
    // Simply allocate a POD memory block
    bytes_dtype_metadata *md = reinterpret_cast<bytes_dtype_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
}

void bytes_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const bytes_dtype_metadata *src_md = reinterpret_cast<const bytes_dtype_metadata *>(src_metadata);
    bytes_dtype_metadata *dst_md = reinterpret_cast<bytes_dtype_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
}

void bytes_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement bytes_dtype::metadata_reset_buffers");
}

void bytes_dtype::metadata_finalize_buffers(char *metadata) const
{
    bytes_dtype_metadata *md = reinterpret_cast<bytes_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void bytes_dtype::metadata_destruct(char *metadata) const
{
    bytes_dtype_metadata *md = reinterpret_cast<bytes_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
}

void bytes_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const bytes_dtype_metadata *md = reinterpret_cast<const bytes_dtype_metadata *>(metadata);
    o << indent << "bytes metadata\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
}
