//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

// Static instance of a void pointer to use as the storage of pointer dtypes
dtype pointer_dtype::m_void_pointer_dtype(new void_pointer_dtype(), false);


pointer_dtype::pointer_dtype(const dtype& target_dtype)
    : base_expression_dtype(pointer_type_id, expression_kind, sizeof(void *),
                    sizeof(void *),
                    inherited_flags(target_dtype.get_flags(), dtype_flag_zeroinit|dtype_flag_blockref),
                    sizeof(pointer_dtype_metadata) + target_dtype.get_metadata_size(),
                    target_dtype.get_undim()),
                    m_target_dtype(target_dtype)
{
    // I'm not 100% sure how blockref pointer dtypes should interact with
    // the computational subsystem, the details will have to shake out
    // when we want to actually do something with them.
    if (target_dtype.get_kind() == expression_kind && target_dtype.get_type_id() != pointer_type_id) {
        stringstream ss;
        ss << "A pointer dtype's target cannot be the expression dtype ";
        ss << target_dtype;
        throw runtime_error(ss.str());
    }
}

pointer_dtype::~pointer_dtype()
{
}

void pointer_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(metadata);
    const char *target_data = *reinterpret_cast<const char * const *>(data) + md->offset;
    m_target_dtype.print_data(o, metadata + sizeof(pointer_dtype_metadata), target_data);
}

void pointer_dtype::print_dtype(std::ostream& o) const
{
    o << "pointer<" << m_target_dtype << ">";
}

bool pointer_dtype::is_expression() const
{
    // Even though the pointer is an instance of an base_expression_dtype,
    // we'll only call it an expression if the target is.
    return m_target_dtype.is_expression();
}

bool pointer_dtype::is_unique_data_owner(const char *metadata) const
{
    const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(*metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             (md->blockref->m_type != pod_memory_block_type &&
              md->blockref->m_type != fixed_size_pod_memory_block_type))) {
        return false;
    }
    return true;
}

void pointer_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    dtype tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_target_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        out_transformed_dtype = dtype(new pointer_dtype(tmp_dtype), false);
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}


dtype pointer_dtype::get_canonical_dtype() const
{
    // The canonical version doesn't include the pointer
    return m_target_dtype;
}

dtype pointer_dtype::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        if (leading_dimension) {
            // Even with 0 indices, throw away the pointer when it's a leading dimension
            return m_target_dtype.apply_linear_index(0, NULL, current_i, root_dt, true);
        } else {
            return dtype(this, true);
        }
    } else {
        dtype dt = m_target_dtype.apply_linear_index(nindices, indices, current_i, root_dt, leading_dimension);
        if (leading_dimension) {
            // If it's a leading dimension, throw away the pointer
            return dt;
        } else if (dt == m_target_dtype) {
            return dtype(this, true);
        } else {
            return dtype(new pointer_dtype(dt), false);
        }
    }
}

intptr_t pointer_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (leading_dimension) {
        // If it's a leading dimension, we always throw away the pointer
        const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(metadata);
        *inout_data = *reinterpret_cast<char **>(*inout_data) + md->offset;
        if (*inout_dataref) {
            memory_block_decref(*inout_dataref);
        }
        *inout_dataref = md->blockref ? md->blockref : embedded_reference;
        memory_block_incref(*inout_dataref);
        if (m_target_dtype.is_builtin()) {
            return 0;
        } else {
            return m_target_dtype.extended()->apply_linear_index(nindices, indices,
                            metadata + sizeof(pointer_dtype_metadata),
                            result_dtype, out_metadata,
                            embedded_reference, current_i, root_dt,
                            true, inout_data, inout_dataref);
        }
    } else {
        const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(metadata);
        pointer_dtype_metadata *out_md = reinterpret_cast<pointer_dtype_metadata *>(out_metadata);
        // If there are no more indices, copy the rest verbatim
        out_md->blockref = md->blockref;
        memory_block_incref(out_md->blockref);
        out_md->offset = md->offset;
        if (!m_target_dtype.is_builtin()) {
            const pointer_dtype *pdt = static_cast<const pointer_dtype *>(result_dtype.extended());
            // The indexing may cause a change to the metadata offset
            out_md->offset += m_target_dtype.extended()->apply_linear_index(nindices, indices,
                            metadata + sizeof(pointer_dtype_metadata),
                            pdt->m_target_dtype, out_metadata + sizeof(pointer_dtype_metadata),
                            embedded_reference, current_i, root_dt,
                            false, NULL, NULL);
        }
        return 0;
    }
}

dtype pointer_dtype::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    // If metadata/data is provided, follow the pointer and call the target dtype's at_single
    if (inout_metadata) {
        const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(*inout_metadata);
        // Modify the metadata
        *inout_metadata += sizeof(pointer_dtype_metadata);
        // If requested, modify the data pointer
        if (inout_data) {
            *inout_data = *reinterpret_cast<const char * const *>(inout_data) + md->offset;
        }
    }
    return m_target_dtype.at_single(i0, inout_metadata, inout_data);
}

dtype pointer_dtype::get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return dtype(this, true);
    } else {
        *inout_metadata += sizeof(pointer_dtype_metadata);
        return m_target_dtype.get_dtype_at_dimension(inout_metadata, i, total_ndim);
    }
}

void pointer_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->get_shape(i, out_shape);
    }
}

void pointer_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    if (get_undim() > 0) {
        m_target_dtype.extended()->get_shape(i, out_shape, metadata + sizeof(pointer_dtype_metadata));
    }
}

axis_order_classification_t pointer_dtype::classify_axis_order(const char *metadata) const
{
    // Return the classification of the target dtype
    if (m_target_dtype.get_undim() > 1) {
        return m_target_dtype.extended()->classify_axis_order(
                        metadata + sizeof(pointer_dtype_metadata));
    } else {
        return axis_order_none;
    }
}

bool pointer_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        return ::is_lossless_assignment(m_target_dtype, src_dt);
    } else {
        return ::is_lossless_assignment(dst_dt, m_target_dtype);
    }
}

bool pointer_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != pointer_type_id) {
        return false;
    } else {
        const pointer_dtype *dt = static_cast<const pointer_dtype*>(&rhs);
        return m_target_dtype == dt->m_target_dtype;
    }
}

dtype pointer_dtype::with_replaced_storage_dtype(const dtype& /*replacement_dtype*/) const
{
    throw runtime_error("TODO: implement pointer_dtype::with_replaced_storage_dtype");
}

void pointer_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
{
    // Simply allocate a POD memory block
    // TODO: Will need a different kind of memory block if the data isn't POD.
    pointer_dtype_metadata *md = reinterpret_cast<pointer_dtype_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_default_construct(metadata + sizeof(pointer_dtype_metadata), ndim, shape);
    }
}

void pointer_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const pointer_dtype_metadata *src_md = reinterpret_cast<const pointer_dtype_metadata *>(src_metadata);
    pointer_dtype_metadata *dst_md = reinterpret_cast<pointer_dtype_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
    dst_md->offset = src_md->offset;
    // Copy the target metadata
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_copy_construct(dst_metadata + sizeof(pointer_dtype_metadata),
                        src_metadata + sizeof(pointer_dtype_metadata), embedded_reference);
    }
}

void pointer_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement pointer_dtype::metadata_reset_buffers");
}

void pointer_dtype::metadata_finalize_buffers(char *metadata) const
{
    pointer_dtype_metadata *md = reinterpret_cast<pointer_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void pointer_dtype::metadata_destruct(char *metadata) const
{
    pointer_dtype_metadata *md =
                    reinterpret_cast<pointer_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_destruct(
                        metadata + sizeof(pointer_dtype_metadata));
    }
}

void pointer_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const pointer_dtype_metadata *md = reinterpret_cast<const pointer_dtype_metadata *>(metadata);
    o << indent << "pointer metadata\n";
    o << indent << " offset: " << md->offset << "\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_debug_print(metadata + sizeof(pointer_dtype_metadata), o, indent + " ");
    }
}

static dtype property_get_target_dtype(const dtype& dt) {
    const pointer_dtype *pd = static_cast<const pointer_dtype *>(dt.extended());
    return pd->get_target_dtype();
}

static pair<string, gfunc::callable> dtype_properties[] = {
    pair<string, gfunc::callable>("target_dtype", gfunc::make_callable(&property_get_target_dtype, "self"))
};

void pointer_dtype::get_dynamic_dtype_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = dtype_properties;
    *out_count = sizeof(dtype_properties) / sizeof(dtype_properties[0]);
}

