//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/make_callable.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

// Static instance of a void pointer to use as the storage of pointer dtypes
ndt::type pointer_type::m_void_pointer_type(new void_pointer_type(), false);


pointer_type::pointer_type(const ndt::type& target_dtype)
    : base_expression_type(pointer_type_id, expression_kind, sizeof(void *),
                    sizeof(void *),
                    inherited_flags(target_dtype.get_flags(), type_flag_zeroinit|type_flag_blockref),
                    sizeof(pointer_type_metadata) + target_dtype.get_metadata_size(),
                    target_dtype.get_undim()),
                    m_target_dtype(target_dtype)
{
    // I'm not 100% sure how blockref pointer dtypes should interact with
    // the computational subsystem, the details will have to shake out
    // when we want to actually do something with them.
    if (target_dtype.get_kind() == expression_kind && target_dtype.get_type_id() != pointer_type_id) {
        stringstream ss;
        ss << "A pointer dtype's target cannot be the expression type ";
        ss << target_dtype;
        throw runtime_error(ss.str());
    }
}

pointer_type::~pointer_type()
{
}

void pointer_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
    const char *target_data = *reinterpret_cast<const char * const *>(data) + md->offset;
    m_target_dtype.print_data(o, metadata + sizeof(pointer_type_metadata), target_data);
}

void pointer_type::print_type(std::ostream& o) const
{
    o << "pointer<" << m_target_dtype << ">";
}

bool pointer_type::is_expression() const
{
    // Even though the pointer is an instance of an base_expression_type,
    // we'll only call it an expression if the target is.
    return m_target_dtype.is_expression();
}

bool pointer_type::is_unique_data_owner(const char *metadata) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(*metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             (md->blockref->m_type != pod_memory_block_type &&
              md->blockref->m_type != fixed_size_pod_memory_block_type))) {
        return false;
    }
    return true;
}

void pointer_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_dtype, bool& out_was_transformed) const
{
    ndt::type tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_target_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        out_transformed_dtype = ndt::type(new pointer_type(tmp_dtype), false);
        out_was_transformed = true;
    } else {
        out_transformed_dtype = ndt::type(this, true);
    }
}


ndt::type pointer_type::get_canonical_type() const
{
    // The canonical version doesn't include the pointer
    return m_target_dtype;
}

ndt::type pointer_type::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        if (leading_dimension) {
            // Even with 0 indices, throw away the pointer when it's a leading dimension
            return m_target_dtype.apply_linear_index(0, NULL, current_i, root_dt, true);
        } else {
            return ndt::type(this, true);
        }
    } else {
        ndt::type dt = m_target_dtype.apply_linear_index(nindices, indices, current_i, root_dt, leading_dimension);
        if (leading_dimension) {
            // If it's a leading dimension, throw away the pointer
            return dt;
        } else if (dt == m_target_dtype) {
            return ndt::type(this, true);
        } else {
            return ndt::type(new pointer_type(dt), false);
        }
    }
}

intptr_t pointer_type::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (leading_dimension) {
        // If it's a leading dimension, we always throw away the pointer
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
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
                            metadata + sizeof(pointer_type_metadata),
                            result_dtype, out_metadata,
                            embedded_reference, current_i, root_dt,
                            true, inout_data, inout_dataref);
        }
    } else {
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
        pointer_type_metadata *out_md = reinterpret_cast<pointer_type_metadata *>(out_metadata);
        // If there are no more indices, copy the rest verbatim
        out_md->blockref = md->blockref;
        memory_block_incref(out_md->blockref);
        out_md->offset = md->offset;
        if (!m_target_dtype.is_builtin()) {
            const pointer_type *pdt = static_cast<const pointer_type *>(result_dtype.extended());
            // The indexing may cause a change to the metadata offset
            out_md->offset += m_target_dtype.extended()->apply_linear_index(nindices, indices,
                            metadata + sizeof(pointer_type_metadata),
                            pdt->m_target_dtype, out_metadata + sizeof(pointer_type_metadata),
                            embedded_reference, current_i, root_dt,
                            false, NULL, NULL);
        }
        return 0;
    }
}

ndt::type pointer_type::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    // If metadata/data is provided, follow the pointer and call the target dtype's at_single
    if (inout_metadata) {
        const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(*inout_metadata);
        // Modify the metadata
        *inout_metadata += sizeof(pointer_type_metadata);
        // If requested, modify the data pointer
        if (inout_data) {
            *inout_data = *reinterpret_cast<const char * const *>(inout_data) + md->offset;
        }
    }
    return m_target_dtype.at_single(i0, inout_metadata, inout_data);
}

ndt::type pointer_type::get_type_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    } else {
        *inout_metadata += sizeof(pointer_type_metadata);
        return m_target_dtype.get_type_at_dimension(inout_metadata, i, total_ndim);
    }
}

void pointer_type::get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const
{
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->get_shape(ndim, i, out_shape,
                        metadata ? (metadata + sizeof(pointer_type_metadata)) : NULL);
    } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << m_target_dtype;
        throw runtime_error(ss.str());
    }
}

axis_order_classification_t pointer_type::classify_axis_order(const char *metadata) const
{
    // Return the classification of the target dtype
    if (m_target_dtype.get_undim() > 1) {
        return m_target_dtype.extended()->classify_axis_order(
                        metadata + sizeof(pointer_type_metadata));
    } else {
        return axis_order_none;
    }
}

bool pointer_type::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const
{
    if (dst_dt.extended() == this) {
        return ::is_lossless_assignment(m_target_dtype, src_dt);
    } else {
        return ::is_lossless_assignment(dst_dt, m_target_dtype);
    }
}

bool pointer_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != pointer_type_id) {
        return false;
    } else {
        const pointer_type *dt = static_cast<const pointer_type*>(&rhs);
        return m_target_dtype == dt->m_target_dtype;
    }
}

ndt::type pointer_type::with_replaced_storage_type(const ndt::type& /*replacement_type*/) const
{
    throw runtime_error("TODO: implement pointer_type::with_replaced_storage_type");
}

void pointer_type::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
{
    // Simply allocate a POD memory block
    // TODO: Will need a different kind of memory block if the data isn't POD.
    pointer_type_metadata *md = reinterpret_cast<pointer_type_metadata *>(metadata);
    md->blockref = make_pod_memory_block().release();
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_default_construct(metadata + sizeof(pointer_type_metadata), ndim, shape);
    }
}

void pointer_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    // Copy the blockref, switching it to the embedded_reference if necessary
    const pointer_type_metadata *src_md = reinterpret_cast<const pointer_type_metadata *>(src_metadata);
    pointer_type_metadata *dst_md = reinterpret_cast<pointer_type_metadata *>(dst_metadata);
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    if (dst_md->blockref) {
        memory_block_incref(dst_md->blockref);
    }
    dst_md->offset = src_md->offset;
    // Copy the target metadata
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_copy_construct(dst_metadata + sizeof(pointer_type_metadata),
                        src_metadata + sizeof(pointer_type_metadata), embedded_reference);
    }
}

void pointer_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement pointer_type::metadata_reset_buffers");
}

void pointer_type::metadata_finalize_buffers(char *metadata) const
{
    pointer_type_metadata *md = reinterpret_cast<pointer_type_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void pointer_type::metadata_destruct(char *metadata) const
{
    pointer_type_metadata *md =
                    reinterpret_cast<pointer_type_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_destruct(
                        metadata + sizeof(pointer_type_metadata));
    }
}

void pointer_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const pointer_type_metadata *md = reinterpret_cast<const pointer_type_metadata *>(metadata);
    o << indent << "pointer metadata\n";
    o << indent << " offset: " << md->offset << "\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
    if (!m_target_dtype.is_builtin()) {
        m_target_dtype.extended()->metadata_debug_print(metadata + sizeof(pointer_type_metadata), o, indent + " ");
    }
}

static ndt::type property_get_target_dtype(const ndt::type& dt) {
    const pointer_type *pd = static_cast<const pointer_type *>(dt.extended());
    return pd->get_target_dtype();
}

static pair<string, gfunc::callable> type_properties[] = {
    pair<string, gfunc::callable>("target_dtype", gfunc::make_callable(&property_get_target_dtype, "self"))
};

void pointer_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

