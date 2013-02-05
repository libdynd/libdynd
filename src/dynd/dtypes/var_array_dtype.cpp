//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/var_array_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/memblock/zeroinit_memory_block.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/callable.hpp>

using namespace std;
using namespace dynd;

var_array_dtype::var_array_dtype(const dtype& element_dtype)
    : base_dtype(var_array_type_id, uniform_array_kind, sizeof(var_array_dtype_data),
                    sizeof(const char *), dtype_flag_zeroinit, element_dtype.get_undim() + 1),
            m_element_dtype(element_dtype)
{
    // Copy ndobject properties and functions from the first non-uniform dimension
    get_nonuniform_ndobject_properties_and_functions(m_ndobject_properties, m_ndobject_functions);
}

var_array_dtype::~var_array_dtype()
{
}

void var_array_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(data);
    const char *element_data = d->begin + md->offset;
    size_t stride = md->stride;
    metadata += sizeof(var_array_dtype_metadata);
    o << "[";
    for (size_t i = 0, i_end = d->size; i != i_end; ++i, element_data += stride) {
        m_element_dtype.print_data(o, metadata, element_data);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void var_array_dtype::print_dtype(std::ostream& o) const
{
    o << "var_array<" << m_element_dtype << ">";
}

bool var_array_dtype::is_uniform_dim() const
{
    return true;
}

bool var_array_dtype::is_expression() const
{
    return m_element_dtype.is_expression();
}

bool var_array_dtype::is_unique_data_owner(const char *metadata) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    if (md->blockref != NULL &&
            (md->blockref->m_use_count != 1 ||
             (md->blockref->m_type != pod_memory_block_type &&
              md->blockref->m_type != zeroinit_memory_block_type))) {
        return false;
    }
    if (m_element_dtype.is_builtin()) {
        return true;
    } else {
        return m_element_dtype.extended()->is_unique_data_owner(metadata + sizeof(var_array_dtype_metadata));
    }
}

void var_array_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    dtype tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_element_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        out_transformed_dtype = dtype(new var_array_dtype(tmp_dtype), false);
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}

dtype var_array_dtype::get_canonical_dtype() const
{
    return dtype(new var_array_dtype(m_element_dtype.get_canonical_dtype()), false);
}

bool var_array_dtype::is_strided() const
{
    return true;
}

void var_array_dtype::process_strided(const char *metadata, const char *data,
                dtype& out_dt, const char *&out_origin,
                intptr_t& out_stride, intptr_t& out_dim_size) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(data);
    out_dt = m_element_dtype;
    out_origin = d->begin;
    out_stride = md->stride;
    out_dim_size = d->size;
}

dtype var_array_dtype::apply_linear_index(int nindices, const irange *indices,
                int current_i, const dtype& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        if (leading_dimension) {
            // In leading dimensions, we convert var_array to strided_array
            return dtype(new strided_array_dtype(m_element_dtype), false);
        } else {
            return dtype(this, true);
        }
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            if (leading_dimension) {
                if (m_element_dtype.is_builtin()) {
                    return m_element_dtype;
                } else {
                    return m_element_dtype.apply_linear_index(0, NULL, current_i, root_dt, true);
                }
            } else {
                // TODO: This is incorrect, but is here as a stopgap to be replaced by a sliced<> dtype
                return make_pointer_dtype(m_element_dtype);
            }
        } else {
            if (leading_dimension) {
                // In leading dimensions, we convert var_array to strided_array
                return dtype(new strided_array_dtype(m_element_dtype), false);
            } else {
                if (indices->is_nop()) {
                    // If the indexing operation does nothing, then leave things unchanged
                    return dtype(this, true);
                } else {
                    // TODO: sliced_var_array_dtype
                    throw runtime_error("TODO: implement var_array_dtype::apply_linear_index for general slices");
                }
            }
        }
    } else {
        if (indices->step() == 0) {
            if (leading_dimension) {
                return m_element_dtype.apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_dt, true);
            } else {
                // TODO: This is incorrect, but is here as a stopgap to be replaced by a sliced<> dtype
                return make_pointer_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_dt, false));
            }
        } else {
            if (leading_dimension) {
                // In leading dimensions, we convert var_array to strided_array
                dtype edt = m_element_dtype.apply_linear_index(nindices-1, indices+1,
                                current_i+1, root_dt, false);
                return dtype(new strided_array_dtype(edt), false);
            } else {
                if (indices->is_nop()) {
                    // If the indexing operation does nothing, then leave things unchanged
                    dtype edt = m_element_dtype.apply_linear_index(nindices-1, indices+1,
                                    current_i+1, root_dt, false);
                    return dtype(new var_array_dtype(edt), false);
                } else {
                    // TODO: sliced_var_array_dtype
                    throw runtime_error("TODO: implement var_array_dtype::apply_linear_index for general slices");
                    //return dtype(new var_array_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1, current_i+1, root_dt)), false);
                }
            }
        }
    }
}

intptr_t var_array_dtype::apply_linear_index(int nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                int current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (nindices == 0) {
        if (leading_dimension) {
            // Copy the full var_array into a strided_array
            const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
            const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(*inout_data);
            strided_array_dtype_metadata *out_md = reinterpret_cast<strided_array_dtype_metadata *>(out_metadata);
            out_md->size = d->size;
            out_md->stride = md->stride;
            *inout_data = d->begin + md->offset;
            if (*inout_dataref) {
                memory_block_decref(*inout_dataref);
            }
            *inout_dataref = md->blockref ? md->blockref : embedded_reference;
            memory_block_incref(*inout_dataref);
            if (!m_element_dtype.is_builtin()) {
                m_element_dtype.extended()->metadata_copy_construct(
                                out_metadata + sizeof(strided_array_dtype_metadata),
                                metadata + sizeof(var_array_dtype_metadata),
                                embedded_reference);
            }
        } else {
            // If there are no more indices, copy the metadata verbatim
            metadata_copy_construct(out_metadata, metadata, embedded_reference);
        }
        return 0;
    } else {
        const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
        if (leading_dimension) {
            const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(*inout_data);
            bool remove_dimension;
            intptr_t start_index, index_stride, dimension_size;
            apply_single_linear_index(*indices, d->size, current_i, &root_dt,
                            remove_dimension, start_index, index_stride, dimension_size);
            if (remove_dimension) {
                // First dereference to point at the actual element
                const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
                const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(*inout_data);
                *inout_data = d->begin + md->offset + start_index * md->stride;
                if (*inout_dataref) {
                    memory_block_decref(*inout_dataref);
                }
                *inout_dataref = md->blockref ? md->blockref : embedded_reference;
                memory_block_incref(*inout_dataref);
                // Then apply a 0-sized index to the element type
                if (!m_element_dtype.is_builtin()) {
                    return m_element_dtype.extended()->apply_linear_index(
                                    nindices - 1, indices + 1,
                                    metadata + sizeof(var_array_dtype_metadata),
                                    result_dtype, out_metadata, embedded_reference,
                                    current_i, root_dt,
                                    true, inout_data, inout_dataref);
                } else {
                    return 0;
                }
            } else {
                // We can dereference the pointer as we
                // index it and produce a strided array result
                strided_array_dtype_metadata *out_md = reinterpret_cast<strided_array_dtype_metadata *>(out_metadata);
                out_md->size = dimension_size;
                out_md->stride = md->stride * index_stride;
                *inout_data = d->begin + md->offset + md->stride * start_index;
                if (*inout_dataref) {
                    memory_block_decref(*inout_dataref);
                }
                *inout_dataref = md->blockref ? md->blockref : embedded_reference;
                memory_block_incref(*inout_dataref);
                if (nindices == 0) {
                    // Copy the rest of the metadata verbatim, because that part of
                    // the dtype didn't change
                    if (!m_element_dtype.is_builtin()) {
                        m_element_dtype.extended()->metadata_copy_construct(
                                        out_metadata + sizeof(strided_array_dtype_metadata),
                                        metadata + sizeof(var_array_dtype_metadata),
                                        embedded_reference);
                    }
                    return 0;
                } else {
                    if (m_element_dtype.is_builtin()) {
                        return 0;
                    } else {
                        const strided_array_dtype *sad = static_cast<const strided_array_dtype *>(result_dtype.extended());
                        return m_element_dtype.extended()->apply_linear_index(
                                        nindices - 1, indices + 1,
                                        metadata + sizeof(var_array_dtype_metadata),
                                        sad->get_element_dtype(),
                                        out_metadata + sizeof(strided_array_dtype_metadata), embedded_reference,
                                        current_i, root_dt,
                                        false, NULL, NULL);
                    }
                }
            }
        } else {
            if (indices->step() == 0) {
                // TODO: This is incorrect, but is here as a stopgap to be replaced by a sliced<> dtype
                pointer_dtype_metadata *out_md = reinterpret_cast<pointer_dtype_metadata *>(out_metadata);
                out_md->blockref = md->blockref ? md->blockref : embedded_reference;
                memory_block_incref(out_md->blockref);
                out_md->offset = indices->start() * md->stride;
                if (!m_element_dtype.is_builtin()) {
                    const pointer_dtype *result_edtype = static_cast<const pointer_dtype *>(result_dtype.extended());
                    out_md->offset += m_element_dtype.extended()->apply_linear_index(
                                    nindices - 1, indices + 1,
                                    metadata + sizeof(var_array_dtype_metadata),
                                    result_edtype->get_target_dtype(), out_metadata + sizeof(pointer_dtype_metadata),
                                    embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
                return 0;
            } else if (indices->is_nop()) {
                // If the indexing operation does nothing, then leave things unchanged
                var_array_dtype_metadata *out_md = reinterpret_cast<var_array_dtype_metadata *>(out_metadata);
                out_md->blockref = md->blockref ? md->blockref : embedded_reference;
                memory_block_incref(out_md->blockref);
                out_md->stride = md->stride;
                out_md->offset = md->offset;
                if (!m_element_dtype.is_builtin()) {
                    const var_array_dtype *vad = static_cast<const var_array_dtype *>(result_dtype.extended());
                    out_md->offset += m_element_dtype.extended()->apply_linear_index(
                                    nindices - 1, indices + 1,
                                    metadata + sizeof(var_array_dtype_metadata),
                                    vad->get_element_dtype(),
                                    out_metadata + sizeof(var_array_dtype_metadata), embedded_reference,
                                    current_i, root_dt,
                                    false, NULL, NULL);
                }
                return 0;
            } else {
                // TODO: sliced_var_array_dtype
                throw runtime_error("TODO: implement var_array_dtype::apply_linear_index for general slices");
                //return dtype(this, true);
            }
        }
    }
}

dtype var_array_dtype::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    if (inout_metadata) {
        const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(*inout_metadata);
        // Modify the metadata
        *inout_metadata += sizeof(var_array_dtype_metadata);
        // If requested, modify the data pointer
        if (inout_data) {
            const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(*inout_data);
            // Bounds-checking of the index
            i0 = apply_single_index(i0, d->size, NULL);
            *inout_data = d->begin + md->offset + i0 * md->stride;
        }
    }
    return m_element_dtype;
}

dtype var_array_dtype::get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return dtype(this, true);
    } else {
        if (inout_metadata) {
            *inout_metadata += sizeof(var_array_dtype_metadata);
        }
        return m_element_dtype.get_dtype_at_dimension(inout_metadata, i - 1, total_ndim + 1);
    }
}

intptr_t var_array_dtype::get_dim_size(const char *data, const char *DYND_UNUSED(metadata)) const
{
    const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(data);
    return d->size;
}

void var_array_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    // Adjust the current shape if necessary
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape);
    }
}

void var_array_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    // Adjust the current shape if necessary
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape, metadata + sizeof(var_array_dtype_metadata));
    }
}

void var_array_dtype::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);

    out_strides[i] = md->stride;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_strides(i+1, out_strides, metadata + sizeof(var_array_dtype_metadata));
    }
}

intptr_t var_array_dtype::get_representative_stride(const char *metadata) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    return md->stride;
}

bool var_array_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == var_array_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void var_array_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("var_array_dtype::get_single_compare_kernel is unimplemented"); 
}

void var_array_dtype::get_dtype_assignment_kernel(const dtype& DYND_UNUSED(dst_dt), const dtype& DYND_UNUSED(src_dt),
                assign_error_mode DYND_UNUSED(errmode),
                kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("var_array_dtype::get_dtype_assignment_kernel is unimplemented"); 
}

bool var_array_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != var_array_type_id) {
        return false;
    } else {
        const var_array_dtype *dt = static_cast<const var_array_dtype*>(&rhs);
        return m_element_dtype == dt->m_element_dtype;
    }
}

size_t var_array_dtype::get_metadata_size() const
{
    size_t result = sizeof(var_array_dtype_metadata);
    if (!m_element_dtype.is_builtin()) {
        result += m_element_dtype.extended()->get_metadata_size();
    }
    return result;
}

void var_array_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    size_t element_size = m_element_dtype.is_builtin() ? m_element_dtype.get_data_size()
                                                     : m_element_dtype.extended()->get_default_data_size(ndim-1, shape+1);

    var_array_dtype_metadata *md = reinterpret_cast<var_array_dtype_metadata *>(metadata);
    md->stride = element_size;
    md->offset = 0;
    // Allocate a POD memory block
    if (m_element_dtype.get_flags()&dtype_flag_zeroinit) {
        md->blockref = make_zeroinit_memory_block().release();
    } else {
        md->blockref = make_pod_memory_block().release();
    }
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_default_construct(metadata + sizeof(var_array_dtype_metadata), ndim-1, shape+1);
    }
}

void var_array_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const var_array_dtype_metadata *src_md = reinterpret_cast<const var_array_dtype_metadata *>(src_metadata);
    var_array_dtype_metadata *dst_md = reinterpret_cast<var_array_dtype_metadata *>(dst_metadata);
    dst_md->stride = src_md->stride;
    dst_md->offset = src_md->offset;
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    memory_block_incref(dst_md->blockref);
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_copy_construct(dst_metadata + sizeof(var_array_dtype_metadata),
                        src_metadata + sizeof(var_array_dtype_metadata), embedded_reference);
    }
}

void var_array_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement var_array_dtype::metadata_reset_buffers");
}

void var_array_dtype::metadata_finalize_buffers(char *metadata) const
{
    // Finalize any child metadata
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_finalize_buffers(metadata + sizeof(var_array_dtype_metadata));
    }

    // Finalize the blockref buffer we own
    var_array_dtype_metadata *md = reinterpret_cast<var_array_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void var_array_dtype::metadata_destruct(char *metadata) const
{
    var_array_dtype_metadata *md = reinterpret_cast<var_array_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_destruct(metadata + sizeof(var_array_dtype_metadata));
    }
}

void var_array_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    o << indent << "var_array metadata\n";
    o << indent << " stride: " << md->stride << "\n";
    o << indent << " offset: " << md->offset << "\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_debug_print(metadata + sizeof(var_array_dtype_metadata), o, indent + "  ");
    }
}

size_t var_array_dtype::get_iterdata_size(int DYND_UNUSED(ndim)) const
{
    throw runtime_error("TODO: implement var_array_dtype::get_iterdata_size");
}

size_t var_array_dtype::iterdata_construct(iterdata_common *DYND_UNUSED(iterdata), const char **DYND_UNUSED(inout_metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape), dtype& DYND_UNUSED(out_uniform_dtype)) const
{
    throw runtime_error("TODO: implement var_array_dtype::iterdata_construct");
}

size_t var_array_dtype::iterdata_destruct(iterdata_common *DYND_UNUSED(iterdata), int DYND_UNUSED(ndim)) const
{
    throw runtime_error("TODO: implement var_array_dtype::iterdata_destruct");
}

namespace {
    struct broadcast_to_var_assign_kernel_extra {
        typedef broadcast_to_var_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        intptr_t dst_target_alignment;
        const var_array_dtype_metadata *dst_md;

        static void single(char *dst, const char *src,
                            hierarchical_kernel_common_base *extra)
        {
            var_array_dtype_data *dst_d = reinterpret_cast<var_array_dtype_data *>(dst);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_array which has a non-zero offset");
                }
                // If we're writing to an empty array, have to allocate the output
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                // Allocate the output array data
                char *dst_end = NULL;
                allocator->allocate(e->dst_md->blockref, e->dst_md->stride,
                            e->dst_target_alignment, &dst_d->begin, &dst_end);
                dst_d->size = 1;
                // Copy a single input to the newly allocated element
                opchild(dst_d->begin, src, echild);
            } else {
                // We're broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                intptr_t size = dst_d->size, dst_stride = e->dst_md->stride;
                for (intptr_t i = 0; i < size; ++i, dst += dst_stride) {
                    opchild(dst, src, echild);
                }
            }
        }

        static void destruct(hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct var_assign_kernel_extra {
        typedef var_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        intptr_t dst_target_alignment;
        const var_array_dtype_metadata *dst_md, *src_md;

        static void single(char *dst, const char *src,
                            hierarchical_kernel_common_base *extra)
        {
            var_array_dtype_data *dst_d = reinterpret_cast<var_array_dtype_data *>(dst);
            const var_array_dtype_data *src_d = reinterpret_cast<const var_array_dtype_data *>(src);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_array which has a non-zero offset");
                }
                // As a special case, allow uninitialized -> uninitialized assignment as a no-op
                if (src_d->begin != NULL) {
                    intptr_t dim_size = src_d->size;
                    intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_md->stride;
                    // If we're writing to an empty array, have to allocate the output
                    memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                    // Allocate the output array data
                    char *dst_end = NULL;
                    allocator->allocate(e->dst_md->blockref, dim_size * dst_stride,
                                e->dst_target_alignment, &dst_d->begin, &dst_end);
                    dst_d->size = dim_size;
                    // Copy to the newly allocated element
                    dst = dst_d->begin;
                    src = src_d->begin + e->src_md->offset;
                    for (intptr_t i = 0; i < dim_size; ++i, dst += dst_stride, src += src_stride) {
                        opchild(dst, src, echild);
                    }
                }
            } else {
                if (src_d->begin == NULL) {
                    throw runtime_error("Cannot assign an uninitialized dynd var_array to an initialized one");
                }
                intptr_t dst_dim_size = dst_d->size, src_dim_size = src_d->size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = src_dim_size != 1 ? e->src_md->stride : 0;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input var_array sized " << src_dim_size << " to output var_array sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                src = src_d->begin + e->src_md->offset;
                for (intptr_t i = 0; i < dst_dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
                }
            }
        }

        static void destruct(hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };

    struct strided_to_var_assign_kernel_extra {
        typedef strided_to_var_assign_kernel_extra extra_type;

        hierarchical_kernel_common_base base;
        intptr_t dst_target_alignment;
        const var_array_dtype_metadata *dst_md;
        intptr_t src_stride, src_dim_size;

        static void single(char *dst, const char *src,
                            hierarchical_kernel_common_base *extra)
        {
            var_array_dtype_data *dst_d = reinterpret_cast<var_array_dtype_data *>(dst);
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            unary_single_operation_t opchild = (e + 1)->base.get_function<unary_single_operation_t>();
            if (dst_d->begin == NULL) {
                if (e->dst_md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_array which has a non-zero offset");
                }
                intptr_t dim_size = e->src_dim_size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_stride;
                // If we're writing to an empty array, have to allocate the output
                memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(e->dst_md->blockref);

                // Allocate the output array data
                char *dst_end = NULL;
                allocator->allocate(e->dst_md->blockref, dim_size * dst_stride,
                            e->dst_target_alignment, &dst_d->begin, &dst_end);
                dst_d->size = dim_size;
                // Copy to the newly allocated element
                dst = dst_d->begin;
                for (intptr_t i = 0; i < dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
                }
            } else {
                intptr_t dst_dim_size = dst_d->size, src_dim_size = e->src_dim_size;
                intptr_t dst_stride = e->dst_md->stride, src_stride = e->src_stride;
                // Check for a broadcasting error
                if (src_dim_size != 1 && dst_dim_size != src_dim_size) {
                    stringstream ss;
                    ss << "error broadcasting input strided array sized " << src_dim_size << " to output var_array sized " << dst_dim_size;
                    throw broadcast_error(ss.str());
                }
                // We're copying/broadcasting elements to an already allocated array segment
                dst = dst_d->begin + e->dst_md->offset;
                for (intptr_t i = 0; i < dst_dim_size; ++i, dst += dst_stride, src += src_stride) {
                    opchild(dst, src, echild);
                }
            }
        }

        static void destruct(hierarchical_kernel_common_base *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            hierarchical_kernel_common_base *echild = &(e + 1)->base;
            if (echild->destructor) {
                echild->destructor(echild);
            }
        }
    };
} // anonymous namespace

void var_array_dtype::make_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t out_offset,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        if (src_dt.get_undim() < dst_dt.get_undim()) {
            // If the src has fewer dimensions, broadcast it across this one
            out->ensure_capacity(out_offset + sizeof(broadcast_to_var_assign_kernel_extra));
            const var_array_dtype_metadata *dst_md =
                            reinterpret_cast<const var_array_dtype_metadata *>(dst_metadata);
            broadcast_to_var_assign_kernel_extra *e = out->get_at<broadcast_to_var_assign_kernel_extra>(out_offset);
            e->base.function = &broadcast_to_var_assign_kernel_extra::single;
            e->base.destructor = &broadcast_to_var_assign_kernel_extra::destruct;
            e->dst_target_alignment = m_element_dtype.get_alignment();
            e->dst_md = dst_md;
            ::make_assignment_kernel(out, out_offset + sizeof(broadcast_to_var_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(var_array_dtype_metadata),
                            src_dt, src_metadata,
                            errmode, ectx);
        } else if (src_dt.get_type_id() == var_array_type_id) {
            // var_array to var_array
            const var_array_dtype *src_vad = static_cast<const var_array_dtype *>(src_dt.extended());
            out->ensure_capacity(out_offset + sizeof(var_assign_kernel_extra));
            const var_array_dtype_metadata *dst_md =
                            reinterpret_cast<const var_array_dtype_metadata *>(dst_metadata);
            const var_array_dtype_metadata *src_md =
                            reinterpret_cast<const var_array_dtype_metadata *>(src_metadata);
            var_assign_kernel_extra *e = out->get_at<var_assign_kernel_extra>(out_offset);
            e->base.function = &var_assign_kernel_extra::single;
            e->base.destructor = &var_assign_kernel_extra::destruct;
            e->dst_target_alignment = m_element_dtype.get_alignment();
            e->dst_md = dst_md;
            e->src_md = src_md;
            ::make_assignment_kernel(out, out_offset + sizeof(var_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(var_array_dtype_metadata),
                            src_vad->get_element_dtype(), src_metadata + sizeof(var_array_dtype_metadata),
                            errmode, ectx);
        } else if (src_dt.get_type_id() == strided_array_type_id) {
            // strided_array to var_array
            const strided_array_dtype *src_sad = static_cast<const strided_array_dtype *>(src_dt.extended());
            out->ensure_capacity(out_offset + sizeof(strided_to_var_assign_kernel_extra));
            const var_array_dtype_metadata *dst_md =
                            reinterpret_cast<const var_array_dtype_metadata *>(dst_metadata);
            const strided_array_dtype_metadata *src_md =
                            reinterpret_cast<const strided_array_dtype_metadata *>(src_metadata);
            strided_to_var_assign_kernel_extra *e = out->get_at<strided_to_var_assign_kernel_extra>(out_offset);
            e->base.function = &strided_to_var_assign_kernel_extra::single;
            e->base.destructor = &strided_to_var_assign_kernel_extra::destruct;
            e->dst_target_alignment = m_element_dtype.get_alignment();
            e->dst_md = dst_md;
            e->src_stride = src_md->stride;
            e->src_dim_size = src_md->size;
            ::make_assignment_kernel(out, out_offset + sizeof(strided_to_var_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(var_array_dtype_metadata),
                            src_sad->get_element_dtype(), src_metadata + sizeof(strided_array_dtype_metadata),
                            errmode, ectx);
        } else if (src_dt.get_type_id() == fixedarray_type_id) {
            // fixedarray to var_array
            const fixedarray_dtype *src_fad = static_cast<const fixedarray_dtype *>(src_dt.extended());
            out->ensure_capacity(out_offset + sizeof(strided_to_var_assign_kernel_extra));
            const var_array_dtype_metadata *dst_md =
                            reinterpret_cast<const var_array_dtype_metadata *>(dst_metadata);
            strided_to_var_assign_kernel_extra *e = out->get_at<strided_to_var_assign_kernel_extra>(out_offset);
            e->base.function = &strided_to_var_assign_kernel_extra::single;
            e->base.destructor = &strided_to_var_assign_kernel_extra::destruct;
            e->dst_target_alignment = m_element_dtype.get_alignment();
            e->dst_md = dst_md;
            e->src_stride = src_fad->get_fixed_stride();
            e->src_dim_size = src_fad->get_fixed_dim_size();
            ::make_assignment_kernel(out, out_offset + sizeof(strided_to_var_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(var_array_dtype_metadata),
                            src_fad->get_element_dtype(), src_metadata,
                            errmode, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_dt << " to " << dst_dt;
            throw runtime_error(ss.str());
        }
    } else if (dst_dt.get_undim() < src_dt.get_undim()) {
        throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
    } else {
        if (dst_dt.get_type_id() == strided_array_type_id) {
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_dt << " to " << dst_dt;
            throw runtime_error(ss.str());
        }
    }
}

void var_array_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    const var_array_dtype_metadata *md = reinterpret_cast<const var_array_dtype_metadata *>(metadata);
    const char *child_metadata = metadata + sizeof(var_array_dtype_metadata);
    const var_array_dtype_data *d = reinterpret_cast<const var_array_dtype_data *>(data);
    data = d->begin + md->offset;
    intptr_t stride = md->stride;
    for (intptr_t i = 0, i_end = d->size; i < i_end; ++i, data += stride) {
        callback(m_element_dtype, data, child_metadata, callback_data);
    }
}

void var_array_dtype::reorder_default_constructed_strides(char *dst_metadata,
                const dtype& src_dtype, const char *src_metadata) const
{
    // The blockref array dtype can't be reordered, so just let any deeper dtypes do their reordering.
    if (!m_element_dtype.is_builtin()) {
        dtype src_child_dtype = src_dtype.at_single(0, &src_metadata);
        m_element_dtype.extended()->reorder_default_constructed_strides(dst_metadata + sizeof(var_array_dtype_metadata),
                        src_child_dtype, src_metadata);
    }
}

void var_array_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_ndobject_properties.empty() ? NULL : &m_ndobject_properties[0];
    *out_count = (int)m_ndobject_properties.size();
}

void var_array_dtype::get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = m_ndobject_functions.empty() ? NULL : &m_ndobject_functions[0];
    *out_count = (int)m_ndobject_functions.size();
}
