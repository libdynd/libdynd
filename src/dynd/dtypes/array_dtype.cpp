//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/array_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/pointer_dtype.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>

using namespace std;
using namespace dynd;

array_dtype::array_dtype(const dtype& element_dtype)
    : extended_dtype(array_type_id, uniform_array_kind, sizeof(array_dtype_data),
                    sizeof(const char *), element_dtype.get_undim() + 1),
            m_element_dtype(element_dtype)
{
}

array_dtype::~array_dtype()
{
}

void array_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(metadata);
    const array_dtype_data *d = reinterpret_cast<const array_dtype_data *>(data);
    size_t stride = md->stride;
    metadata += sizeof(array_dtype_metadata);
    o << "[";
    for (size_t i = 0, i_end = d->size; i != i_end; ++i, data += stride) {
        m_element_dtype.print_data(o, metadata, data);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void array_dtype::print_dtype(std::ostream& o) const
{
    o << "strided_array<" << m_element_dtype << ">";
}

bool array_dtype::is_scalar() const
{
    return false;
}

bool array_dtype::is_uniform_dim() const
{
    return true;
}

bool array_dtype::is_expression() const
{
    return m_element_dtype.is_expression();
}

void array_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    dtype tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_element_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        out_transformed_dtype = dtype(new array_dtype(tmp_dtype));
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}

dtype array_dtype::get_canonical_dtype() const
{
    return dtype(new array_dtype(m_element_dtype.get_canonical_dtype()));
}

dtype array_dtype::apply_linear_index(int nindices, const irange *indices, int current_i, const dtype& root_dt) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            return make_pointer_dtype(m_element_dtype);
        } else {
            return dtype(this, true);
        }
    } else {
        if (indices->step() == 0) {
            return make_pointer_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1, current_i+1, root_dt));
        } else {
            // TODO: sliced_array_dtype
            return dtype(new array_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1, current_i+1, root_dt)));
        }
    }
}

intptr_t array_dtype::apply_linear_index(int DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices), char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata),
                const dtype& DYND_UNUSED(result_dtype), char *DYND_UNUSED(out_metadata),
                memory_block_data *DYND_UNUSED(embedded_reference),
                int DYND_UNUSED(current_i), const dtype& DYND_UNUSED(root_dt)) const
{
    throw runtime_error("TODO: implement array_dtype::apply_linear_index");
}

dtype array_dtype::at(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    if (inout_metadata) {
        const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(*inout_metadata);
        // Modify the metadata
        *inout_metadata += sizeof(array_dtype_metadata);
        // If requested, modify the data pointer
        if (inout_data) {
            const array_dtype_data *d = reinterpret_cast<const array_dtype_data *>(*inout_data);
            // Bounds-checking of the index
            i0 = apply_single_index(i0, d->size, NULL);
            *inout_data = d->begin + i0 * md->stride;
        }
    }
    return m_element_dtype;
}

dtype array_dtype::get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return dtype(this, true);
    } else {
        if (inout_metadata) {
            *inout_metadata += sizeof(array_dtype_metadata);
        }
        return m_element_dtype.get_dtype_at_dimension(inout_metadata, i - 1, total_ndim + 1);
    }
}

intptr_t array_dtype::get_dim_size(const char *data, const char *DYND_UNUSED(metadata)) const
{
    const array_dtype_data *d = reinterpret_cast<const array_dtype_data *>(data);
    return d->size;
}

void array_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    // Adjust the current shape if necessary
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape);
    }
}

void array_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    // Adjust the current shape if necessary
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape, metadata);
    }
}

void array_dtype::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(metadata);

    out_strides[i] = md->stride;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_strides(i+1, out_strides, metadata + sizeof(array_dtype_metadata));
    }
}

intptr_t array_dtype::get_representative_stride(const char *metadata) const
{
    const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(metadata);
    return md->stride;
}

bool array_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == array_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void array_dtype::get_single_compare_kernel(single_compare_kernel_instance& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("array_dtype::get_single_compare_kernel is unimplemented"); 
}

void array_dtype::get_dtype_assignment_kernel(const dtype& DYND_UNUSED(dst_dt), const dtype& DYND_UNUSED(src_dt),
                assign_error_mode DYND_UNUSED(errmode),
                kernel_instance<unary_operation_pair_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("array_dtype::get_dtype_assignment_kernel is unimplemented"); 
}

bool array_dtype::operator==(const extended_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != array_type_id) {
        return false;
    } else {
        const array_dtype *dt = static_cast<const array_dtype*>(&rhs);
        return m_element_dtype == dt->m_element_dtype;
    }
}

size_t array_dtype::get_metadata_size() const
{
    size_t result = sizeof(array_dtype_metadata);
    if (!m_element_dtype.is_builtin()) {
        result += m_element_dtype.extended()->get_metadata_size();
    }
    return result;
}

void array_dtype::metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const
{
    size_t element_size = m_element_dtype.is_builtin() ? m_element_dtype.get_data_size()
                                                     : m_element_dtype.extended()->get_default_data_size(ndim-1, shape+1);

    array_dtype_metadata *md = reinterpret_cast<array_dtype_metadata *>(metadata);
    md->stride = element_size;
    // Allocate a POD memory block
    md->blockref = make_pod_memory_block().release();
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_default_construct(metadata + sizeof(array_dtype_metadata), ndim-1, shape+1);
    }
}

void array_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const array_dtype_metadata *src_md = reinterpret_cast<const array_dtype_metadata *>(src_metadata);
    array_dtype_metadata *dst_md = reinterpret_cast<array_dtype_metadata *>(dst_metadata);
    dst_md->stride = src_md->stride;
    dst_md->blockref = src_md->blockref ? src_md->blockref : embedded_reference;
    memory_block_incref(dst_md->blockref);
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_copy_construct(dst_metadata + sizeof(array_dtype_metadata),
                        src_metadata + sizeof(array_dtype_metadata), embedded_reference);
    }
}

void array_dtype::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    throw runtime_error("TODO implement array_dtype::metadata_reset_buffers");
}

void array_dtype::metadata_finalize_buffers(char *metadata) const
{
    // Finalize any child metadata
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_finalize_buffers(metadata + sizeof(array_dtype_metadata));
    }

    // Finalize the blockref buffer we own
    array_dtype_metadata *md = reinterpret_cast<array_dtype_metadata *>(metadata);
    if (md->blockref != NULL) {
        // Finalize the memory block
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        if (allocator != NULL) {
            allocator->finalize(md->blockref);
        }
    }
}

void array_dtype::metadata_destruct(char *metadata) const
{
    array_dtype_metadata *md = reinterpret_cast<array_dtype_metadata *>(metadata);
    if (md->blockref) {
        memory_block_decref(md->blockref);
    }
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_destruct(metadata + sizeof(array_dtype_metadata));
    }
}

void array_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(metadata);
    o << indent << "array metadata\n";
    o << indent << " stride: " << md->stride << "\n";
    memory_block_debug_print(md->blockref, o, indent + " ");
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_debug_print(metadata + sizeof(array_dtype_metadata), o, indent + "  ");
    }
}

size_t array_dtype::get_iterdata_size(int DYND_UNUSED(ndim)) const
{
    throw runtime_error("TODO: implement array_dtype::get_iterdata_size");
}

size_t array_dtype::iterdata_construct(iterdata_common *DYND_UNUSED(iterdata), const char **DYND_UNUSED(inout_metadata), int DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape), dtype& DYND_UNUSED(out_uniform_dtype)) const
{
    throw runtime_error("TODO: implement array_dtype::iterdata_construct");
}

size_t array_dtype::iterdata_destruct(iterdata_common *DYND_UNUSED(iterdata), int DYND_UNUSED(ndim)) const
{
    throw runtime_error("TODO: implement array_dtype::iterdata_destruct");
}

void array_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    const array_dtype_metadata *md = reinterpret_cast<const array_dtype_metadata *>(metadata);
    const char *child_metadata = metadata + sizeof(array_dtype_metadata);
    const array_dtype_data *d = reinterpret_cast<const array_dtype_data *>(data);
    data = d->begin;
    intptr_t stride = md->stride;
    for (intptr_t i = 0, i_end = d->size; i < i_end; ++i, data += stride) {
        callback(m_element_dtype, data, child_metadata, callback_data);
    }
}

void array_dtype::reorder_default_constructed_strides(char *dst_metadata,
                const dtype& src_dtype, const char *src_metadata) const
{
    // The blockref array dtype can't be reordered, so just let any deeper dtypes do their reordering.
    if (!m_element_dtype.is_builtin()) {
        dtype src_child_dtype = src_dtype.at_single(0, &src_metadata);
        m_element_dtype.extended()->reorder_default_constructed_strides(dst_metadata + sizeof(array_dtype_metadata),
                        src_child_dtype, src_metadata);
    }
}
