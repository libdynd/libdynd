//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

fixedarray_dtype::fixedarray_dtype(size_t dimension_size, const dtype& element_dtype)
    : base_dtype(fixedarray_type_id, uniform_array_kind, 0, 1, dtype_flag_none,
                    element_dtype.get_metadata_size(), element_dtype.get_undim() + 1),
            m_element_dtype(element_dtype), m_dimension_size(dimension_size)
{
    size_t child_element_size = element_dtype.get_data_size();
    if (child_element_size == 0) {
        stringstream ss;
        ss << "Cannot create fixedarray dtype with element type " << element_dtype;
        ss << ", as it does not have a fixed size";
        throw runtime_error(ss.str());
    }
    m_stride = m_dimension_size > 1 ? element_dtype.get_data_size() : 0;
    m_members.data_size = m_stride * (m_dimension_size-1) + child_element_size;
    m_members.alignment = (uint8_t)m_element_dtype.get_alignment();
    // Propagate the zeroinit flag from the element
    m_members.flags |= (element_dtype.get_flags()&dtype_flag_zeroinit);

    // Copy ndobject properties and functions from the first non-uniform dimension
    get_nonuniform_ndobject_properties_and_functions(m_ndobject_properties, m_ndobject_functions);
}

fixedarray_dtype::fixedarray_dtype(size_t dimension_size, const dtype& element_dtype, intptr_t stride)
    : base_dtype(fixedarray_type_id, uniform_array_kind, 0, 1, dtype_flag_none,
                    element_dtype.get_metadata_size(), element_dtype.get_undim() + 1),
            m_element_dtype(element_dtype), m_stride(stride), m_dimension_size(dimension_size)
{
    size_t child_element_size = element_dtype.get_data_size();
    if (child_element_size == 0) {
        stringstream ss;
        ss << "Cannot create fixedarray dtype with element type " << element_dtype;
        ss << ", as it does not have a fixed size";
        throw runtime_error(ss.str());
    }
    if (dimension_size <= 1 && stride != 0) {
        stringstream ss;
        ss << "Cannot create fixedarray dtype with size " << dimension_size;
        ss << " and stride " << stride << ", as the stride must be zero when the dimension size is 1";
        throw runtime_error(ss.str());
    }
    if (dimension_size > 1 && stride == 0) {
        stringstream ss;
        ss << "Cannot create fixedarray dtype with size " << dimension_size;
        ss << " and stride 0, as the stride must be non-zero when the dimension size is > 1";
        throw runtime_error(ss.str());
    }
    m_members.data_size = m_stride * (m_dimension_size-1) + child_element_size;
    m_members.alignment = (uint8_t)m_element_dtype.get_alignment();
    // Propagate the zeroinit flag from the element
    m_members.flags |= (element_dtype.get_flags()&dtype_flag_zeroinit);

    // Copy ndobject properties and functions from the first non-uniform dimension
    get_nonuniform_ndobject_properties_and_functions(m_ndobject_properties, m_ndobject_functions);
}

fixedarray_dtype::~fixedarray_dtype()
{
}

void fixedarray_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    size_t stride = m_stride;
    o << "[";
    for (size_t i = 0, i_end = m_dimension_size; i != i_end; ++i, data += stride) {
        m_element_dtype.print_data(o, metadata, data);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void fixedarray_dtype::print_dtype(std::ostream& o) const
{
    o << "fixedarray<";
    o << m_dimension_size;
    if ((size_t)m_stride != m_element_dtype.get_data_size()) {
        o << ", stride=" << m_stride;
    }
    o << ", " << m_element_dtype;
    o << ">";
}

bool fixedarray_dtype::is_uniform_dim() const
{
    return true;
}

bool fixedarray_dtype::is_expression() const
{
    return m_element_dtype.is_expression();
}

bool fixedarray_dtype::is_unique_data_owner(const char *metadata) const
{
    if (m_element_dtype.is_builtin()) {
        return true;
    } else {
        return m_element_dtype.extended()->is_unique_data_owner(metadata);
    }
}

void fixedarray_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    dtype tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_element_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        if (tmp_dtype.get_data_size() != 0) {
            out_transformed_dtype = dtype(new fixedarray_dtype(m_dimension_size, tmp_dtype), false);
        } else {
            out_transformed_dtype = dtype(new strided_array_dtype(tmp_dtype), false);
        }
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}


dtype fixedarray_dtype::get_canonical_dtype() const
{
    dtype canonical_element_dtype = m_element_dtype.get_canonical_dtype();
    // The transformed dtype may no longer have a fixed size, so check whether
    // we have to switch to the more flexible strided_array_dtype
    if (canonical_element_dtype.get_data_size() != 0) {
        return dtype(new fixedarray_dtype(m_dimension_size, canonical_element_dtype), false);
    } else {
        return dtype(new strided_array_dtype(canonical_element_dtype), false);
    }
}

bool fixedarray_dtype::is_strided() const
{
    return true;
}

void fixedarray_dtype::process_strided(const char *DYND_UNUSED(metadata), const char *data,
                dtype& out_dt, const char *&out_origin,
                intptr_t& out_stride, intptr_t& out_dim_size) const
{
    out_dt = m_element_dtype;
    out_origin = data;
    out_stride = m_stride;
    out_dim_size = m_dimension_size;
}

dtype fixedarray_dtype::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const dtype& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        return dtype(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            if (leading_dimension && !m_element_dtype.is_builtin()) {
                // For leading dimensions, need to give the next dtype a chance
                // to collapse itself even though indexing doesn't continue further.
                return m_element_dtype.extended()->apply_linear_index(0, NULL, current_i, root_dt, true);
            } else {
                return m_element_dtype;
            }
        } else {
            if (indices->is_nop()) {
                return dtype(this, true);
            } else {
                return dtype(new strided_array_dtype(m_element_dtype), false);
            }
        }
    } else {
        if (indices->step() == 0) {
            return m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, leading_dimension);
        } else {
            return dtype(new strided_array_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, false)), false);
        }
    }
}

intptr_t fixedarray_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (nindices == 0 || result_dtype.get_type_id() == fixedarray_type_id) {
        // If there are no more indices, or the operation is a no-op
        // as signaled by retaining the fixed array dtype,
        // copy the metadata verbatim
        metadata_copy_construct(out_metadata, metadata, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_dimension_size, current_i, &root_dt,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            // Apply the strided offset and continue applying the index
            intptr_t offset = m_stride * start_index;
            if (!m_element_dtype.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata, result_dtype, out_metadata, embedded_reference, current_i + 1, root_dt,
                                    true, inout_data, inout_dataref);
                } else {
                    offset += m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata, result_dtype, out_metadata, embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
            }
            return offset;
        } else {
            strided_array_dtype_metadata *out_md = reinterpret_cast<strided_array_dtype_metadata *>(out_metadata);
            // Produce the new offset data, stride, and size for the resulting array,
            // which is now a strided_array instead of a fixedarray
            intptr_t offset = m_stride * start_index;
            out_md->stride = m_stride * index_stride;
            out_md->size = dimension_size;
            if (!m_element_dtype.is_builtin()) {
                const strided_array_dtype *result_edtype = static_cast<const strided_array_dtype *>(result_dtype.extended());
                offset += m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                metadata,
                                result_edtype->get_element_dtype(),
                                out_metadata + sizeof(strided_array_dtype_metadata),
                                embedded_reference, current_i + 1, root_dt,
                                false, NULL, NULL);
            }
            return offset;
        }
    }
}

dtype fixedarray_dtype::at_single(intptr_t i0,
                const char **DYND_UNUSED(inout_metadata), const char **inout_data) const
{
    // Bounds-checking of the index
    i0 = apply_single_index(i0, m_dimension_size, NULL);
    // The fixedarray dtype has no metadata
    // If requested, modify the data
    if (inout_data) {
        *inout_data += i0 * m_stride;
    }
    return m_element_dtype;
}

dtype fixedarray_dtype::get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return dtype(this, true);
    } else {
        return m_element_dtype.get_dtype_at_dimension(inout_metadata, i - 1, total_ndim + 1);
    }
}

intptr_t fixedarray_dtype::get_dim_size(const char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata)) const
{
    return m_dimension_size;
}

void fixedarray_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    out_shape[i] = m_dimension_size;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape);
    }
}

void fixedarray_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    out_shape[i] = m_dimension_size;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape, metadata);
    }
}

void fixedarray_dtype::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    out_strides[i] = m_stride;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_strides(i+1, out_strides, metadata);
    }
}

intptr_t fixedarray_dtype::get_representative_stride(const char *DYND_UNUSED(metadata)) const
{
    return m_stride;
}

bool fixedarray_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == fixedarray_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void fixedarray_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("fixedarray_dtype::get_single_compare_kernel is unimplemented"); 
}

bool fixedarray_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixedarray_type_id) {
        return false;
    } else {
        const fixedarray_dtype *dt = static_cast<const fixedarray_dtype*>(&rhs);
        return m_element_dtype == dt->m_element_dtype &&
                m_dimension_size == dt->m_dimension_size &&
                m_stride == dt->m_stride;
    }
}

void fixedarray_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim > 0) {
        if (shape[0] >= 0 && (size_t)shape[0] != m_dimension_size) {
            stringstream ss;
            ss << "Cannot construct dynd object of dtype " << dtype(this, true);
            ss << " with dimension size " << shape[0] << ", the size must be " << m_dimension_size;
            throw runtime_error(ss.str());
        }
    }

    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_default_construct(metadata, ndim ? (ndim-1) : 0, shape+1);
    }
}

void fixedarray_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

void fixedarray_dtype::metadata_reset_buffers(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_reset_buffers(metadata);
    }
}

void fixedarray_dtype::metadata_finalize_buffers(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_finalize_buffers(metadata);
    }
}

void fixedarray_dtype::metadata_destruct(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_destruct(metadata);
    }
}

void fixedarray_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_debug_print(metadata, o, indent);
    }
}

size_t fixedarray_dtype::get_iterdata_size(size_t ndim) const
{
    if (ndim == 0) {
        return 0;
    } else if (ndim == 1) {
        return sizeof(fixedarray_dtype_iterdata);
    } else {
        return m_element_dtype.get_iterdata_size(ndim - 1) + sizeof(fixedarray_dtype_iterdata);
    }
}

// Does one iterator increment for this dtype
static char *iterdata_incr(iterdata_common *iterdata, size_t level)
{
    fixedarray_dtype_iterdata *id = reinterpret_cast<fixedarray_dtype_iterdata *>(iterdata);
    if (level == 0) {
        id->data += id->stride;
        return id->data;
    } else {
        id->data = (id + 1)->common.incr(&(id + 1)->common, level - 1);
        return id->data;
    }
}

static char *iterdata_reset(iterdata_common *iterdata, char *data, size_t ndim)
{
    fixedarray_dtype_iterdata *id = reinterpret_cast<fixedarray_dtype_iterdata *>(iterdata);
    if (ndim == 1) {
        id->data = data;
        return data;
    } else {
        id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
        return id->data;
    }
}

size_t fixedarray_dtype::iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, size_t ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        // Place any inner iterdata earlier than the outer iterdata
        inner_size = m_element_dtype.extended()->iterdata_construct(iterdata, inout_metadata,
                        ndim - 1, shape + 1, out_uniform_dtype);
        iterdata = reinterpret_cast<iterdata_common *>(reinterpret_cast<char *>(iterdata) + inner_size);
    } else {
        out_uniform_dtype = m_element_dtype;
    }

    if (m_dimension_size != 1 && (size_t)shape[0] != m_dimension_size) {
        stringstream ss;
        ss << "Cannot construct dynd iterator of dtype " << dtype(this, true);
        ss << " with dimension size " << shape[0] << ", the size must be " << m_dimension_size;
        throw runtime_error(ss.str());
    }

    fixedarray_dtype_iterdata *id = reinterpret_cast<fixedarray_dtype_iterdata *>(iterdata);

    id->common.incr = &iterdata_incr;
    id->common.reset = &iterdata_reset;
    id->data = NULL;
    id->stride = m_stride;

    return inner_size + sizeof(fixedarray_dtype_iterdata);
}

size_t fixedarray_dtype::iterdata_destruct(iterdata_common *iterdata, size_t ndim) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        inner_size = m_element_dtype.extended()->iterdata_destruct(iterdata, ndim - 1);
    }
    // No dynamic data to free
    return inner_size + sizeof(fixedarray_dtype_iterdata);
}

size_t fixedarray_dtype::make_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        out->ensure_capacity(offset_out + sizeof(strided_assign_kernel_extra));
        strided_assign_kernel_extra *e = out->get_at<strided_assign_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(&strided_assign_kernel_extra::single);
        e->base.destructor = strided_assign_kernel_extra::destruct;
        if (src_dt.get_undim() < dst_dt.get_undim()) {
            // If the src has fewer dimensions, broadcast it across this one
            e->size = get_fixed_dim_size();
            e->dst_stride = get_fixed_stride();
            e->src_stride = 0;
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(strided_array_dtype_metadata),
                            src_dt, src_metadata,
                            errmode, ectx);
        } else if (src_dt.get_type_id() == fixedarray_type_id) {
            // fixed_array -> strided_array
            const fixedarray_dtype *src_fad = static_cast<const fixedarray_dtype *>(src_dt.extended());
            intptr_t src_size = src_fad->get_fixed_dim_size();
            intptr_t dst_size = get_fixed_dim_size();
            // Check for a broadcasting error
            if (src_size != 1 && dst_size != src_size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
            }
            e->size = dst_size;
            e->dst_stride = get_fixed_stride();
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride = src_fad->get_fixed_stride();
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata,
                            src_fad->get_element_dtype(), src_metadata,
                            errmode, ectx);
        } else if (src_dt.get_type_id() == strided_array_type_id) {
            // strided_array -> strided_array
            const strided_array_dtype *src_sad = static_cast<const strided_array_dtype *>(src_dt.extended());
            const strided_array_dtype_metadata *src_md =
                        reinterpret_cast<const strided_array_dtype_metadata *>(src_metadata);
            intptr_t dst_size = get_fixed_dim_size();
            // Check for a broadcasting error
            if (src_md->size != 1 && dst_size != src_md->size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
            }
            e->size = dst_size;
            e->dst_stride = get_fixed_stride();
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride = src_md->stride;
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata,
                            src_sad->get_element_dtype(), src_metadata + sizeof(strided_array_dtype_metadata),
                            errmode, ectx);
        } else if (!src_dt.is_builtin()) {
            // Give the src dtype a chance to make a kernel
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            errmode, ectx);
        } else {
            stringstream ss;
            ss << "Cannot assign from " << src_dt << " to " << dst_dt;
            throw runtime_error(ss.str());
        }
    } else if (dst_dt.get_undim() < src_dt.get_undim()) {
        throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
    } else {
        stringstream ss;
        ss << "Cannot assign from " << src_dt << " to " << dst_dt;
        throw runtime_error(ss.str());
    }
}

void fixedarray_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    intptr_t stride = m_stride;
    for (intptr_t i = 0, i_end = m_dimension_size; i < i_end; ++i, data += stride) {
        callback(m_element_dtype, data, metadata, callback_data);
    }
}

void fixedarray_dtype::reorder_default_constructed_strides(char *DYND_UNUSED(dst_metadata),
                const dtype& DYND_UNUSED(src_dtype), const char *DYND_UNUSED(src_metadata)) const
{
    // Because everything contained in the fixedarray must have fixed size, it can't
    // be reordered. This makes this function a NOP
}

dtype dynd::make_fixedarray_dtype(size_t ndim, const intptr_t *shape,
                const dtype& uniform_dtype, const int *axis_perm)
{
    if (axis_perm == NULL) {
        // Build a C-order fixed array dtype
        dtype result = uniform_dtype;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = make_fixedarray_dtype(shape[i], result);
        }
        return result;
    } else {
        // Create strides with the axis permutation
        dimvector strides(ndim);
        intptr_t stride = uniform_dtype.get_data_size();
        for (size_t i = 0; i < ndim; ++i) {
            int i_perm = axis_perm[i];
            size_t dim_size = shape[i_perm];
            strides[i_perm] = dim_size > 1 ? stride : 0;
            stride *= dim_size;
        }
        // Build the fixed array dtype
        dtype result = uniform_dtype;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = make_fixedarray_dtype(shape[i], result, strides[i]);
        }
        return result;
    }
}

static size_t get_fixed_dim_size(const dtype& dt) {
    const fixedarray_dtype *d = static_cast<const fixedarray_dtype *>(dt.extended());
    return d->get_fixed_dim_size();
}

static intptr_t get_fixed_dim_stride(const dtype& dt) {
    const fixedarray_dtype *d = static_cast<const fixedarray_dtype *>(dt.extended());
    return d->get_fixed_stride();
}

static pair<string, gfunc::callable> fixedarray_dtype_properties[] = {
    pair<string, gfunc::callable>("fixed_dim_size", gfunc::make_callable(&get_fixed_dim_size, "self")),
    pair<string, gfunc::callable>("fixed_dim_stride", gfunc::make_callable(&get_fixed_dim_stride, "self"))
};

void fixedarray_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = fixedarray_dtype_properties;
    *out_count = sizeof(fixedarray_dtype_properties) / sizeof(fixedarray_dtype_properties[0]);
}

void fixedarray_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_ndobject_properties.empty() ? NULL : &m_ndobject_properties[0];
    *out_count = (int)m_ndobject_properties.size();
}

void fixedarray_dtype::get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = m_ndobject_functions.empty() ? NULL : &m_ndobject_functions[0];
    *out_count = (int)m_ndobject_functions.size();
}
