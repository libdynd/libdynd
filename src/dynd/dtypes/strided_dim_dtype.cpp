//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

strided_dim_dtype::strided_dim_dtype(const dtype& element_dtype)
    : base_dtype(strided_dim_type_id, uniform_dim_kind, 0,
                    element_dtype.get_alignment(), dtype_flag_none,
                    element_dtype.get_metadata_size() + sizeof(strided_dim_dtype_metadata),
                    element_dtype.get_undim() + 1),
            m_element_dtype(element_dtype)
{
    // Propagate the zeroinit flag from the element
    m_members.flags |= (element_dtype.get_flags()&dtype_flag_zeroinit);

    // Copy ndobject properties and functions from the first non-uniform dimension
    get_nonuniform_ndobject_properties_and_functions(m_ndobject_properties, m_ndobject_functions);
}

strided_dim_dtype::~strided_dim_dtype()
{
}

size_t strided_dim_dtype::get_default_data_size(size_t ndim, const intptr_t *shape) const
{
    if (ndim == 0 || shape[0] < 0) {
        throw std::runtime_error("the strided_dim dtype requires a shape be specified for default construction");
    }

    if (!m_element_dtype.is_builtin()) {
        return shape[0] * m_element_dtype.extended()->get_default_data_size(ndim-1, shape+1);
    } else {
        return shape[0] * m_element_dtype.get_data_size();
    }
}


void strided_dim_dtype::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    size_t stride = md->stride;
    metadata += sizeof(strided_dim_dtype_metadata);
    o << "[";
    for (size_t i = 0, i_end = md->size; i != i_end; ++i, data += stride) {
        m_element_dtype.print_data(o, metadata, data);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void strided_dim_dtype::print_dtype(std::ostream& o) const
{
    o << "strided_dim<" << m_element_dtype << ">";
}

bool strided_dim_dtype::is_uniform_dim() const
{
    return true;
}

bool strided_dim_dtype::is_expression() const
{
    return m_element_dtype.is_expression();
}

bool strided_dim_dtype::is_unique_data_owner(const char *metadata) const
{
    if (m_element_dtype.is_builtin()) {
        return true;
    } else {
        return m_element_dtype.extended()->is_unique_data_owner(metadata + sizeof(strided_dim_dtype_metadata));
    }
}

void strided_dim_dtype::transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                dtype& out_transformed_dtype, bool& out_was_transformed) const
{
    dtype tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_element_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        out_transformed_dtype = dtype(new strided_dim_dtype(tmp_dtype), false);
        out_was_transformed = true;
    } else {
        out_transformed_dtype = dtype(this, true);
    }
}


dtype strided_dim_dtype::get_canonical_dtype() const
{
    return dtype(new strided_dim_dtype(m_element_dtype.get_canonical_dtype()), false);
}

bool strided_dim_dtype::is_strided() const
{
    return true;
}

void strided_dim_dtype::process_strided(const char *metadata, const char *data,
                dtype& out_dt, const char *&out_origin,
                intptr_t& out_stride, intptr_t& out_dim_size) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    out_dt = m_element_dtype;
    out_origin = data;
    out_stride = md->stride;
    out_dim_size = md->size;
}

dtype strided_dim_dtype::apply_linear_index(size_t nindices, const irange *indices,
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
            return dtype(this, true);
        }
    } else {
        if (indices->step() == 0) {
            return m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, leading_dimension);
        } else {
            return dtype(new strided_dim_dtype(m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, false)), false);
        }
    }
}

intptr_t strided_dim_dtype::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const dtype& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const dtype& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    strided_dim_dtype_metadata *out_md = reinterpret_cast<strided_dim_dtype_metadata *>(out_metadata);
    if (nindices == 0) {
        // If there are no more indices, copy the rest verbatim
        metadata_copy_construct(out_metadata, metadata, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, md->size, current_i, &root_dt,
                        remove_dimension, start_index, index_stride, dimension_size);
        if (remove_dimension) {
            // Apply the strided offset and continue applying the index
            intptr_t offset = md->stride * start_index;
            if (!m_element_dtype.is_builtin()) {
                if (leading_dimension) {
                    // In the case of a leading dimension, first bake the offset into
                    // the data pointer, so that it's pointing at the right element
                    // for the collapsing of leading dimensions to work correctly.
                    *inout_data += offset;
                    offset = m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + sizeof(strided_dim_dtype_metadata),
                                    result_dtype, out_metadata,
                                    embedded_reference, current_i + 1, root_dt,
                                    true, inout_data, inout_dataref);
                } else {
                    offset += m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                    metadata + sizeof(strided_dim_dtype_metadata),
                                    result_dtype, out_metadata,
                                    embedded_reference, current_i + 1, root_dt,
                                    false, NULL, NULL);
                }
            }
            return offset;
        } else {
            // Produce the new offset data, stride, and size for the resulting array
            intptr_t offset = md->stride * start_index;
            out_md->stride = md->stride * index_stride;
            out_md->size = dimension_size;
            if (!m_element_dtype.is_builtin()) {
                const strided_dim_dtype *result_edtype = static_cast<const strided_dim_dtype *>(result_dtype.extended());
                offset += m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                metadata + sizeof(strided_dim_dtype_metadata),
                                result_edtype->m_element_dtype, out_metadata + sizeof(strided_dim_dtype_metadata),
                                embedded_reference, current_i + 1, root_dt,
                                false, NULL, NULL);
            }
            return offset;
        }
    }
}

dtype strided_dim_dtype::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    if (inout_metadata) {
        const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(*inout_metadata);
        // Bounds-checking of the index
        i0 = apply_single_index(i0, md->size, NULL);
        // Modify the metadata
        *inout_metadata += sizeof(strided_dim_dtype_metadata);
        // If requested, modify the data
        if (inout_data) {
            *inout_data += i0 * md->stride;
        }
    }
    return m_element_dtype;
}

dtype strided_dim_dtype::get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return dtype(this, true);
    } else {
        if (inout_metadata) {
            *inout_metadata += sizeof(strided_dim_dtype_metadata);
        }
        return m_element_dtype.get_dtype_at_dimension(inout_metadata, i - 1, total_ndim + 1);
    }
}

intptr_t strided_dim_dtype::get_dim_size(const char *DYND_UNUSED(data), const char *metadata) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    return md->size;
}

void strided_dim_dtype::get_shape(size_t i, intptr_t *out_shape) const
{
    // Adjust the current shape if necessary
    out_shape[i] = shape_signal_varying;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape);
    }
}

void strided_dim_dtype::get_shape(size_t i, intptr_t *out_shape, const char *metadata) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);

    out_shape[i] = md->size;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_shape(i+1, out_shape, metadata + sizeof(strided_dim_dtype_metadata));
    }
}

void strided_dim_dtype::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);

    out_strides[i] = md->stride;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_strides(i+1, out_strides, metadata + sizeof(strided_dim_dtype_metadata));
    }
}

intptr_t strided_dim_dtype::get_representative_stride(const char *metadata) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    return md->stride;
}

bool strided_dim_dtype::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == strided_dim_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

void strided_dim_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const
{
    throw runtime_error("strided_dim_dtype::get_single_compare_kernel is unimplemented"); 
}

bool strided_dim_dtype::operator==(const base_dtype& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != strided_dim_type_id) {
        return false;
    } else {
        const strided_dim_dtype *dt = static_cast<const strided_dim_dtype*>(&rhs);
        return m_element_dtype == dt->m_element_dtype;
    }
}

void strided_dim_dtype::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim == 0 || shape[0] < 0) {
        throw std::runtime_error("the strided_dim dtype requires a shape be specified for default construction");
    }
    size_t element_size = m_element_dtype.is_builtin() ? m_element_dtype.get_data_size()
                                                     : m_element_dtype.extended()->get_default_data_size(ndim-1, shape+1);

    strided_dim_dtype_metadata *md = reinterpret_cast<strided_dim_dtype_metadata *>(metadata);
    md->size = shape[0];
    if (shape[0] > 1) {
        md->stride = element_size;
    } else {
        md->stride = 0;
    }
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_default_construct(metadata + sizeof(strided_dim_dtype_metadata), ndim-1, shape+1);
    }
}

void strided_dim_dtype::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    const strided_dim_dtype_metadata *src_md = reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata);
    strided_dim_dtype_metadata *dst_md = reinterpret_cast<strided_dim_dtype_metadata *>(dst_metadata);
    dst_md->size = src_md->size;
    dst_md->stride = src_md->stride;
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_copy_construct(dst_metadata + sizeof(strided_dim_dtype_metadata),
                        src_metadata + sizeof(strided_dim_dtype_metadata), embedded_reference);
    }
}

void strided_dim_dtype::metadata_reset_buffers(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_reset_buffers(metadata + sizeof(strided_dim_dtype_metadata));
    }
}

void strided_dim_dtype::metadata_finalize_buffers(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_finalize_buffers(metadata + sizeof(strided_dim_dtype_metadata));
    }
}

void strided_dim_dtype::metadata_destruct(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_destruct(metadata + sizeof(strided_dim_dtype_metadata));
    }
}

void strided_dim_dtype::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    o << indent << "strided_dim metadata\n";
    o << indent << " stride: " << md->stride << "\n";
    o << indent << " size: " << md->size << "\n";
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_debug_print(metadata + sizeof(strided_dim_dtype_metadata), o, indent + " ");
    }
}

size_t strided_dim_dtype::get_iterdata_size(size_t ndim) const
{
    if (ndim == 0) {
        return 0;
    } else if (ndim == 1) {
        return sizeof(strided_dim_dtype_iterdata);
    } else {
        return m_element_dtype.get_iterdata_size(ndim - 1) + sizeof(strided_dim_dtype_iterdata);
    }
}

// Does one iterator increment for this dtype
static char *iterdata_incr(iterdata_common *iterdata, size_t level)
{
    strided_dim_dtype_iterdata *id = reinterpret_cast<strided_dim_dtype_iterdata *>(iterdata);
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
    strided_dim_dtype_iterdata *id = reinterpret_cast<strided_dim_dtype_iterdata *>(iterdata);
    if (ndim == 1) {
        id->data = data;
        return data;
    } else {
        id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
        return id->data;
    }
}

size_t strided_dim_dtype::iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, size_t ndim, const intptr_t* shape, dtype& out_uniform_dtype) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(*inout_metadata);
    *inout_metadata += sizeof(strided_dim_dtype_metadata);
    size_t inner_size = 0;
    if (ndim > 1) {
        // Place any inner iterdata earlier than the outer iterdata
        inner_size = m_element_dtype.extended()->iterdata_construct(iterdata, inout_metadata,
                        ndim - 1, shape + 1, out_uniform_dtype);
        iterdata = reinterpret_cast<iterdata_common *>(reinterpret_cast<char *>(iterdata) + inner_size);
    } else {
        out_uniform_dtype = m_element_dtype;
    }

    strided_dim_dtype_iterdata *id = reinterpret_cast<strided_dim_dtype_iterdata *>(iterdata);

    id->common.incr = &iterdata_incr;
    id->common.reset = &iterdata_reset;
    id->data = NULL;
    id->stride = md->stride;

    return inner_size + sizeof(strided_dim_dtype_iterdata);
}

size_t strided_dim_dtype::iterdata_destruct(iterdata_common *iterdata, size_t ndim) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        inner_size = m_element_dtype.extended()->iterdata_destruct(iterdata, ndim - 1);
    }
    // No dynamic data to free
    return inner_size + sizeof(strided_dim_dtype_iterdata);
}

size_t strided_dim_dtype::make_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        out->ensure_capacity(offset_out + sizeof(strided_assign_kernel_extra));
        const strided_dim_dtype_metadata *dst_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(dst_metadata);
        strided_assign_kernel_extra *e = out->get_at<strided_assign_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(&strided_assign_kernel_extra::single);
        e->base.destructor = strided_assign_kernel_extra::destruct;
        if (src_dt.get_undim() < dst_dt.get_undim()) {
            // If the src has fewer dimensions, broadcast it across this one
            e->size = dst_md->size;
            e->dst_stride = dst_md->stride;
            e->src_stride = 0;
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(strided_dim_dtype_metadata),
                            src_dt, src_metadata,
                            errmode, ectx);
        } else if (src_dt.get_type_id() == strided_dim_type_id) {
            // strided_dim -> strided_dim
            const strided_dim_dtype *src_sad = static_cast<const strided_dim_dtype *>(src_dt.extended());
            const strided_dim_dtype_metadata *src_md =
                        reinterpret_cast<const strided_dim_dtype_metadata *>(src_metadata);
            // Check for a broadcasting error
            if (src_md->size != 1 && dst_md->size != src_md->size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
            }
            e->size = dst_md->size;
            e->dst_stride = dst_md->stride;
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride = src_md->stride;
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(strided_dim_dtype_metadata),
                            src_sad->get_element_dtype(), src_metadata + sizeof(strided_dim_dtype_metadata),
                            errmode, ectx);
        } else if (src_dt.get_type_id() == fixed_dim_type_id) {
            // fixed_array -> strided_dim
            const fixed_dim_dtype *src_fad = static_cast<const fixed_dim_dtype *>(src_dt.extended());
            intptr_t src_size = src_fad->get_fixed_dim_size();
            // Check for a broadcasting error
            if (src_size != 1 && dst_md->size != src_size) {
                throw broadcast_error(dst_dt, dst_metadata, src_dt, src_metadata);
            }
            e->size = dst_md->size;
            e->dst_stride = dst_md->stride;
            // In DyND, the src stride is required to be zero for size-one dimensions,
            // so we don't have to check the size here.
            e->src_stride = src_fad->get_fixed_stride();
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata + sizeof(strided_dim_dtype_metadata),
                            src_fad->get_element_dtype(), src_metadata,
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

void strided_dim_dtype::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
    const char *child_metadata = metadata + sizeof(strided_dim_dtype_metadata);
    intptr_t stride = md->stride;
    for (intptr_t i = 0, i_end = md->size; i < i_end; ++i, data += stride) {
        callback(m_element_dtype, data, child_metadata, callback_data);
    }
}

void strided_dim_dtype::reorder_default_constructed_strides(char *dst_metadata,
                const dtype& src_dtype, const char *src_metadata) const
{
    // If the next dimension isn't also strided, then nothing can be reordered
    if (m_element_dtype.get_type_id() != strided_dim_type_id) {
        if (!m_element_dtype.is_builtin()) {
            dtype src_child_dtype = src_dtype.at_single(0, &src_metadata);
            m_element_dtype.extended()->reorder_default_constructed_strides(dst_metadata + sizeof(strided_dim_dtype_metadata),
                            src_child_dtype, src_metadata);
        }
        return;
    }

    // Find the total number of dimensions we might be reordering, then process
    // them all at once. This code handles a whole chain of strided_dim_dtype
    // instances at once.
    int ndim = 1;
    dtype last_dt = m_element_dtype;
    do {
        ++ndim;
        last_dt = static_cast<const strided_dim_dtype *>(last_dt.extended())->get_element_dtype();
    } while (last_dt.get_type_id() == strided_dim_type_id);

    // Get the representative strides from all the dimensions, and
    // advance the src_metadata pointer. Track if the
    // result is C-order in which case we can skip all sorting and manipulation
    dimvector strides(ndim);
    dtype last_src_dtype = src_dtype;
    intptr_t previous_stride = 0;
    bool c_order = true;
    for (int i = 0; i < ndim; ++i) {
        intptr_t stride = last_src_dtype.extended()->get_representative_stride(src_metadata);
        // To check for C-order, we skip over any 0-strides, and check if a stride ever gets
        // bigger instead of always getting smaller.
        if (stride != 0) {
            if (previous_stride != 0 && previous_stride < stride) {
                c_order = false;
            }
            previous_stride = stride;
        }
        strides[i] = stride;
        last_src_dtype = last_src_dtype.extended()->at_single(0, &src_metadata, NULL);
    }

    if (!c_order) {
        shortvector<int> axis_perm(ndim);
        strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
        strided_dim_dtype_metadata *md = reinterpret_cast<strided_dim_dtype_metadata *>(dst_metadata);
        intptr_t stride = md[ndim-1].stride;
        for (int i = 0; i < ndim; ++i) {
            int i_perm = axis_perm[i];
            strided_dim_dtype_metadata& i_md = md[i_perm];
            intptr_t dim_size = i_md.size;
            i_md.stride = dim_size > 1 ? stride : 0;
            stride *= dim_size;
        }
    }

    // Allow further subtypes to reorder their strides as well
    if (!last_dt.is_builtin()) {
        last_dt.extended()->reorder_default_constructed_strides(dst_metadata + ndim * sizeof(strided_dim_dtype_metadata),
                        last_src_dtype, src_metadata);
    }
}

void strided_dim_dtype::get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = m_ndobject_properties.empty() ? NULL : &m_ndobject_properties[0];
    *out_count = (int)m_ndobject_properties.size();
}

void strided_dim_dtype::get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    *out_functions = m_ndobject_functions.empty() ? NULL : &m_ndobject_functions[0];
    *out_count = (int)m_ndobject_functions.size();
}
