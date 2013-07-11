//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/fixed_dim_type.hpp>
#include <dynd/dtypes/strided_dim_type.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/shortvector.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;

fixed_dim_type::fixed_dim_type(size_t dimension_size, const ndt::type& element_dtype)
    : base_uniform_dim_type(fixed_dim_type_id, element_dtype, 0, element_dtype.get_data_alignment(),
                    0, type_flag_none),
            m_dim_size(dimension_size)
{
    size_t child_element_size = element_dtype.get_data_size();
    if (child_element_size == 0) {
        stringstream ss;
        ss << "Cannot create fixed_dim dtype with element type " << element_dtype;
        ss << ", as it does not have a fixed size";
        throw runtime_error(ss.str());
    }
    m_stride = m_dim_size > 1 ? element_dtype.get_data_size() : 0;
    m_members.data_size = m_stride * (m_dim_size-1) + child_element_size;
    // Propagate the operand flags from the element
    m_members.flags |= (element_dtype.get_flags()&dtype_flags_operand_inherited);

    // Copy ndobject properties and functions from the first non-array dimension
    get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

fixed_dim_type::fixed_dim_type(size_t dimension_size, const ndt::type& element_dtype, intptr_t stride)
    : base_uniform_dim_type(fixed_dim_type_id, element_dtype, 0, element_dtype.get_data_alignment(),
                    0, type_flag_none),
            m_stride(stride), m_dim_size(dimension_size)
{
    size_t child_element_size = element_dtype.get_data_size();
    if (child_element_size == 0) {
        stringstream ss;
        ss << "Cannot create fixed_dim dtype with element type " << element_dtype;
        ss << ", as it does not have a fixed size";
        throw runtime_error(ss.str());
    }
    if (dimension_size <= 1 && stride != 0) {
        stringstream ss;
        ss << "Cannot create fixed_dim dtype with size " << dimension_size;
        ss << " and stride " << stride << ", as the stride must be zero when the dimension size is 1";
        throw runtime_error(ss.str());
    }
    if (dimension_size > 1 && stride == 0) {
        stringstream ss;
        ss << "Cannot create fixed_dim dtype with size " << dimension_size;
        ss << " and stride 0, as the stride must be non-zero when the dimension size is > 1";
        throw runtime_error(ss.str());
    }
    m_members.data_size = m_stride * (m_dim_size-1) + child_element_size;
    // Propagate the zeroinit flag from the element
    m_members.flags |= (element_dtype.get_flags()&type_flag_zeroinit);

    // Copy ndobject properties and functions from the first non-array dimension
    get_scalar_properties_and_functions(m_array_properties, m_array_functions);
}

fixed_dim_type::~fixed_dim_type()
{
}

void fixed_dim_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    size_t stride = m_stride;
    o << "[";
    for (size_t i = 0, i_end = m_dim_size; i != i_end; ++i, data += stride) {
        m_element_dtype.print_data(o, metadata, data);
        if (i != i_end - 1) {
            o << ", ";
        }
    }
    o << "]";
}

void fixed_dim_type::print_dtype(std::ostream& o) const
{
    o << "fixed_dim<";
    o << m_dim_size;
    if ((size_t)m_stride != m_element_dtype.get_data_size()) {
        o << ", stride=" << m_stride;
    }
    o << ", " << m_element_dtype;
    o << ">";
}

bool fixed_dim_type::is_expression() const
{
    return m_element_dtype.is_expression();
}

bool fixed_dim_type::is_unique_data_owner(const char *metadata) const
{
    if (m_element_dtype.is_builtin()) {
        return true;
    } else {
        return m_element_dtype.extended()->is_unique_data_owner(metadata);
    }
}

void fixed_dim_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_dtype, bool& out_was_transformed) const
{
    ndt::type tmp_dtype;
    bool was_transformed = false;
    transform_fn(m_element_dtype, extra, tmp_dtype, was_transformed);
    if (was_transformed) {
        if (tmp_dtype.get_data_size() != 0) {
            out_transformed_dtype = ndt::type(new fixed_dim_type(m_dim_size, tmp_dtype), false);
        } else {
            out_transformed_dtype = ndt::type(new strided_dim_type(tmp_dtype), false);
        }
        out_was_transformed = true;
    } else {
        out_transformed_dtype = ndt::type(this, true);
    }
}


ndt::type fixed_dim_type::get_canonical_type() const
{
    ndt::type canonical_element_dt = m_element_dtype.get_canonical_type();
    // The transformed dtype may no longer have a fixed size, so check whether
    // we have to switch to the more flexible strided_dim_type
    if (canonical_element_dt.get_data_size() != 0) {
        return ndt::type(new fixed_dim_type(m_dim_size, canonical_element_dt), false);
    } else {
        return ndt::type(new strided_dim_type(canonical_element_dt), false);
    }
}

bool fixed_dim_type::is_strided() const
{
    return true;
}

void fixed_dim_type::process_strided(const char *DYND_UNUSED(metadata), const char *data,
                ndt::type& out_dt, const char *&out_origin,
                intptr_t& out_stride, intptr_t& out_dim_size) const
{
    out_dt = m_element_dtype;
    out_origin = data;
    out_stride = m_stride;
    out_dim_size = m_dim_size;
}

ndt::type fixed_dim_type::apply_linear_index(size_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_dt, bool leading_dimension) const
{
    if (nindices == 0) {
        return ndt::type(this, true);
    } else if (nindices == 1) {
        if (indices->step() == 0) {
            return m_element_dtype;
        } else {
            return ndt::type(new strided_dim_type(m_element_dtype), false);
        }
    } else {
        if (indices->step() == 0) {
            return m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, leading_dimension);
        } else {
            return ndt::type(new strided_dim_type(m_element_dtype.apply_linear_index(nindices-1, indices+1,
                            current_i+1, root_dt, false)), false);
        }
    }
}

intptr_t fixed_dim_type::apply_linear_index(size_t nindices, const irange *indices, const char *metadata,
                const ndt::type& result_dtype, char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& root_dt,
                bool leading_dimension, char **inout_data,
                memory_block_data **inout_dataref) const
{
    if (nindices == 0) {
        // If there are no more indices, copy the metadata verbatim
        metadata_copy_construct(out_metadata, metadata, embedded_reference);
        return 0;
    } else {
        bool remove_dimension;
        intptr_t start_index, index_stride, dimension_size;
        apply_single_linear_index(*indices, m_dim_size, current_i, &root_dt,
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
            strided_dim_type_metadata *out_md = reinterpret_cast<strided_dim_type_metadata *>(out_metadata);
            // Produce the new offset data, stride, and size for the resulting array,
            // which is now a strided_dim instead of a fixed_dim
            intptr_t offset = m_stride * start_index;
            out_md->stride = m_stride * index_stride;
            out_md->size = dimension_size;
            if (!m_element_dtype.is_builtin()) {
                const strided_dim_type *result_edtype = static_cast<const strided_dim_type *>(result_dtype.extended());
                offset += m_element_dtype.extended()->apply_linear_index(nindices - 1, indices + 1,
                                metadata,
                                result_edtype->get_element_type(),
                                out_metadata + sizeof(strided_dim_type_metadata),
                                embedded_reference, current_i + 1, root_dt,
                                false, NULL, NULL);
            }
            return offset;
        }
    }
}

ndt::type fixed_dim_type::at_single(intptr_t i0,
                const char **DYND_UNUSED(inout_metadata), const char **inout_data) const
{
    // Bounds-checking of the index
    i0 = apply_single_index(i0, m_dim_size, NULL);
    // The fixed_dim dtype has no metadata
    // If requested, modify the data
    if (inout_data) {
        *inout_data += i0 * m_stride;
    }
    return m_element_dtype;
}

ndt::type fixed_dim_type::get_type_at_dimension(char **inout_metadata, size_t i, size_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    } else {
        return m_element_dtype.get_type_at_dimension(inout_metadata, i - 1, total_ndim + 1);
    }
}

intptr_t fixed_dim_type::get_dim_size(const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data)) const
{
    return m_dim_size;
}

void fixed_dim_type::get_shape(size_t ndim, size_t i, intptr_t *out_shape, const char *metadata) const
{
    out_shape[i] = m_dim_size;

    // Process the later shape values
    if (i+1 < ndim) {
        if (!m_element_dtype.is_builtin()) {
            m_element_dtype.extended()->get_shape(ndim, i+1, out_shape, metadata);
        } else {
            stringstream ss;
            ss << "requested too many dimensions from type " << ndt::type(this, true);
            throw runtime_error(ss.str());
        }
    }
}

void fixed_dim_type::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    out_strides[i] = m_stride;

    // Process the later shape values
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->get_strides(i+1, out_strides, metadata);
    }
}

axis_order_classification_t fixed_dim_type::classify_axis_order(const char *metadata) const
{
    if (m_element_dtype.get_undim() > 0) {
        if (m_stride != 0) {
            // Call the helper function to do the classification
            return classify_strided_axis_order(m_stride, m_element_dtype,
                            metadata);
        } else {
            // Use the classification of the element dtype
            return m_element_dtype.extended()->classify_axis_order(
                            metadata);
        }
    } else {
        return axis_order_none;
    }
}

bool fixed_dim_type::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const
{
    if (dst_dt.extended() == this) {
        if (src_dt.extended() == this) {
            return true;
        } else if (src_dt.get_type_id() == fixed_dim_type_id) {
            return *dst_dt.extended() == *src_dt.extended();
        }
    }

    return false;
}

bool fixed_dim_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != fixed_dim_type_id) {
        return false;
    } else {
        const fixed_dim_type *dt = static_cast<const fixed_dim_type*>(&rhs);
        return m_element_dtype == dt->m_element_dtype &&
                m_dim_size == dt->m_dim_size &&
                m_stride == dt->m_stride;
    }
}

void fixed_dim_type::metadata_default_construct(char *metadata, size_t ndim, const intptr_t* shape) const
{
    // Validate that the shape is ok
    if (ndim > 0) {
        if (shape[0] >= 0 && (size_t)shape[0] != m_dim_size) {
            stringstream ss;
            ss << "Cannot construct dynd object of dtype " << ndt::type(this, true);
            ss << " with dimension size " << shape[0] << ", the size must be " << m_dim_size;
            throw runtime_error(ss.str());
        }
    }

    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_default_construct(metadata, ndim ? (ndim-1) : 0, shape+1);
    }
}

void fixed_dim_type::metadata_copy_construct(
                char *dst_metadata, const char *src_metadata,
                memory_block_data *embedded_reference) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_copy_construct(dst_metadata, src_metadata, embedded_reference);
    }
}

size_t fixed_dim_type::metadata_copy_construct_onedim(
                char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata),
                memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    // No metadata to copy for fixed_dim
    return 0;
}

void fixed_dim_type::metadata_reset_buffers(char *metadata) const
{
    if (m_element_dtype.get_metadata_size() > 0) {
        m_element_dtype.extended()->metadata_reset_buffers(metadata);
    }
}

void fixed_dim_type::metadata_finalize_buffers(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_finalize_buffers(metadata);
    }
}

void fixed_dim_type::metadata_destruct(char *metadata) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_destruct(metadata);
    }
}

void fixed_dim_type::metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const
{
    if (!m_element_dtype.is_builtin()) {
        m_element_dtype.extended()->metadata_debug_print(metadata, o, indent);
    }
}

size_t fixed_dim_type::get_iterdata_size(size_t ndim) const
{
    if (ndim == 0) {
        return 0;
    } else if (ndim == 1) {
        return sizeof(fixed_dim_type_iterdata);
    } else {
        return m_element_dtype.get_iterdata_size(ndim - 1) + sizeof(fixed_dim_type_iterdata);
    }
}

// Does one iterator increment for this dtype
static char *iterdata_incr(iterdata_common *iterdata, size_t level)
{
    fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);
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
    fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);
    if (ndim == 1) {
        id->data = data;
        return data;
    } else {
        id->data = (id + 1)->common.reset(&(id + 1)->common, data, ndim - 1);
        return id->data;
    }
}

size_t fixed_dim_type::iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, size_t ndim, const intptr_t* shape, ndt::type& out_uniform_dtype) const
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

    if (m_dim_size != 1 && (size_t)shape[0] != m_dim_size) {
        stringstream ss;
        ss << "Cannot construct dynd iterator of dtype " << ndt::type(this, true);
        ss << " with dimension size " << shape[0] << ", the size must be " << m_dim_size;
        throw runtime_error(ss.str());
    }

    fixed_dim_type_iterdata *id = reinterpret_cast<fixed_dim_type_iterdata *>(iterdata);

    id->common.incr = &iterdata_incr;
    id->common.reset = &iterdata_reset;
    id->data = NULL;
    id->stride = m_stride;

    return inner_size + sizeof(fixed_dim_type_iterdata);
}

size_t fixed_dim_type::iterdata_destruct(iterdata_common *iterdata, size_t ndim) const
{
    size_t inner_size = 0;
    if (ndim > 1) {
        inner_size = m_element_dtype.extended()->iterdata_destruct(iterdata, ndim - 1);
    }
    // No dynamic data to free
    return inner_size + sizeof(fixed_dim_type_iterdata);
}

void fixed_dim_type::data_destruct(const char *metadata, char *data) const
{
    m_element_dtype.extended()->data_destruct_strided(
                    metadata, data, m_stride, m_dim_size);
}

void fixed_dim_type::data_destruct_strided(const char *metadata, char *data,
                intptr_t stride, size_t count) const
{
    intptr_t child_stride = m_stride;
    size_t child_size = m_dim_size;

    for (size_t i = 0; i != count; ++i, data += stride) {
        m_element_dtype.extended()->data_destruct_strided(
                        metadata, data, child_stride, child_size);
    }
}

size_t fixed_dim_type::make_assignment_kernel(
                hierarchical_kernel *out, size_t offset_out,
                const ndt::type& dst_dt, const char *dst_metadata,
                const ndt::type& src_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    if (this == dst_dt.extended()) {
        out->ensure_capacity(offset_out + sizeof(strided_assign_kernel_extra));
        strided_assign_kernel_extra *e = out->get_at<strided_assign_kernel_extra>(offset_out);
        switch (kernreq) {
            case kernel_request_single:
                e->base.set_function<unary_single_operation_t>(&strided_assign_kernel_extra::single);
                break;
            case kernel_request_strided:
                e->base.set_function<unary_strided_operation_t>(&strided_assign_kernel_extra::strided);
                break;
            default: {
                stringstream ss;
                ss << "strided_dim_type::make_assignment_kernel: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
        }
        e->base.destructor = strided_assign_kernel_extra::destruct;
        if (src_dt.get_undim() < dst_dt.get_undim()) {
            // If the src has fewer dimensions, broadcast it across this one
            e->size = get_fixed_dim_size();
            e->dst_stride = get_fixed_stride();
            e->src_stride = 0;
            return ::make_assignment_kernel(out, offset_out + sizeof(strided_assign_kernel_extra),
                            m_element_dtype, dst_metadata,
                            src_dt, src_metadata,
                            kernel_request_strided, errmode, ectx);
        } else if (src_dt.get_type_id() == fixed_dim_type_id) {
            // fixed_array -> strided_dim
            const fixed_dim_type *src_fad = static_cast<const fixed_dim_type *>(src_dt.extended());
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
                            src_fad->get_element_type(), src_metadata,
                            kernel_request_strided, errmode, ectx);
        } else if (src_dt.get_type_id() == strided_dim_type_id) {
            // strided_dim -> strided_dim
            const strided_dim_type *src_sad = static_cast<const strided_dim_type *>(src_dt.extended());
            const strided_dim_type_metadata *src_md =
                        reinterpret_cast<const strided_dim_type_metadata *>(src_metadata);
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
                            src_sad->get_element_type(), src_metadata + sizeof(strided_dim_type_metadata),
                            kernel_request_strided, errmode, ectx);
        } else if (!src_dt.is_builtin()) {
            // Give the src dtype a chance to make a kernel
            return src_dt.extended()->make_assignment_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_dt, src_metadata,
                            kernreq, errmode, ectx);
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

void fixed_dim_type::foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const
{
    intptr_t stride = m_stride;
    for (intptr_t i = 0, i_end = m_dim_size; i < i_end; ++i, data += stride) {
        callback(m_element_dtype, data, metadata, callback_data);
    }
}

ndt::type dynd::make_fixed_dim_type(size_t ndim, const intptr_t *shape,
                const ndt::type& uniform_dtype, const int *axis_perm)
{
    if (axis_perm == NULL) {
        // Build a C-order fixed array dtype
        ndt::type result = uniform_dtype;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = make_fixed_dim_type(shape[i], result);
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
        ndt::type result = uniform_dtype;
        for (ptrdiff_t i = (ptrdiff_t)ndim-1; i >= 0; --i) {
            result = make_fixed_dim_type(shape[i], result, strides[i]);
        }
        return result;
    }
}

static size_t get_fixed_dim_size(const ndt::type& dt) {
    const fixed_dim_type *d = static_cast<const fixed_dim_type *>(dt.extended());
    return d->get_fixed_dim_size();
}

static intptr_t get_fixed_dim_stride(const ndt::type& dt) {
    const fixed_dim_type *d = static_cast<const fixed_dim_type *>(dt.extended());
    return d->get_fixed_stride();
}

static ndt::type get_element_type(const ndt::type& dt) {
    const fixed_dim_type *d = static_cast<const fixed_dim_type *>(dt.extended());
    return d->get_element_type();
}

static pair<string, gfunc::callable> fixed_dim_type_properties[] = {
    pair<string, gfunc::callable>("fixed_dim_size", gfunc::make_callable(&get_fixed_dim_size, "self")),
    pair<string, gfunc::callable>("fixed_dim_stride", gfunc::make_callable(&get_fixed_dim_stride, "self")),
    pair<string, gfunc::callable>("element_type", gfunc::make_callable(&get_element_type, "self"))
};

void fixed_dim_type::get_dynamic_dtype_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = fixed_dim_type_properties;
    *out_count = sizeof(fixed_dim_type_properties) / sizeof(fixed_dim_type_properties[0]);
}

void fixed_dim_type::get_dynamic_array_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    *out_properties = m_array_properties.empty() ? NULL : &m_array_properties[0];
    *out_count = (int)m_array_properties.size();
}

void fixed_dim_type::get_dynamic_array_functions(
                const std::pair<std::string, gfunc::callable> **out_functions,
                size_t *out_count) const
{
    *out_functions = m_array_functions.empty() ? NULL : &m_array_functions[0];
    *out_count = (int)m_array_functions.size();
}
