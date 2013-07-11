//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/gfunc/callable.hpp>
#include <dynd/dtypes/builtin_type_properties.hpp>

using namespace std;
using namespace dynd;

// Default destructor for the extended dtype does nothing
base_type::~base_type()
{
}

bool base_type::is_expression() const
{
    return false;
}

bool base_type::is_strided() const
{
    return false;
}

void base_type::process_strided(const char *DYND_UNUSED(metadata), const char *DYND_UNUSED(data),
                ndt::type& DYND_UNUSED(out_dt), const char *&DYND_UNUSED(out_origin),
                intptr_t& DYND_UNUSED(out_stride), intptr_t& DYND_UNUSED(out_dim_size)) const
{
    stringstream ss;
    ss << "dynd type " << ndt::type(this, true) << " is not strided, so process_strided should not be called";
    throw runtime_error(ss.str());
}


bool base_type::is_unique_data_owner(const char *DYND_UNUSED(metadata)) const
{
    return true;
}

void base_type::transform_child_types(type_transform_fn_t DYND_UNUSED(transform_fn), void *DYND_UNUSED(extra),
                ndt::type& out_transformed_type, bool& DYND_UNUSED(out_was_transformed)) const
{
    // Default to behavior with no child dtypes
    out_transformed_type = ndt::type(this, true);
}

ndt::type base_type::get_canonical_type() const
{
    // Default to no transformation of the dtype
    return ndt::type(this, true);
}

ndt::type base_type::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices),
                size_t current_i, const ndt::type& DYND_UNUSED(root_dt), bool DYND_UNUSED(leading_dimension)) const
{
    // Default to scalar behavior
    if (nindices == 0) {
        return ndt::type(this, true);
    } else {
        throw too_many_indices(ndt::type(this, true), current_i + nindices, current_i);
    }
}

intptr_t base_type::apply_linear_index(size_t nindices, const irange *DYND_UNUSED(indices), const char *metadata,
                const ndt::type& DYND_UNUSED(result_dtype), char *out_metadata,
                memory_block_data *embedded_reference,
                size_t current_i, const ndt::type& DYND_UNUSED(root_dt),
                bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    // Default to scalar behavior
    if (nindices == 0) {
        // Copy any metadata verbatim
        metadata_copy_construct(out_metadata, metadata, embedded_reference);
        return 0;
    } else {
        throw too_many_indices(ndt::type(this, true), current_i + nindices, current_i);
    }
}

ndt::type base_type::at_single(intptr_t DYND_UNUSED(i0), const char **DYND_UNUSED(inout_metadata),
                const char **DYND_UNUSED(inout_data)) const
{
    // Default to scalar behavior
    throw too_many_indices(ndt::type(this, true), 1, 0);
}

ndt::type base_type::get_type_at_dimension(char **DYND_UNUSED(inout_metadata), size_t i, size_t total_ndim) const
{
    // Default to heterogeneous dimension/scalar behavior
    if (i == 0) {
        return ndt::type(this, true);
    } else {
        throw too_many_indices(ndt::type(this, true), total_ndim + i, total_ndim);
    }
}

void base_type::get_shape(size_t DYND_UNUSED(ndim), size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape),
                const char *DYND_UNUSED(metadata)) const
{
    // Default to scalar behavior
    stringstream ss;
    ss << "requested too many dimensions from type " << ndt::type(this, true);
    throw runtime_error(ss.str());
}

void base_type::get_strides(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_strides),
                const char *DYND_UNUSED(metadata)) const
{
    // Default to scalar behavior
}

axis_order_classification_t base_type::classify_axis_order(
                const char *DYND_UNUSED(metadata)) const
{
    // Scalar types have no axis order
    return axis_order_none;
}


bool base_type::is_lossless_assignment(const ndt::type& dst_dt,
                const ndt::type& src_dt) const
{
    // Default to just an equality check
    return dst_dt == src_dt;
}

size_t base_type::get_default_data_size(size_t DYND_UNUSED(ndim),
                const intptr_t *DYND_UNUSED(shape)) const
{
    return get_data_size();
}

// TODO: Make this a pure virtual function eventually
void base_type::metadata_default_construct(char *DYND_UNUSED(metadata),
                size_t DYND_UNUSED(ndim),
                const intptr_t* DYND_UNUSED(shape)) const
{
    stringstream ss;
    ss << "TODO: metadata_default_construct for " << ndt::type(this, true) << " is not implemented";
    throw std::runtime_error(ss.str());
}

void base_type::metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    stringstream ss;
    ss << "TODO: metadata_copy_construct for " << ndt::type(this, true) << " is not implemented";
    throw std::runtime_error(ss.str());
}

void base_type::metadata_reset_buffers(char *DYND_UNUSED(metadata)) const
{
    // By default there are no buffers to reset
}

void base_type::metadata_finalize_buffers(char *DYND_UNUSED(metadata)) const
{
    // By default there are no buffers to finalize
}

// TODO: Make this a pure virtual function eventually
void base_type::metadata_destruct(char *DYND_UNUSED(metadata)) const
{
    stringstream ss;
    ss << "TODO: metadata_destruct for " << ndt::type(this, true) << " is not implemented";
    throw std::runtime_error(ss.str());
}

// TODO: Make this a pure virtual function eventually
void base_type::metadata_debug_print(const char *DYND_UNUSED(metadata),
                std::ostream& DYND_UNUSED(o),
                const std::string& DYND_UNUSED(indent)) const
{
    stringstream ss;
    ss << "TODO: metadata_debug_print for " << ndt::type(this, true) << " is not implemented";
    throw std::runtime_error(ss.str());
}

void base_type::data_destruct(const char *DYND_UNUSED(metadata),
                char *DYND_UNUSED(data)) const
{
    stringstream ss;
    ss << "TODO: data_destruct for " << ndt::type(this, true) << " is not implemented";
    throw runtime_error(ss.str());
}

void base_type::data_destruct_strided(const char *DYND_UNUSED(metadata),
                char *DYND_UNUSED(data), intptr_t DYND_UNUSED(stride),
                size_t DYND_UNUSED(count)) const
{
    stringstream ss;
    ss << "TODO: data_destruct_strided for " << ndt::type(this, true) << " is not implemented";
    throw runtime_error(ss.str());
}

size_t base_type::get_iterdata_size(size_t DYND_UNUSED(ndim)) const
{
    stringstream ss;
    ss << "get_iterdata_size: dynd type " << ndt::type(this, true) << " is not uniformly iterable";
    throw std::runtime_error(ss.str());
}

size_t base_type::iterdata_construct(iterdata_common *DYND_UNUSED(iterdata), const char **DYND_UNUSED(inout_metadata),
                size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape), ndt::type& DYND_UNUSED(out_uniform_dtype)) const
{
    stringstream ss;
    ss << "iterdata_default_construct: dynd type " << ndt::type(this, true) << " is not uniformly iterable";
    throw std::runtime_error(ss.str());
}

size_t base_type::iterdata_destruct(iterdata_common *DYND_UNUSED(iterdata), size_t DYND_UNUSED(ndim)) const
{
    stringstream ss;
    ss << "iterdata_destruct: dynd type " << ndt::type(this, true) << " is not uniformly iterable";
    throw std::runtime_error(ss.str());
}

size_t base_type::make_assignment_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const ndt::type& dst_dt, const char *DYND_UNUSED(dst_metadata),
                const ndt::type& src_dt, const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), assign_error_mode DYND_UNUSED(errmode),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "make_assignment_kernel has not been implemented for ";
    if (this == dst_dt.extended()) {
        ss << dst_dt;
    } else {
        ss << src_dt;
    }
    throw std::runtime_error(ss.str());
}

size_t base_type::make_comparison_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const ndt::type& src0_dt, const char *DYND_UNUSED(src0_metadata),
                const ndt::type& src1_dt, const char *DYND_UNUSED(src1_metadata),
                comparison_type_t DYND_UNUSED(comptype),
                const eval::eval_context *DYND_UNUSED(ectx)) const
{
    stringstream ss;
    ss << "make_comparison_kernel has not been implemented for ";
    if (this == src0_dt.extended()) {
        ss << src0_dt;
    } else {
        ss << src1_dt;
    }
    throw std::runtime_error(ss.str());
}

void base_type::foreach_leading(char *DYND_UNUSED(data), const char *DYND_UNUSED(metadata),
                foreach_fn_t DYND_UNUSED(callback), void *DYND_UNUSED(callback_data)) const
{
    // Default to scalar behavior
    stringstream ss;
    ss << "dynd type " << ndt::type(this, true) << " is a scalar, foreach_leading cannot process";
    throw std::runtime_error(ss.str());
}

void base_type::get_scalar_properties_and_functions(
                std::vector<std::pair<std::string, gfunc::callable> >& out_properties,
                std::vector<std::pair<std::string, gfunc::callable> >& out_functions) const
{
    // This copies properties from the first non-array data type dimension to
    // the requested vectors. It is for use by array data types, which by convention
    // expose the properties from the first non-array data types, and possibly add
    // additional properties of their own.
    size_t ndim = get_undim();
    size_t properties_count = 0, functions_count = 0;
    const std::pair<std::string, gfunc::callable> *properties = NULL, *functions = NULL;
    if (ndim == 0) {
        get_dynamic_array_properties(&properties, &properties_count);
        get_dynamic_array_functions(&functions, &functions_count);
    } else {
        ndt::type dt = get_type_at_dimension(NULL, ndim);
        if (!dt.is_builtin()) {
            dt.extended()->get_dynamic_array_properties(&properties, &properties_count);
            dt.extended()->get_dynamic_array_functions(&functions, &functions_count);
        } else {
            get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties, &properties_count);
        }
    }
    out_properties.resize(properties_count);
    for (size_t i = 0; i < properties_count; ++i) {
        out_properties[i] = properties[i];
    }
    out_functions.resize(functions_count);
    for (size_t i = 0; i < functions_count; ++i) {
        out_functions[i] = functions[i];
    }
}

void base_type::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    // Default to no properties
    *out_properties = NULL;
    *out_count = 0;
}

void base_type::get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    // Default to no functions
    *out_functions = NULL;
    *out_count = 0;
}

void base_type::get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    // Default to no properties
    *out_properties = NULL;
    *out_count = 0;
}

void base_type::get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const
{
    // Default to no functions
    *out_functions = NULL;
    *out_count = 0;
}

size_t base_type::get_elwise_property_index(const std::string& property_name) const
{
    std::stringstream ss;
    ss << "the dtype " << ndt::type(this, true);
    ss << " doesn't have a property \"" << property_name << "\"";
    throw std::runtime_error(ss.str());
}

ndt::type base_type::get_elwise_property_type(size_t DYND_UNUSED(elwise_property_index),
            bool& DYND_UNUSED(out_readable), bool& DYND_UNUSED(out_writable)) const
{
    throw std::runtime_error("get_elwise_property_type: this dtype does not have any properties");
}

size_t base_type::make_elwise_property_getter_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata),
                const char *DYND_UNUSED(src_metadata),
                size_t DYND_UNUSED(src_elwise_property_index),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    std::stringstream ss;
    ss << "the dtype " << ndt::type(this, true);
    ss << " doesn't have any readable properties";
    throw std::runtime_error(ss.str());
}

size_t base_type::make_elwise_property_setter_kernel(
                hierarchical_kernel *DYND_UNUSED(out), size_t DYND_UNUSED(offset_out),
                const char *DYND_UNUSED(dst_metadata),
                size_t DYND_UNUSED(dst_elwise_property_index),
                const char *DYND_UNUSED(src_metadata),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx)) const
{
    std::stringstream ss;
    ss << "the dtype " << ndt::type(this, true);
    ss << " doesn't have any writable properties";
    throw std::runtime_error(ss.str());
}
