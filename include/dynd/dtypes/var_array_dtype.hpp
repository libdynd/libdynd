//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_DTYPE_HPP_
#define _DYND__ARRAY_DTYPE_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

struct var_array_dtype_metadata {
    /**
     * A reference to the memory block which contains the array's data.
     */
    memory_block_data *blockref;
    intptr_t stride;
    /* Each pointed-to destination is offset by this amount */
    intptr_t offset;
};

struct var_array_dtype_data {
    char *begin;
    size_t size;
};

class var_array_dtype : public base_dtype {
    dtype m_element_dtype;
    std::vector<std::pair<std::string, gfunc::callable> > m_ndobject_properties, m_ndobject_functions;
public:
    var_array_dtype(const dtype& element_dtype);

    virtual ~var_array_dtype();

    size_t get_default_data_size(int DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return sizeof(var_array_dtype_data);
    }


    const dtype& get_element_dtype() const {
        return m_element_dtype;
    }

    /** Alignment of the data being pointed to. */
    size_t get_data_alignment() const {
        return m_element_dtype.get_alignment();
    }

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    // This is about the native storage, buffering code needs to check whether
    // the value_dtype is an object type separately.
    dtype_memory_management_t get_memory_management() const {
        return blockref_memory_management;
    }

    bool is_scalar() const;
    bool is_uniform_dim() const;
    bool is_expression() const;
    bool is_unique_data_owner(const char *metadata) const;
    void transform_child_dtypes(dtype_transform_fn_t transform_fn, const void *extra,
                    dtype& out_transformed_dtype, bool& out_was_transformed) const;
    dtype get_canonical_dtype() const;
    bool is_strided() const;
    void process_strided(const char *metadata, const char *data,
                    dtype& out_dt, const char *&out_origin,
                    intptr_t& out_stride, intptr_t& out_dim_size) const;

    dtype apply_linear_index(int nindices, const irange *indices,
                int current_i, const dtype& root_dt, bool leading_dimension) const;
    intptr_t apply_linear_index(int nindices, const irange *indices, const char *metadata,
                    const dtype& result_dtype, char *out_metadata,
                    memory_block_data *embedded_reference,
                    int current_i, const dtype& root_dt,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    dtype at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const;

    dtype get_dtype_at_dimension(char **inout_metadata, size_t i, size_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *data, const char *metadata) const;
    void get_shape(size_t i, intptr_t *out_shape) const;
    void get_shape(size_t i, intptr_t *out_shape, const char *metadata) const;
    void get_strides(size_t i, intptr_t *out_strides, const char *metadata) const;
    intptr_t get_representative_stride(const char *metadata) const;

    bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt) const;

    void get_single_compare_kernel(kernel_instance<compare_operations_t>& out_kernel) const;

    void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_pair_t>& out_kernel) const;

    bool operator==(const base_dtype& rhs) const;

    size_t get_metadata_size() const;
    void metadata_default_construct(char *metadata, int ndim, const intptr_t* shape) const;
    void metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const;
    void metadata_reset_buffers(char *metadata) const;
    void metadata_finalize_buffers(char *metadata) const;
    void metadata_destruct(char *metadata) const;
    void metadata_debug_print(const char *metadata, std::ostream& o, const std::string& indent) const;

    size_t get_iterdata_size(int ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_metadata, int ndim, const intptr_t* shape, dtype& out_uniform_dtype) const;
    size_t iterdata_destruct(iterdata_common *iterdata, int ndim) const;

    void foreach_leading(char *data, const char *metadata, foreach_fn_t callback, void *callback_data) const;
    
    void reorder_default_constructed_strides(char *dst_metadata, const dtype& src_dtype, const char *src_metadata) const;

    void get_dynamic_ndobject_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_ndobject_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;
};

inline dtype make_var_array_dtype(const dtype& element_dtype) {
    return dtype(new var_array_dtype(element_dtype), false);
}

} // namespace dynd

#endif // _DYND__ARRAY_DTYPE_HPP_
