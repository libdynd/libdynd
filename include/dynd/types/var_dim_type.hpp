//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_TYPES_VAR_DIM_TYPE_HPP
#define DYND_TYPES_VAR_DIM_TYPE_HPP

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

struct var_dim_type_arrmeta {
    /**
     * A reference to the memory block which contains the array's data.
     */
    memory_block_data *blockref;
    intptr_t stride;
    /* Each pointed-to destination is offset by this amount */
    intptr_t offset;
};

struct var_dim_type_data {
    char *begin;
    size_t size;
};

class var_dim_type : public base_dim_type {
    std::vector<std::pair<std::string, gfunc::callable> > m_array_properties, m_array_functions;
public:
    var_dim_type(const ndt::type& element_tp);

    virtual ~var_dim_type();

    size_t get_default_data_size(intptr_t DYND_UNUSED(ndim), const intptr_t *DYND_UNUSED(shape)) const {
        return sizeof(var_dim_type_data);
    }


    /** Alignment of the data being pointed to. */
    size_t get_target_alignment() const {
        return m_element_tp.get_data_alignment();
    }

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, void *extra,
                    ndt::type& out_transformed_tp, bool& out_was_transformed) const;
    ndt::type get_canonical_type() const;

    ndt::type apply_linear_index(intptr_t nindices, const irange *indices,
                size_t current_i, const ndt::type& root_tp, bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                    const ndt::type& result_tp, char *out_arrmeta,
                    memory_block_data *embedded_reference,
                    size_t current_i, const ndt::type& root_tp,
                    bool leading_dimension, char **inout_data,
                    memory_block_data **inout_dataref) const;
    ndt::type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    ndt::type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
        void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;
    void get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const;

    axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim, const intptr_t* shape) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                    memory_block_data *embedded_reference) const;

    size_t get_iterdata_size(intptr_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim, const intptr_t* shape, ndt::type& out_uniform_tp) const;
    size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    void foreach_leading(const char *arrmeta, char *data,
                         foreach_fn_t callback, void *callback_data) const;

    void get_dynamic_type_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_properties(
                    const std::pair<std::string, gfunc::callable> **out_properties,
                    size_t *out_count) const;
    void get_dynamic_array_functions(
                    const std::pair<std::string, gfunc::callable> **out_functions,
                    size_t *out_count) const;
};

namespace ndt {
    type make_var_dim(const type& element_tp);

    /**
     * A helper function for reserving initial space in a var dim element.
     * This requires that the element being created (at `data`) is NULL, and
     * it allocates `count` elements to start of the var element.
     *
     * \param tp  This must be a var_dim type.
     * \param arrmeta  Arrmeta for `tp`.
     * \param data  Array data for the `tp`, `arrmeta` pair, this
     *              is written to.
     * \param count  The number of elements to start off with.
     */
    void var_dim_element_initialize(const type& tp,
            const char *arrmeta, char *data, intptr_t count);

    /**
     * A helper function for resizing the allocated space in a var dim
     * element. The element's `begin` pointer and
     * `size` count must not have been modified since the last
     * initialize/resize operation. If the element has not been
     * initialized previously, it is initialized to the requested count.
     *
     * \param tp  This must be a var_dim type.
     * \param arrmeta  Arrmeta for `tp`.
     * \param data  Array data for the `tp`, `arrmeta` pair, this
     *              is written to.
     * \param count  The number of elements to resize to.
     */
    void var_dim_element_resize(const type& tp,
            const char *arrmeta, char *data, intptr_t count);
} // namespace ndt

} // namespace dynd

#endif // DYND_TYPES_VAR_DIM_TYPE_HPP
