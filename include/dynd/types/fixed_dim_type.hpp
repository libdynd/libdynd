//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>

namespace dynd {

// fixed_dim (redundantly) uses the same arrmeta as strided_dim
typedef size_stride_t fixed_dim_type_arrmeta;

struct DYNDT_API fixed_dim_type_iterdata {
  iterdata_common common;
  char *data;
  intptr_t stride;
};

namespace ndt {

  class DYNDT_API fixed_dim_type : public base_dim_type {
    intptr_t m_dim_size;

  public:
    typedef size_stride_t metadata_type;

    fixed_dim_type(type_id_t id, intptr_t dim_size, const type &element_tp = make_type<any_kind_type>())
        : base_dim_type(id, fixed_dim_kind_id, element_tp, 0, element_tp.get_data_alignment(),
                        sizeof(fixed_dim_type_arrmeta), type_flag_none, true),
          m_dim_size(dim_size) {
      // Propagate the inherited flags from the element
      this->flags |= (element_tp.get_flags() & (type_flags_operand_inherited | type_flags_value_inherited));
    }

    size_t get_default_data_size() const;

    intptr_t get_fixed_dim_size() const { return m_dim_size; }

    intptr_t get_fixed_stride(const char *arrmeta) const {
      return reinterpret_cast<const size_stride_t *>(arrmeta)->stride;
    }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_c_contiguous(const char *arrmeta) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;
    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;
    void get_strides(size_t i, intptr_t *out_strides, const char *arrmeta) const;

    axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const nd::memory_block &embedded_reference) const;

    size_t get_iterdata_size(intptr_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                              const intptr_t *shape, type &out_uniform_tp) const;
    size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback, void *callback_data) const;

    /**
     * Modifies arrmeta allocated using the arrmeta_default_construct function,
     *to be used
     * immediately after nd::array construction. Given an input type/arrmeta,
     *edits the output
     * arrmeta in place to match.
     *
     * \param dst_arrmeta  The arrmeta created by arrmeta_default_construct,
     *which is modified in place
     * \param src_tp  The type of the input nd::array whose stride ordering is
     *to be matched.
     * \param src_arrmeta  The arrmeta of the input nd::array whose stride
     *ordering is to be matched.
     */
    void reorder_default_constructed_strides(char *dst_arrmeta, const type &src_tp, const char *src_arrmeta) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual type with_element_type(const type &element_tp) const;
  };

  template <>
  struct id_of<fixed_dim_type> : std::integral_constant<type_id_t, fixed_dim_id> {};

  inline type make_fixed_dim(size_t dim_size, const type &element_tp) {
    return make_type<fixed_dim_type>(dim_size, element_tp);
  }

} // namespace dynd::ndt
} // namespace dynd
