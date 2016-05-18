//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/buffer.hpp>
#include <dynd/types/base_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API var_dim_type : public base_dim_type {

  public:
    struct metadata_type {
      nd::memory_block blockref; // A reference to the memory block which contains the array's data.
      intptr_t stride;
      intptr_t offset; // Each pointed-to destination is offset by this amount
    };

    struct data_type {
      char *begin;
      size_t size;
    };

    var_dim_type(const type &element_tp);

    var_dim_type(size_t ndim, const type &element_tp)
        : var_dim_type(ndim == 1 ? element_tp : make_type<var_dim_type>(ndim - 1, element_tp)) {}

    size_t get_default_data_size() const { return sizeof(data_type); }

    /** Alignment of the data being pointed to. */
    size_t get_target_alignment() const { return m_element_tp.get_data_alignment(); }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

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

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    size_t get_iterdata_size(intptr_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                              const intptr_t *shape, type &out_uniform_tp) const;
    size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback, void *callback_data) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual type with_element_type(const type &element_tp) const;
  };

  inline type make_var_dim(const type &element_tp) { return make_type<var_dim_type>(element_tp); }

} // namespace dynd::ndt

template <>
struct type_id_of<ndt::var_dim_type> {
  static const type_id_t value = var_dim_id;
};

} // namespace dynd
