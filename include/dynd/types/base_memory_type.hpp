//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace ndt {

  /**
   * Base class for all types of memory_kind. This indicates a type is not in
   * default memory, meaning it may not be readable from the host, but maybe
   * only on a GPU or similar device. Users of this class should implement
   * methods that allocate and free data, among other things.
   *
   * Contrast this with memory on the host, but allocated by a system outside of
   * dynd. This memory can be tracked via the object in
   * memblock/external_memory_block.hpp.
   */
  class DYND_API base_memory_type : public base_type {
  protected:
    type m_element_tp;
    size_t m_storage_arrmeta_offset;

  public:
    base_memory_type(type_id_t type_id, const type &element_tp,
                     size_t data_size, size_t alignment,
                     size_t storage_arrmeta_offset, flags_type flags)
        : base_type(type_id, memory_kind, data_size, alignment, flags,
                    storage_arrmeta_offset + element_tp.get_arrmeta_size(),
                    element_tp.get_ndim(), 0),
          m_element_tp(element_tp),
          m_storage_arrmeta_offset(storage_arrmeta_offset)
    {
      if (element_tp.get_kind() == memory_kind) {
        std::stringstream ss;
        ss << "a memory space cannot be specified for type " << element_tp;
        throw std::runtime_error(ss.str());
      }
    }

    virtual ~base_memory_type();

    const type &get_element_type() const { return m_element_tp; }

    virtual size_t get_default_data_size() const;

    virtual void get_vars(std::unordered_set<std::string> &vars) const;

    virtual void print_data(std::ostream &o, const char *arrmeta,
                            const char *data) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                   const char *arrmeta, const char *data) const;

    void get_strides(size_t i, intptr_t *out_strides,
                     const char *arrmeta) const;

    type apply_linear_index(intptr_t nindices, const irange *indices,
                            size_t current_i, const type &root_tp,
                            bool leading_dimension) const;

    intptr_t apply_linear_index(intptr_t nindices, const irange *indices,
                                const char *arrmeta, const type &result_type,
                                char *out_arrmeta,
                                const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp,
                                bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;

    type at_single(intptr_t i0, const char **inout_arrmeta,
                   const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i,
                               intptr_t total_ndim = 0) const;

    virtual bool is_lossless_assignment(const type &dst_tp,
                                        const type &src_tp) const;

    virtual bool operator==(const base_type &rhs) const = 0;

    inline bool is_type_subarray(const type &subarray_tp) const
    {
      return (!subarray_tp.is_builtin() &&
              (*this) == (*subarray_tp.extended())) ||
             m_element_tp.is_type_subarray(subarray_tp);
    }

    virtual void transform_child_types(type_transform_fn_t transform_fn,
                                       intptr_t arrmeta_offset, void *extra,
                                       type &out_transformed_tp,
                                       bool &out_was_transformed) const;
    virtual type get_canonical_type() const;

    virtual type with_replaced_storage_type(const type &storage_tp) const = 0;

    virtual void arrmeta_default_construct(char *arrmeta,
                                           bool blockref_alloc) const;
    virtual void
    arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                           const intrusive_ptr<memory_block_data> &embedded_reference) const;
    virtual void arrmeta_destruct(char *arrmeta) const;

    virtual void data_alloc(char **data, size_t size) const = 0;
    virtual void data_zeroinit(char *data, size_t size) const = 0;
    virtual void data_free(char *data) const = 0;

    virtual bool match(const char *arrmeta, const type &candidate_tp,
                       const char *candidate_arrmeta,
                       std::map<std::string, type> &tp_vars) const;

    virtual void get_dynamic_type_properties(
        const std::pair<std::string, nd::callable> **out_properties,
        size_t *out_count) const;
  };

} // namespace dynd::ndt
} // namespace dynd
