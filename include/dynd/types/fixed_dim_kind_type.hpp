//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/dim_kind_type.hpp>

namespace dynd {

template <typename ElementType, int NDim>
class fixed_dim_iterator;

template <typename ElementType>
class fixed_dim : public as_t<ElementType> {
protected:
  fixed_dim operator()(const char *metadata, char *data) { return fixed_dim(metadata, data); }

  template <typename Index0Type, typename... IndexType>
  decltype(auto) operator()(const char *metadata, char *data, Index0Type index0, IndexType... index) {
    return as_t<ElementType>::operator()(metadata + sizeof(size_stride_t),
                                         data + index0 * reinterpret_cast<const size_stride_t *>(metadata)->stride,
                                         index...);
  }

public:
  static const intptr_t ndim = as_t<ElementType>::ndim + 1;
  typedef typename as_t<ElementType>::data_type data_type;

  template <int NDim>
  class iterator_type : public fixed_dim_iterator<ElementType, NDim> {
  public:
    iterator_type(const char *metadata, char *data) : fixed_dim_iterator<ElementType, NDim>(metadata, data) {}
  };

  fixed_dim(const char *metadata, char *data) : as_t<ElementType>(metadata, data) {}

  size_t size() const { return reinterpret_cast<const size_stride_t *>(this->m_metadata)->dim_size; }

  void set_data(char *data) { this->m_data = data; }

  template <typename... IndexType>
  decltype(auto) operator()(IndexType... index) {
    static_assert(sizeof...(IndexType) <= ndim, "too many indices");
    return (*this)(this->m_metadata, this->m_data, index...);
  }

  template <int NDim = 1>
  iterator_type<NDim> begin() {
    return iterator_type<NDim>(this->m_metadata, this->m_data);
  }

  template <int NDim = 1>
  iterator_type<NDim> end() {
    return iterator_type<NDim>(this->m_metadata,
                               this->m_data +
                                   reinterpret_cast<const size_stride_t *>(this->m_metadata)->dim_size *
                                       reinterpret_cast<const size_stride_t *>(this->m_metadata)->stride);
  }
};

template <typename ElementType>
class fixed_dim_iterator<ElementType, 0> {
protected:
  const char *m_metadata;
  char *m_data;

public:
  fixed_dim_iterator(const char *metadata, char *data) : m_metadata(metadata), m_data(data) {}

  fixed_dim<ElementType> operator*() { return fixed_dim<ElementType>(m_metadata, m_data); }

  bool operator==(const fixed_dim_iterator &rhs) const { return m_data == rhs.m_data; }

  bool operator!=(const fixed_dim_iterator &rhs) const { return m_data != rhs.m_data; }
};

template <typename ElementType, int NDim>
class fixed_dim_iterator : public as_t<ElementType>::template iterator_type<NDim - 1> {
  intptr_t m_stride;

public:
  fixed_dim_iterator(const char *metadata, char *data)
      : as_t<ElementType>::template iterator_type<NDim - 1>(metadata + sizeof(size_stride_t), data),
        m_stride(reinterpret_cast<const size_stride_t *>(metadata)->stride) {}

  fixed_dim_iterator &operator++() {
    this->m_data += m_stride;
    return *this;
  }

  fixed_dim_iterator operator++(int) {
    fixed_dim_iterator tmp(*this);
    operator++();
    return tmp;
  }
};

namespace ndt {

  class DYNDT_API fixed_dim_kind_type : public base_dim_type {
  public:
    using base_dim_type::base_dim_type;

    fixed_dim_kind_type(type_id_t id, const type &element_tp = make_type<any_kind_type>())
        : base_dim_type(id, element_tp, 0, element_tp.get_data_alignment(), sizeof(size_stride_t), type_flag_symbolic,
                        true) {
      // Propagate the inherited flags from the element
      this->flags |= (element_tp.get_flags() & (type_flags_operand_inherited | type_flags_value_inherited));
    }

    size_t get_default_data_size() const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;
    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

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

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    virtual type with_element_type(const type &element_tp) const;
  };

  template <>
  struct id_of<fixed_dim_kind_type> : std::integral_constant<type_id_t, fixed_dim_kind_id> {};

  template <typename T>
  struct traits<T[]> {
    static type equivalent() { return make_type<fixed_dim_kind_type>(make_type<T>()); }
  };

  // Need to handle const properly
  template <typename T>
  struct traits<const T[]> {
    static type equivalent() { return make_type<T[]>(); }
  };

  template <typename ElementType>
  struct traits<fixed_dim<ElementType>> {
    static const bool is_same_layout = false;

    static type equivalent() { return make_type<fixed_dim_kind_type>(make_type<ElementType>()); }
  };

} // namespace dynd::ndt
} // namespace dynd
