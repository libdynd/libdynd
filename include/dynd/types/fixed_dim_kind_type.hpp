//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/dim_kind_type.hpp>

namespace dynd {

template <typename ElementType, typename Enable = void>
class fixed;

template <typename ElementType>
class fixed<ElementType, std::enable_if_t<ndt::traits<ElementType>::is_same_layout>> {
  union {
    size_stride_t m_size_stride;
    char m_metadata[ndt::traits<fixed<ElementType>>::metadata_size];
  };
  char *m_data;

public:
  class iterator {
    intptr_t m_stride;
    char *m_data;

  public:
    iterator(const char *metadata, char *data)
        : m_stride(reinterpret_cast<const size_stride_t *>(metadata)->stride), m_data(data) {}

    ElementType &operator*() { return *reinterpret_cast<ElementType *>(m_data); }

    iterator &operator++() {
      m_data += m_stride;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator==(const iterator &rhs) const { return m_data == rhs.m_data; }

    bool operator!=(const iterator &rhs) const { return m_data != rhs.m_data; }
  };

  fixed(const char *metadata, char *data) : m_data(data) {
    ndt::traits<fixed<ElementType>>::metadata_copy_construct(m_metadata, metadata);
  }

  char *data() const { return m_data; }
  size_t size() const { return m_size_stride.dim_size; }
  intptr_t stride() const { return m_size_stride.stride; }

  fixed &assign(char *data) {
    m_data = data;
    return *this;
  }

  const ElementType &operator[](size_t i) const {
    return *reinterpret_cast<const ElementType *>(this->m_data + i * m_size_stride.stride);
  }

  ElementType &operator[](size_t i) {
    return *reinterpret_cast<ElementType *>(this->m_data + i * m_size_stride.stride);
  }

  iterator begin() const { return iterator(this->m_metadata, this->m_data); }

  iterator end() const {
    return iterator(this->m_metadata, this->m_data + m_size_stride.dim_size * m_size_stride.stride);
  }
};

template <typename ElementType>
class fixed<ElementType, std::enable_if_t<!ndt::traits<ElementType>::is_same_layout>> {
  union {
    size_stride_t m_size_stride;
    char m_metadata[ndt::traits<fixed<ElementType>>::metadata_size];
  };
  char *m_data;

public:
  class iterator;

  friend class iterator;

  class iterator {
    intptr_t m_stride;
    ElementType m_current;

  public:
    iterator(const char *metadata, char *data)
        : m_stride(reinterpret_cast<const size_stride_t *>(metadata)->stride),
          m_current(metadata + sizeof(size_stride_t), data) {}

    ElementType operator*() { return m_current; }

    iterator &operator++() {
      m_current.assign(m_current.data() + m_stride);
      return *this;
    }

    iterator operator++(int) {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator==(const iterator &rhs) const { return m_current.data() == rhs.m_current.data(); }

    bool operator!=(const iterator &rhs) const { return m_current.data() != rhs.m_current.data(); }
  };

  fixed(const char *metadata, char *data) : m_data(data) {
    ndt::traits<fixed<ElementType>>::metadata_copy_construct(m_metadata, metadata);
  }

  char *data() const { return m_data; }
  size_t size() const { return m_size_stride.dim_size; }
  intptr_t stride() const { return m_size_stride.stride; }

  fixed &assign(char *data) {
    m_data = data;
    return *this;
  }

  ElementType operator[](size_t i) const {
    return ElementType(m_metadata + sizeof(size_stride_t), m_data + i * m_size_stride.stride);
  }

  ElementType operator[](size_t i) {
    return ElementType(m_metadata + sizeof(size_stride_t), m_data + i * m_size_stride.stride);
  }

  iterator begin() const { return iterator(m_metadata, m_data); }

  iterator end() const { return iterator(m_metadata, m_data + m_size_stride.dim_size * m_size_stride.stride); }
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

    static ndt::type construct_type(type_id_t id, const nd::buffer &args, const ndt::type &element_type);
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
  struct traits<fixed<ElementType>> {
    static const size_t metadata_size = sizeof(size_stride_t) + traits<ElementType>::metadata_size;
    static const size_t ndim = 1 + traits<ElementType>::ndim;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<fixed_dim_kind_type>(make_type<ElementType>()); }

    static void metadata_copy_construct(char *dst, const char *src) {
      reinterpret_cast<size_stride_t *>(dst)->dim_size = reinterpret_cast<const size_stride_t *>(src)->dim_size;
      reinterpret_cast<size_stride_t *>(dst)->stride = reinterpret_cast<const size_stride_t *>(src)->stride;

      traits<ElementType>::metadata_copy_construct(dst + sizeof(size_stride_t), src + sizeof(size_stride_t));
    }
  };

} // namespace dynd::ndt
} // namespace dynd
