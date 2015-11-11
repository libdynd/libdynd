//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/array.hpp>

namespace dynd {

// fixed_dim (redundantly) uses the same arrmeta as strided_dim
typedef size_stride_t fixed_dim_type_arrmeta;

struct DYND_API fixed_dim_type_iterdata {
  iterdata_common common;
  char *data;
  intptr_t stride;
};

template <typename ElementType, int NDim>
class fixed_dim_iterator;

template <typename ElementType>
class fixed_dim : public as_t<ElementType> {
protected:
  fixed_dim operator()(const char *metadata, char *data)
  {
    return fixed_dim(metadata, data);
  }

  template <typename Index0Type, typename... IndexType>
  decltype(auto) operator()(const char *metadata, char *data, Index0Type index0, IndexType... index)
  {
    return as_t<ElementType>::operator()(
        metadata + sizeof(fixed_dim_type_arrmeta),
        data + index0 * reinterpret_cast<const fixed_dim_type_arrmeta *>(metadata)->stride, index...);
  }

public:
  static const intptr_t ndim = as_t<ElementType>::ndim + 1;
  typedef typename as_t<ElementType>::data_type data_type;

  template <int NDim>
  class iterator_type : public fixed_dim_iterator<ElementType, NDim> {
  public:
    iterator_type(const char *metadata, char *data) : fixed_dim_iterator<ElementType, NDim>(metadata, data)
    {
    }
  };

  fixed_dim(const char *metadata, char *data) : as_t<ElementType>(metadata, data)
  {
  }

  size_t size() const
  {
    return reinterpret_cast<const fixed_dim_type_arrmeta *>(this->m_metadata)->dim_size;
  }

  void set_data(char *data)
  {
    this->m_data = data;
  }

  template <typename... IndexType>
  decltype(auto) operator()(IndexType... index)
  {
    static_assert(sizeof...(IndexType) <= ndim, "too many indices");
    return (*this)(this->m_metadata, this->m_data, index...);
  }

  template <int NDim = 1>
  iterator_type<NDim> begin()
  {
    return iterator_type<NDim>(this->m_metadata, this->m_data);
  }

  template <int NDim = 1>
  iterator_type<NDim> end()
  {
    return iterator_type<NDim>(this->m_metadata,
                               this->m_data +
                                   reinterpret_cast<const fixed_dim_type_arrmeta *>(this->m_metadata)->dim_size *
                                       reinterpret_cast<const fixed_dim_type_arrmeta *>(this->m_metadata)->stride);
  }
};

template <typename ElementType>
class fixed_dim_iterator<ElementType, 0> {
protected:
  const char *m_metadata;
  char *m_data;

public:
  fixed_dim_iterator(const char *metadata, char *data) : m_metadata(metadata), m_data(data)
  {
  }

  fixed_dim<ElementType> operator*()
  {
    return fixed_dim<ElementType>(m_metadata, m_data);
  }

  bool operator==(const fixed_dim_iterator &rhs) const
  {
    return m_data == rhs.m_data;
  }

  bool operator!=(const fixed_dim_iterator &rhs) const
  {
    return m_data != rhs.m_data;
  }
};

template <typename ElementType, int NDim>
class fixed_dim_iterator : public as_t<ElementType>::template iterator_type<NDim - 1> {
  intptr_t m_stride;

public:
  fixed_dim_iterator(const char *metadata, char *data)
      : as_t<ElementType>::template iterator_type<NDim - 1>(metadata + sizeof(fixed_dim_type_arrmeta), data),
        m_stride(reinterpret_cast<const fixed_dim_type_arrmeta *>(metadata)->stride)
  {
  }

  fixed_dim_iterator &operator++()
  {
    this->m_data += m_stride;
    return *this;
  }

  fixed_dim_iterator operator++(int)
  {
    fixed_dim_iterator tmp(*this);
    operator++();
    return tmp;
  }
};

namespace ndt {

  class DYND_API fixed_dim_type : public base_dim_type {
    intptr_t m_dim_size;
    std::vector<std::pair<std::string, gfunc::callable>> m_array_properties, m_array_functions;

  public:
    fixed_dim_type(intptr_t dim_size, const type &element_tp);

    virtual ~fixed_dim_type();

    size_t get_default_data_size() const;

    intptr_t get_fixed_dim_size() const
    {
      return m_dim_size;
    }

    intptr_t get_fixed_stride(const char *arrmeta) const
    {
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
                                char *out_arrmeta, const intrusive_ptr<memory_block_data> &embedded_reference,
                                size_t current_i, const type &root_tp, bool leading_dimension, char **inout_data,
                                intrusive_ptr<memory_block_data> &inout_dataref) const;
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
                                const intrusive_ptr<memory_block_data> &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;
    size_t arrmeta_copy_construct_onedim(char *dst_arrmeta, const char *src_arrmeta,
                                         const intrusive_ptr<memory_block_data> &embedded_reference) const;

    size_t get_iterdata_size(intptr_t ndim) const;
    size_t iterdata_construct(iterdata_common *iterdata, const char **inout_arrmeta, intptr_t ndim,
                              const intptr_t *shape, type &out_uniform_tp) const;
    size_t iterdata_destruct(iterdata_common *iterdata, intptr_t ndim) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    intptr_t make_assignment_kernel(void *ckb, intptr_t ckb_offset, const type &dst_tp, const char *dst_arrmeta,
                                    const type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                    const eval::eval_context *ectx) const;

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

    bool match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
               std::map<std::string, type> &tp_vars) const;

    void get_dynamic_type_properties(const std::pair<std::string, nd::callable> **out_properties,
                                     size_t *out_count) const;
    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties,
                                      size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions,
                                     size_t *out_count) const;

    virtual type with_element_type(const type &element_tp) const;
  };

  /**
   * Does a value lookup into an array of type "N * T", without
   * bounds checking the index ``i`` or validating that ``a`` has the
   * required type. Use only when these checks have been done externally.
   */
  template <typename T>
  inline const T &unchecked_fixed_dim_get(const nd::array &a, intptr_t i)
  {
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(a.get()->metadata());
    return *reinterpret_cast<const T *>(a.cdata() + i * md->stride);
  }

  /**
   * Does a writable value lookup into an array of type "N * T", without
   * bounds checking the index ``i`` or validating that ``a`` has the
   * required type. Use only when these checks have been done externally.
   */
  template <typename T>
  inline T &unchecked_fixed_dim_get_rw(const nd::array &a, intptr_t i)
  {
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(a.get()->metadata());
    return *reinterpret_cast<T *>(a.data() + i * md->stride);
  }

  DYND_API type make_fixed_dim(size_t dim_size, const type &element_tp);

  DYND_API type make_fixed_dim(intptr_t ndim, const intptr_t *shape, const type &dtp);

  inline type make_fixed_dim(size_t dim_size, const type &element_tp, intptr_t ndim)
  {
    type result = element_tp;
    for (intptr_t i = 0; i < ndim; ++i) {
      result = make_fixed_dim(dim_size, result);
    }

    return result;
  }

  template <typename T, int N>
  struct type::equivalent<T[N]> {
    static type make()
    {
      return make_fixed_dim(N, type::make<T>());
    }
  };

  // Need to handle const properly
  template <typename T, int N>
  struct type::equivalent<const T[N]> {
    static type make()
    {
      return type::make<T[N]>();
    }
  };

  template <typename ElementType>
  struct type::equivalent<fixed_dim<ElementType>> {
    static type make()
    {
      return fixed_dim_kind_type::make(type::make<ElementType>());
    }
  };

} // namespace dynd::ndt
} // namespace dynd
