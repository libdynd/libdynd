//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/string_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {
namespace nd {

  template <typename ValueType>
  struct init {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const ValueType &value) const { *reinterpret_cast<ValueType *>(data) = value; }

    void contiguous(char *data, const ValueType *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(ValueType);
      }
    }
  };

  template <>
  struct init<bool> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, bool value) const { *reinterpret_cast<bool1 *>(data) = value; }

    void contiguous(char *data, const bool *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(bool1);
      }
    }
  };

  template <>
  struct init<bytes> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const bytes &value) const
    {
      reinterpret_cast<bytes *>(data)->assign(value.data(), value.size());
    }

    void contiguous(char *data, const bytes *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(bytes);
      }
    }
  };

  template <>
  struct init<std::string> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const std::string &value) const
    {
      reinterpret_cast<string *>(data)->assign(value.data(), value.size());
    }

    void contiguous(char *data, const std::string *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <>
  struct init<const char *> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) const { reinterpret_cast<string *>(data)->assign(value, strlen(value)); }

    void contiguous(char *data, const char *const *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <size_t N>
  struct init<char[N]> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) const { reinterpret_cast<string *>(data)->assign(value, N - 1); }

    void contiguous(char *data, const char *const *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <size_t N>
  struct init<const char[N]> {
    init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) const { reinterpret_cast<string *>(data)->assign(value, N - 1); }

    void contiguous(char *data, const char *const *values, size_t size) const
    {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <typename ContainerType, size_t Rank>
  struct container_init {
    typedef void (*closure_type)(const container_init *, char *, const ContainerType &);
    typedef typename ContainerType::value_type value_type;

    intptr_t stride;
    closure_type closure;
    init<value_type> child;

    container_init(const ndt::type &tp, const char *metadata)
        : child(tp.extended<ndt::base_dim_type>()->get_element_type(),
                metadata + tp.extended<ndt::base_dim_type>()->get_element_arrmeta_offset())
    {
      switch (tp.get_id()) {
      case fixed_dim_id:
        stride = reinterpret_cast<const size_stride_t *>(metadata)->stride;
        closure = [](const container_init *self, char *data, const ContainerType &values) {
          for (const value_type &value : values) {
            self->child.single(data, value);
            data += self->stride;
          }
        };
        break;
      default:
        throw std::runtime_error("unsupported");
      }
    }

    void single(char *data, const ContainerType &values) const { closure(this, data, values); }
  };

  template <typename ValueType>
  struct container_init<std::initializer_list<ValueType>, 1> {
    typedef void (*closure_type)(const container_init *, char *, const std::initializer_list<ValueType> &);
    typedef ValueType value_type;

    intrusive_ptr<memory_block_data> memblock;
    closure_type closure;
    init<value_type> child;

    container_init(const ndt::type &tp, const char *metadata)
        : child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t))
    {
      switch (tp.get_id()) {
      case fixed_dim_id:
        closure = [](const container_init *self, char *data, const std::initializer_list<ValueType> &values) {
          self->child.contiguous(data, values.begin(), values.size());
        };
        break;
      case var_dim_id:
        memblock = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(metadata)->blockref;
        closure = [](const container_init *self, char *data, const std::initializer_list<ValueType> &values) {
          reinterpret_cast<ndt::var_dim_type::data_type *>(data)->begin = self->memblock->alloc(values.size());
          reinterpret_cast<ndt::var_dim_type::data_type *>(data)->size = values.size();
          self->child.contiguous(reinterpret_cast<ndt::var_dim_type::data_type *>(data)->begin, values.begin(),
                                 values.size());
        };
        break;
      default:
        throw std::runtime_error("unexpected type id");
      }
    }

    void single(char *data, const std::initializer_list<ValueType> &values) const { closure(this, data, values); }
  };

  template <typename ContainerType>
  struct container_init<ContainerType, 1> {
    typedef typename ContainerType::value_type value_type;

    void (*func)(const container_init *, char *, const ContainerType &);
    init<value_type> child;

    container_init(const ndt::type &tp, const char *metadata)
        : child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t))
    {
    }

    void single(char *data, const ContainerType &values) const { child.contiguous(data, values.data(), values.size()); }
  };

  template <typename T>
  struct init<std::initializer_list<T>> : container_init<std::initializer_list<T>, ndt::traits<T>::ndim + 1> {
    using container_init<std::initializer_list<T>, ndt::traits<T>::ndim + 1>::container_init;
  };

  template <typename T>
  struct init<std::vector<T>> : container_init<std::vector<T>, ndt::traits<T>::ndim + 1> {
    using container_init<std::vector<T>, ndt::traits<T>::ndim + 1>::container_init;
  };

} // namespace dynd::nd
} // namespace dynd
