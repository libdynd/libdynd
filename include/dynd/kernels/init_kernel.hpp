//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {

template <typename T>
struct value_type {
  typedef typename T::value_type type;
};

template <typename T, size_t N>
struct value_type<T[N]> {
  typedef T type;
};

template <typename T>
using value_type_t = typename value_type<T>::type;

namespace nd {

  template <typename ValueType>
  struct init_kernel {
    init_kernel(const char *DYND_UNUSED(metadata)) {}

    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const ValueType &value) { *reinterpret_cast<ValueType *>(data) = value; }

    void contiguous(char *data, const ValueType *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(ValueType);
      }
    }
  };

  template <>
  struct init_kernel<bool> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, bool value) { *reinterpret_cast<bool1 *>(data) = value; }

    void contiguous(char *data, const bool *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(bool1);
      }
    }
  };

  template <>
  struct init_kernel<bytes> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const bytes &value) { reinterpret_cast<bytes *>(data)->assign(value.data(), value.size()); }

    void contiguous(char *data, const bytes *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(bytes);
      }
    }
  };

  template <>
  struct init_kernel<std::string> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const std::string &value) {
      reinterpret_cast<string *>(data)->assign(value.data(), value.size());
    }

    void contiguous(char *data, const std::string *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <>
  struct init_kernel<const char *> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) { reinterpret_cast<string *>(data)->assign(value, strlen(value)); }

    void contiguous(char *data, const char *const *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <size_t N>
  struct init_kernel<char[N]> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) { reinterpret_cast<string *>(data)->assign(value, N - 1); }

    void contiguous(char *data, const char *const *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  template <size_t N>
  struct init_kernel<const char[N]> {
    init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

    void single(char *data, const char *value) { reinterpret_cast<string *>(data)->assign(value, N - 1); }

    void contiguous(char *data, const char *const *values, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        single(data, values[i]);
        data += sizeof(string);
      }
    }
  };

  namespace detail {

    template <typename ResType, typename ContainerType, typename Enable = void>
    struct init_kernel;

    // ContainerType is same_layout -> direct memcpy
    // ContainerType is not same_layout, but ContainerType is contiguous -> child.contiguous of unpacked data
    // otherwise for loop

    template <typename ContainerType>
    struct init_kernel<ndt::fixed_dim_type, ContainerType,
                       std::enable_if_t<ndt::traits<ContainerType>::is_same_layout>> {
      typedef value_type_t<ContainerType> value_type;

      nd::init_kernel<value_type> child;

      init_kernel(const ndt::type &tp, const char *metadata)
          : child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t)) {}

      void single(char *data, const ContainerType &values) { memcpy(data, values, sizeof(ContainerType)); }

      void contiguous(char *data, const ContainerType *values, size_t size) {
        memcpy(data, values, size * sizeof(ContainerType));
      }
    };

    template <typename ContainerType>
    struct init_kernel<ndt::fixed_dim_type, ContainerType,
                       std::enable_if_t<!ndt::traits<ContainerType>::is_same_layout &&
                                        ndt::traits<ContainerType>::is_contiguous_container>> {
      typedef value_type_t<ContainerType> value_type;

      size_t size;
      intptr_t stride;
      nd::init_kernel<value_type> child;

      init_kernel(const ndt::type &tp, const char *metadata)
          : size(reinterpret_cast<const size_stride_t *>(metadata)->dim_size),
            stride(reinterpret_cast<const size_stride_t *>(metadata)->stride),
            child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t)) {}

      /*
          void single(char *data, const ContainerType &values) {
            for (const value_type &value : values) {
              child.single(data, value);
              data += stride;
            }
          }
      */

      void single(char *data, const ContainerType &values) { child.contiguous(data, &values[0], size); }

      void contiguous(char *data, const ContainerType *values, size_t size) {
        for (size_t i = 0; i < size; ++i) {
          single(data, values[i]);
          data += size * stride;
        }
      }
    };

    template <typename ContainerType>
    struct init_kernel<ndt::fixed_dim_type, ContainerType,
                       std::enable_if_t<!ndt::traits<ContainerType>::is_same_layout &&
                                        !ndt::traits<ContainerType>::is_contiguous_container>> {
      typedef value_type_t<ContainerType> value_type;

      intptr_t stride;
      nd::init_kernel<value_type> child;

      init_kernel(const ndt::type &tp, const char *metadata)
          : stride(reinterpret_cast<const size_stride_t *>(metadata)->stride),
            child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t)) {}

      void single(char *data, const ContainerType &values) {
        for (const value_type &value : values) {
          child.single(data, value);
          data += stride;
        }
      }
    };

  } // namespace dynd::nd::detail

  template <typename ContainerType, size_t Rank>
  struct container_init {
    typedef void (*closure_type)(container_init *, char *, const ContainerType &);
    typedef typename ContainerType::value_type value_type;

    intptr_t stride;
    closure_type closure;
    init_kernel<value_type> child;

    container_init(const ndt::type &tp, const char *metadata)
        : child(tp.extended<ndt::base_dim_type>()->get_element_type(),
                metadata + tp.extended<ndt::base_dim_type>()->get_element_arrmeta_offset()) {
      switch (tp.get_id()) {
      case fixed_dim_id:
        stride = reinterpret_cast<const size_stride_t *>(metadata)->stride;
        closure = [](container_init *self, char *data, const ContainerType &values) {
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

    void single(char *data, const ContainerType &values) { closure(this, data, values); }
  };

  template <typename ValueType>
  struct container_init<std::initializer_list<ValueType>, 1> {
    typedef void (*closure_type)(container_init *, char *, const std::initializer_list<ValueType> &);
    typedef ValueType value_type;

    memory_block memblock;
    closure_type closure;
    init_kernel<value_type> child;

    container_init(const ndt::type &tp, const char *metadata)
        : child(tp.extended<ndt::base_dim_type>()->get_element_type(), metadata + sizeof(size_stride_t)) {
      switch (tp.get_id()) {
      case fixed_dim_id:
        closure = [](container_init *self, char *data, const std::initializer_list<ValueType> &values) {
          self->child.contiguous(data, values.begin(), values.size());
        };
        break;
      case var_dim_id:
        memblock = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(metadata)->blockref;
        closure = [](container_init *self, char *data, const std::initializer_list<ValueType> &values) {
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

    void single(char *data, const std::initializer_list<ValueType> &values) { closure(this, data, values); }
  };

  template <typename ValueType, size_t Size>
  struct init_kernel<ValueType[Size]> : detail::init_kernel<ndt::fixed_dim_type, ValueType[Size]> {
    using detail::init_kernel<ndt::fixed_dim_type, ValueType[Size]>::init_kernel;
  };

  template <typename ValueType, size_t Size>
  struct init_kernel<std::array<ValueType, Size>>
      : detail::init_kernel<ndt::fixed_dim_type, std::array<ValueType, Size>> {
    using detail::init_kernel<ndt::fixed_dim_type, std::array<ValueType, Size>>::init_kernel;
  };

  template <typename T>
  struct init_kernel<std::initializer_list<T>> : container_init<std::initializer_list<T>, ndt::traits<T>::ndim + 1> {
    using container_init<std::initializer_list<T>, ndt::traits<T>::ndim + 1>::container_init;
  };

  template <typename T>
  struct init_kernel<std::vector<T>> : detail::init_kernel<ndt::fixed_dim_type, std::vector<T>> {
    using detail::init_kernel<ndt::fixed_dim_type, std::vector<T>>::init_kernel;
  };

  template <typename... ElementTypes>
  struct init_kernel<std::tuple<ElementTypes...>> {
    struct on_each {
      template <typename ElementType, size_t I>
      void operator()(const char *metadata, std::tuple<init_kernel<ElementTypes>...> &children, char *data,
                      const std::tuple<ElementTypes...> &value) {
        init_kernel<ElementType> &child = std::get<I>(children);
        child.single(data + *(reinterpret_cast<const uintptr_t *>(metadata) + I), std::get<I>(value));
      }
    };

    const char *metadata;
    std::tuple<init_kernel<ElementTypes>...> children;

    init_kernel(const ndt::type *field_tp, const char *metadata)
        : metadata(postfix_add(metadata, sizeof...(ElementTypes) * sizeof(uintptr_t))),
          children({*postfix_add(field_tp, 1), postfix_add(metadata, ndt::traits<ElementTypes>::metadata_size)}...) {}

    init_kernel(const ndt::type &tp, const char *metadata)
        : init_kernel(tp.extended<ndt::tuple_type>()->get_field_types_raw(), metadata) {}

    void single(char *data, const std::tuple<ElementTypes...> &value) {
      for_each<type_sequence<ElementTypes...>, 0>(on_each(), metadata, children, data, value);
    }
  };

} // namespace dynd::nd
} // namespace dynd
