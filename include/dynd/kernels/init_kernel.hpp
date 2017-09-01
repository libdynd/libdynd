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

template <typename ContainerType>
decltype(auto) front_data(const ContainerType &values) {
  return &values[0];
}

template <typename ValueType>
decltype(auto) front_data(const std::initializer_list<ValueType> &values) {
  return values.begin();
}

namespace nd {

  template <typename ValueType>
  struct init_kernel {
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

    template <typename ContainerType>
    struct init_kernel<ndt::fixed_dim_type, ContainerType,
                       std::enable_if_t<ndt::traits<ContainerType>::is_same_layout>> {
      init_kernel(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(metadata)) {}

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

      void single(char *data, const ContainerType &values) { child.contiguous(data, front_data(values), size); }

      void contiguous(char *data, const ContainerType *values, size_t size) {
        for (size_t i = 0; i < size; ++i) {
          single(data, values[i]);
          data += values[i].size() * stride;
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

    template <typename ContainerType>
    struct init_kernel<ndt::var_dim_type, ContainerType> {
      typedef value_type_t<ContainerType> value_type;

      memory_block memblock;
      nd::init_kernel<value_type> child;

      init_kernel(const ndt::type &tp, const char *metadata)
          : memblock(reinterpret_cast<const ndt::var_dim_type::metadata_type *>(metadata)->blockref),
            child(tp.extended<ndt::base_dim_type>()->get_element_type(),
                  metadata + sizeof(ndt::var_dim_type::metadata_type)) {}

      void single(char *data, const ContainerType &values) {
        reinterpret_cast<ndt::var_dim_type::data_type *>(data)->begin = memblock->alloc(values.size());
        reinterpret_cast<ndt::var_dim_type::data_type *>(data)->size = values.size();
        child.contiguous(reinterpret_cast<ndt::var_dim_type::data_type *>(data)->begin, values.begin(), values.size());
      }

      void contiguous(char *data, const ContainerType *values, size_t size) {
        for (size_t i = 0; i < size; ++i) {
          single(data, values[i]);
          data += sizeof(ndt::var_dim_type::data_type);
        }
      }
    };

  } // namespace dynd::nd::detail

  template <typename ContainerType, typename Enable = void>
  struct container_init;

  template <typename ValueType>
  struct container_init<std::initializer_list<ValueType>,
                        std::enable_if_t<ndt::traits<std::initializer_list<ValueType>>::ndim != 1>>
      : detail::init_kernel<ndt::fixed_dim_type, std::initializer_list<ValueType>> {
    using detail::init_kernel<ndt::fixed_dim_type, std::initializer_list<ValueType>>::init_kernel;
  };

  template <typename ContainerType>
  struct container_init<ContainerType, std::enable_if_t<ndt::traits<ContainerType>::ndim == 1>> {
    typedef value_type_t<ContainerType> value_type;

    std::aligned_union_t<1, detail::init_kernel<ndt::fixed_dim_type, ContainerType>,
                         detail::init_kernel<ndt::var_dim_type, ContainerType>>
        child;

    void (*destruct_wrapper)(container_init *);
    void (*single_wrapper)(container_init *, char *, const ContainerType &);
    void (*contiguous_wrapper)(container_init *, char *, const ContainerType *, size_t);

    template <typename ResType>
    void init(const ndt::type &tp, const char *metadata) {
      typedef detail::init_kernel<ResType, ContainerType> kernel;

      new (&child) kernel(tp, metadata);
      destruct_wrapper = [](container_init *self) { reinterpret_cast<kernel *>(&self->child)->~kernel(); };
      single_wrapper = [](container_init *self, char *data, const ContainerType &values) {
        reinterpret_cast<kernel *>(&self->child)->single(data, values);
      };
      contiguous_wrapper = [](container_init *self, char *data, const ContainerType *values, size_t size) {
        reinterpret_cast<kernel *>(&self->child)->contiguous(data, values, size);
      };
    }

    container_init(const ndt::type &tp, const char *metadata) {
      switch (tp.get_id()) {
      case fixed_dim_id:
        init<ndt::fixed_dim_type>(tp, metadata);
        break;
      case var_dim_id:
        init<ndt::var_dim_type>(tp, metadata);
        break;
      default:
        throw std::runtime_error("unexpected type id");
      }
    }

    ~container_init() { destruct_wrapper(this); }

    void single(char *data, const ContainerType &values) { single_wrapper(this, data, values); }

    void contiguous(char *data, const ContainerType *values, size_t size) {
      contiguous_wrapper(this, data, values, size);
    }
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
  struct init_kernel<std::initializer_list<T>> : container_init<std::initializer_list<T>> {
    using container_init<std::initializer_list<T>>::container_init;
  };

  template <typename T>
  struct init_kernel<std::vector<T>> : detail::init_kernel<ndt::fixed_dim_type, std::vector<T>> {
    using detail::init_kernel<ndt::fixed_dim_type, std::vector<T>>::init_kernel;
  };

  template <typename T, T... first, T... second>
  auto join_integer_sequence(std::integer_sequence<T, first...> DYND_UNUSED(f),
                             std::integer_sequence<T, second...> DYND_UNUSED(s)) -> decltype(auto) {
    return std::integer_sequence<T, first..., second...>();
  }

  /*
    template <typename T, T... vals>
    auto reverse_integer_sequence(std::integer_sequence<T, vals...> DYND_UNUSED(v)) {
      return reverse_integer_sequence_impl(std::integer_sequence<T, vals...>(), std::integer_sequence<T>());
    }

    template <typename T, T... done>
    auto reverse_integer_sequence_impl(std::integer_sequence<T> DYND_UNUSED(f), std::integer_sequence<T, done...>
    DYND_UNUSED(r)) -> decltype(auto) {
      return std::integer_sequence<T, done...>();
    }

    template <typename T, T first, T... remaining, T... done>
    auto reverse_integer_sequence_impl(std::integer_sequence<T, first, remaining...> DYND_UNUSED(f),
    std::integer_sequence<T, done...> DYND_UNUSED(r)) -> decltype(auto) {
      return reverse_integer_sequence_impl(std::integer_sequence<T, remaining>(), std::integer_sequence<T, first,
    done...>());
    }
  */

  template <typename T, T sum, T... done>
  auto prefix_sum_integer_sequence_impl(std::integral_constant<T, sum> DYND_UNUSED(val),
                                        std::integer_sequence<T, done...> DYND_UNUSED(l),
                                        std::integer_sequence<T> DYND_UNUSED(r)) -> decltype(auto) {
    return std::integer_sequence<T, done...>();
  }

  template <typename T, T sum, T current, T... done, T... remaining>
  auto prefix_sum_integer_sequence_impl(std::integral_constant<T, sum> DYND_UNUSED(val),
                                        std::integer_sequence<T, done...> DYND_UNUSED(l),
                                        std::integer_sequence<T, current, remaining...> DYND_UNUSED(r))
      -> decltype(auto) {
    return prefix_sum_integer_sequence_impl(std::integral_constant<T, sum + current>(),
                                            std::integer_sequence<T, done..., sum + current>(),
                                            std::integer_sequence<T, remaining...>());
  }

  template <typename T, T... vals>
  auto prefix_sum_integer_sequence(std::integer_sequence<T, vals...> DYND_UNUSED(v)) -> decltype(auto) {
    return prefix_sum_integer_sequence_impl(std::integral_constant<T, 0>(), std::integer_sequence<T>(),
                                            std::integer_sequence<T, vals...>());
  }

  template <typename T, T val>
  auto end_of_integer_sequence(std::integer_sequence<T, val> DYND_UNUSED(v)) -> decltype(auto) {
    return std::integral_constant<T, val>();
  }

  template <typename T, T val, T next, T... vals>
  auto end_of_integer_sequence(std::integer_sequence<T, val, next, vals...> DYND_UNUSED(v)) -> decltype(auto) {
    return end_of_integer_sequence(std::integer_sequence<T, next, vals...>());
  }

  template <typename T, T val, T... done>
  auto right_truncate_integer_sequence_impl(std::integer_sequence<T, done...> DYND_UNUSED(d),
                                            std::integer_sequence<T, val> DYND_UNUSED(r)) -> decltype(auto) {
    return std::integer_sequence<T, done...>();
  }

  template <typename T, T val, T next, T... done, T... remaining>
  auto right_truncate_integer_sequence_impl(std::integer_sequence<T, done...> DYND_UNUSED(d),
                                            std::integer_sequence<T, val, next, remaining...> DYND_UNUSED(r)) -> auto {
    return right_truncate_integer_sequence_impl(std::integer_sequence<T, done..., val>(),
                                                std::integer_sequence<T, next, remaining...>());
  }

  template <typename T, T... vals>
  auto right_truncate_integer_sequence(std::integer_sequence<T, vals...> v) -> decltype(auto) {
    static_assert(sizeof...(vals) > 0, "Cannot right truncate an empty integer sequence.");
    return right_truncate_integer_sequence_impl(std::integer_sequence<T>(), v);
  }

  template <typename T>
  auto right_shift_integer_sequence(std::integer_sequence<T> v) -> decltype(auto) {
    return v;
  }

  template <typename T, T val, T... vals>
  auto right_shift_integer_sequence(std::integer_sequence<T, val, vals...> v) -> decltype(auto) {
    return join_integer_sequence(std::integer_sequence<T, 0>(), right_truncate_integer_sequence(v));
  }

  template <typename T>
  auto tuple_metadata_offset(std::integer_sequence<T> DYND_UNUSED(v)) -> decltype(auto) {
    return std::integral_constant<T, 0>();
  }

  template <typename T, T val, T... vals>
  auto tuple_metadata_offset(std::integer_sequence<T, val, vals...> offsets) -> decltype(auto) {
    return end_of_integer_sequence(offsets);
  }

  template <std::size_t LastOffset, typename... ElementTypes, std::size_t... Indices, std::size_t... Offsets>
  auto construct_tuple_children(const ndt::type *&field_tp, const char *&metadata,
                                type_sequence<ElementTypes...> DYND_UNUSED(tps),
                                std::index_sequence<Indices...> DYND_UNUSED(idxs),
                                std::index_sequence<Offsets...> DYND_UNUSED(ofsts),
                                std::integral_constant<std::size_t, LastOffset> last) -> decltype(auto) {
    static_assert(sizeof...(ElementTypes) == sizeof...(Indices), "Indices and ElementTypes must have same size.");
    static_assert(sizeof...(ElementTypes) == sizeof...(Offsets), "Offsets and ElementTypes must have same size.");
    std::tuple<init_kernel<ElementTypes>...> ret({*(field_tp + Indices), metadata + Offsets}...);
    metadata += last();
    return ret;
  }

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
          children(construct_tuple_children(
              field_tp, metadata, type_sequence<ElementTypes...>(), std::make_index_sequence<sizeof...(ElementTypes)>(),
              right_shift_integer_sequence(
                  prefix_sum_integer_sequence(std::index_sequence<ndt::traits<ElementTypes>::metadata_size...>())),
              tuple_metadata_offset(
                  prefix_sum_integer_sequence(std::index_sequence<ndt::traits<ElementTypes>::metadata_size...>())))) {}

    init_kernel(const ndt::type &tp, const char *metadata)
        : init_kernel(tp.extended<ndt::tuple_type>()->get_field_types_raw(), metadata) {}

    void single(char *data, const std::tuple<ElementTypes...> &value) {
      for_each<type_sequence<ElementTypes...>, 0>(on_each(), metadata, children, data, value);
    }
  };

} // namespace dynd::nd
} // namespace dynd
