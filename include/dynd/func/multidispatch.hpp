//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <numeric>

#include <dynd/iterator.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/kernels/multidispatch_kernel.hpp>

#include <typeinfo>

namespace dynd {

template <typename T>
struct ndim {
  static const int value = nd::detail::ndim_from_array<T>::value;
};

template <typename T>
struct Void {
  typedef void type;
};

template <typename T, typename U = void>
struct has_key_type {
  static const bool value = false;
};

template <typename T>
struct has_key_type<T, typename Void<typename T::key_type>::type> {
  static const bool value = true;
};

namespace nd {
  namespace functional {

    /**
     * Creates a multiple dispatch callable out of a set of callables. The
     * input callables must have concrete signatures.
     *
     * \param naf  The number of callables provided.
     * \param af  The array of input callables, sized ``naf``.
     */
    callable old_multidispatch(intptr_t naf, const callable *af);

    inline callable
    old_multidispatch(const std::initializer_list<callable> &children)
    {
      return old_multidispatch(children.size(), children.begin());
    }

    namespace detail {

      template <typename ContainerType,
                bool HasKeyType = has_key_type<ContainerType>::value>
      struct multidispatch_is_array_subscript;

      template <typename ContainerType>
      struct multidispatch_is_array_subscript<ContainerType, false> {
        static const bool value = false;
      };

      template <typename ContainerType>
      struct multidispatch_is_array_subscript<ContainerType, true> {
        static const bool value =
            !std::is_same<typename ContainerType::key_type, type_id_t>::value;
      };

      template <int N, typename T>
      typename std::enable_if<N == 1, const nd::callable &>::type
      multidispatch_subscript(T &children, const type_id_t *i)
      {
        return children[*i];
      }

      template <int N, typename T>
      typename std::enable_if<(N > 1), const nd::callable &>::type
      multidispatch_subscript(T &children, const type_id_t *i)
      {
        return multidispatch_subscript<N - 1>(children[*i], i + 1);
      }

      template <int N, typename T, bool UseArrayAsSubscript>
      struct multidispatch_base_static_data;

      template <int N, typename T>
      struct multidispatch_base_static_data<N, T, false> {
        T children;
        const callable &default_child;

        multidispatch_base_static_data(T &&children,
                                       const callable &default_child)
            : children(children), default_child(default_child)
        {
        }

        const callable &operator()(const std::array<type_id_t, N> &key)
        {
          const callable &child =
              multidispatch_subscript<N>(children, key.data());
          if (child.is_null()) {
            return default_child;
          }

          return child;
        }
      };

      template <int N, typename T>
      struct multidispatch_base_static_data<N, T, true> {
        T children;
        const callable &default_child;

        multidispatch_base_static_data(T &&children,
                                       const callable &default_child)
            : children(children), default_child(default_child)
        {
        }

        const callable &operator()(const std::array<type_id_t, N> &key)
        {
          const callable &child = children[key];
          if (child.is_null()) {
            return default_child;
          }

          return child;
        }
      };

      template <int N, int K, typename T>
      std::size_t
      multidispatch_match(const ndt::type &self_tp, T &&children,
                          const callable &DYND_UNUSED(default_child))
      {
        size_t data_size_max = 0;
        for (auto it = dynd::begin<K>(children), end = dynd::end<K>(children);
             it != end; ++it) {
          const callable &child = get_second_if_pair(*it);
          if (!child.is_null()) {
            std::map<string, ndt::type> tp_vars;
            if (!self_tp.match(child.get_array_type(), tp_vars)) {
              //            This needs to be reenabled, but it needs to
              //            appended keywords properly
              //            throw std::invalid_argument("could not match
              //            callables");
            }

            size_t data_size = child.get()->data_size;
            if (data_size > data_size_max) {
              data_size_max = data_size;
            }
          }
        }

        return data_size_max;
      }

    } // namespace dynd::nd::functional::detail

    template <int N, typename T,
              bool ArraySubscript = detail::multidispatch_is_array_subscript<
                  typename std::remove_reference<T>::type>::value>
    callable multidispatch(const ndt::type &self_tp, T &&children,
                           const callable &default_child,
                           const std::vector<intptr_t> &permutation)
    {
      using base_static_data =
          detail::multidispatch_base_static_data<N, T, ArraySubscript>;

      struct static_data : base_static_data {
        size_t data_size_max;
        intptr_t permutation[N];

        static_data(T &&children, const callable &default_child,
                    size_t data_size_max, const intptr_t *permutation)
            : base_static_data(std::forward<T>(children), default_child),
              data_size_max(data_size_max)
        {
          std::memcpy(this->permutation, permutation,
                      sizeof(this->permutation));
        }

        const callable &operator()(const ndt::type &dst_tp, intptr_t nsrc,
                                   const ndt::type *src_tp)
        {
          std::vector<ndt::type> tp;
          tp.push_back(dst_tp);
          for (int j = 0; j < nsrc; ++j) {
            tp.push_back(src_tp[j]);
          }
          ndt::type *new_src_tp = tp.data() + 1;

          std::array<type_id_t, N> key;
          for (int i = 0; i < N; ++i) {
            key[i] = (new_src_tp + permutation[i])->get_type_id();
          }

          return base_static_data::operator()(key);
        }
      };

      detail::multidispatch_match<N, (ArraySubscript ? 1 : N)>(
          self_tp, children, default_child);
      return callable::make<multidispatch_kernel<static_data>>(
          self_tp,
          std::make_shared<static_data>(std::forward<T>(children),
                                        default_child, 0, permutation.data()),
          0);
    }

    template <typename T>
    callable multidispatch(const ndt::type &self_tp, T &&children,
                           const callable &default_child,
                           const std::vector<intptr_t> &permutation)
    {
      typedef typename std::remove_reference<T>::type ContainerType;

      return multidispatch<ndim<ContainerType>::value>(
          self_tp, std::forward<T>(children), default_child, permutation);
    }

    template <int N, typename T>
    callable multidispatch(const ndt::type &self_tp, T &&children,
                           const callable &default_child)
    {
      std::vector<intptr_t> permutation(N);
      std::iota(permutation.begin(), permutation.end(), 0);

      return multidispatch<N>(self_tp, std::forward<T>(children), default_child,
                              permutation);
    }

    template <typename T>
    callable multidispatch(const ndt::type &self_tp, T &&children,
                           const callable &default_child)
    {
      typedef typename std::remove_reference<T>::type ContainerType;

      return multidispatch<ndim<ContainerType>::value>(
          self_tp, std::forward<T>(children), default_child);
    }

    template <int N, typename IteratorType>
    typename std::enable_if<N == 1, callable>::type
    multidispatch(const ndt::type &self_tp, const IteratorType &begin,
                  const IteratorType &end, const callable &default_child)
    {
      std::map<type_id_t, callable> children;
      for (IteratorType it = begin; it != end; ++it) {
        const callable &child = *it;

        type_id_t key = child.get_type()->get_pos_type(0).get_type_id();
        children[key] = child;
      }

      return multidispatch<1>(self_tp, std::move(children), default_child);
    }

    template <int N, typename IteratorType>
    typename std::enable_if<N != 1, callable>::type
    multidispatch(const ndt::type &self_tp, const IteratorType &begin,
                  const IteratorType &end, const callable &default_child)
    {
      std::map<std::array<type_id_t, N>, callable> children;
      for (IteratorType it = begin; it != end; ++it) {
        const callable &child = *it;

        std::array<type_id_t, N> key;
        for (int i = 0; i < N; ++i) {
          key[i] = child.get_type()->get_pos_type(i).get_type_id();
        }

        children[key] = child;
      }

      return multidispatch<N>(self_tp, std::move(children), default_child);
    }

    template <typename IteratorType>
    callable multidispatch(const ndt::type &self_tp, const IteratorType &begin,
                           const IteratorType &end,
                           const callable &default_child = callable())
    {
      switch (self_tp.extended<ndt::callable_type>()->get_npos()) {
      case 1:
        return multidispatch<1>(self_tp, begin, end, default_child);
      case 2:
        return multidispatch<2>(self_tp, begin, end, default_child);
      case 3:
        return multidispatch<3>(self_tp, begin, end, default_child);
      case 4:
        return multidispatch<4>(self_tp, begin, end, default_child);
      case 5:
        return multidispatch<5>(self_tp, begin, end, default_child);
      case 6:
        return multidispatch<6>(self_tp, begin, end, default_child);
      case 7:
        return multidispatch<7>(self_tp, begin, end, default_child);
      default:
        throw std::runtime_error("error");
      }
    }

    template <int N>
    callable multidispatch(const ndt::type &self_tp,
                           const std::initializer_list<callable> &children,
                           const callable &default_child = callable())
    {
      return multidispatch<N>(self_tp, children.begin(), children.end(),
                              default_child);
    }

    callable multidispatch(const ndt::type &self_tp,
                           const std::initializer_list<callable> &children,
                           const callable &default_child = callable());

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd