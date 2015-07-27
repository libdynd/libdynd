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

    template <class T, class... Args>
    std::unique_ptr<T> make_unique(Args &&... args)
    {
      return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template <typename DispatcherType>
    callable multidispatch(const ndt::type &tp,
                           const DispatcherType &dispatcher,
                           std::size_t data_size)
    {
      return callable::make<multidispatch_kernel<DispatcherType>>(
          tp, make_unique<DispatcherType>(dispatcher), data_size);
    }

    namespace detail {

      template <typename IteratorType, typename DispatcherType,
                typename OnNullType>
      callable
      multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                    const IteratorType &end_child,
                    const DispatcherType &dispatcher, const OnNullType &on_null)
      {
        typedef typename std::result_of<DispatcherType(
            const ndt::type &, intptr_t, const ndt::type *)>::type key_type;

        std::map<key_type, callable> children;
        std::size_t data_size = 0;

        for (IteratorType it = begin_child; it != end_child; ++it) {
          const callable &child = *it;
          if (child.is_null()) {
            continue;
          }

          std::map<string, ndt::type> tp_vars;
          if (!tp.match(child.get_array_type(), tp_vars)) {
          }

          const ndt::type &ret_tp = child.get_ret_type();
          const array &arg_tp = child.get_arg_types();

          children[dispatcher(
              ret_tp, arg_tp.get_dim_size(),
              reinterpret_cast<const ndt::type *>(arg_tp.get_data()))] = child;
          if (child.get()->data_size > data_size) {
            data_size = child.get()->data_size;
          }
        }

        return functional::multidispatch(
            tp, [ children, dispatcher, on_null ](
                    const ndt::type &dst_tp, intptr_t nsrc,
                    const ndt::type *src_tp) mutable -> callable & {
                  callable &child = children[dispatcher(dst_tp, nsrc, src_tp)];
                  if (child.is_null()) {
                    return on_null();
                  }

                  return child;
                },
            data_size);
      }

      template <typename IteratorType, typename OnNullType>
      callable
      multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                    const IteratorType &end_child, const OnNullType &on_null)
      {
        if (!tp.extended<ndt::callable_type>()->is_pos_variadic()) {
          switch (tp.extended<ndt::callable_type>()->get_npos()) {
          case 0:
            throw std::runtime_error(
                "cannot multidispatch on a function with no arguments");
          case 1:
            return multidispatch(
                tp, begin_child, end_child,
                [](const ndt::type &DYND_UNUSED(dst_tp),
                   intptr_t DYND_UNUSED(nsrc),
                   const ndt::type *src_tp) { return src_tp[0].get_type_id(); },
                on_null);
          case 2:
            return multidispatch(
                tp, begin_child, end_child,
                [](const ndt::type &DYND_UNUSED(dst_tp),
                   intptr_t DYND_UNUSED(nsrc),
                   const ndt::type *src_tp) -> std::array<type_id_t, 2> {
                  return {{src_tp[0].get_type_id(), src_tp[1].get_type_id()}};
                },
                on_null);
          case 3:
            return multidispatch(
                tp, begin_child, end_child,
                [](const ndt::type &DYND_UNUSED(dst_tp),
                   intptr_t DYND_UNUSED(nsrc),
                   const ndt::type *src_tp) -> std::array<type_id_t, 3> {
                  return {{src_tp[0].get_type_id(), src_tp[1].get_type_id(),
                           src_tp[2].get_type_id()}};
                },
                on_null);
          default:
            break;
          }
        }

        return multidispatch(tp, begin_child, end_child,
                             [](const ndt::type &DYND_UNUSED(dst_tp),
                                intptr_t nsrc, const ndt::type *src_tp) {
                               std::vector<type_id_t> key;
                               for (std::intptr_t i = 0; i < nsrc; ++i) {
                                 key.push_back(src_tp[i].get_type_id());
                               }
                               return key;
                             },
                             on_null);
      }

      template <typename IteratorType, typename OnNullType>
      callable multidispatch(const ndt::type &tp,
                             const IteratorType &begin_child,
                             const IteratorType &end_child,
                             const std::vector<intptr_t> &permutation,
                             const OnNullType &on_null)
      {
        return multidispatch(tp, begin_child, end_child,
                             [permutation](const ndt::type &DYND_UNUSED(dst_tp),
                                           std::intptr_t DYND_UNUSED(nsrc),
                                           const ndt::type *src_tp) {
                               std::vector<type_id_t> key;
                               for (std::intptr_t i : permutation) {
                                 key.push_back((src_tp + i)->get_type_id());
                               }
                               return key;
                             },
                             on_null);
      }

    } // namespace dynd::nd::functional::detail

    template <typename IteratorType>
    callable multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                           const IteratorType &end_child)
    {
      return detail::multidispatch(tp, begin_child, end_child,
                                   []() -> callable & {
        std::stringstream ss;
        ss << "no viable overload for nd::functional::multidispatch "
              "with argument types";
        throw std::runtime_error(ss.str());
      });
    }

    template <typename IteratorType>
    callable multidispatch(const IteratorType &begin_child,
                           const IteratorType &end_child)
    {
      return multidispatch(ndt::type("(...) -> Any"), begin_child, end_child);
    }

    inline callable
    multidispatch(const std::initializer_list<callable> &children)
    {
      return multidispatch(std::begin(children), std::end(children));
    }

    inline callable
    multidispatch(const ndt::type &tp,
                  const std::initializer_list<callable> &children)
    {
      return multidispatch(tp, std::begin(children), std::end(children));
    }

    template <typename IteratorType>
    callable multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const callable &default_child)
    {
      return detail::multidispatch(tp, begin_child, end_child,
                                   [default_child]() -> callable & {
        return const_cast<callable &>(default_child);
      });
    }

    template <typename IteratorType>
    callable multidispatch(const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const callable &default_child)
    {
      return multidispatch(ndt::type("(...) -> Any"), begin_child, end_child,
                           default_child);
    }

    inline callable
    multidispatch(const std::initializer_list<callable> &children,
                  const callable &default_child)
    {
      return multidispatch(std::begin(children), std::end(children),
                           default_child);
    }

    inline callable
    multidispatch(const ndt::type &tp,
                  const std::initializer_list<callable> &children,
                  const callable &default_child)
    {
      return multidispatch(tp, std::begin(children), std::end(children),
                           default_child);
    }

    template <typename IteratorType>
    callable multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const std::vector<intptr_t> &permutation)
    {
      return detail::multidispatch(tp, begin_child, end_child, permutation,
                                   []() -> callable & {
        std::stringstream ss;
        ss << "no viable overload for nd::functional::multidispatch "
              "with argument types";
        throw std::runtime_error(ss.str());
      });
    }

    template <typename IteratorType>
    callable multidispatch(const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const std::vector<intptr_t> &permutation)
    {
      return multidispatch(ndt::type("(...) -> Any"), begin_child, end_child,
                           permutation);
    }

    inline callable
    multidispatch(const std::initializer_list<callable> &children,
                  const std::vector<intptr_t> &permutation)
    {
      return multidispatch(std::begin(children), std::end(children),
                           permutation);
    }

    inline callable
    multidispatch(const ndt::type &tp,
                  const std::initializer_list<callable> &children,
                  const std::vector<intptr_t> &permutation)
    {
      return multidispatch(tp, std::begin(children), std::end(children),
                           permutation);
    }

    template <typename IteratorType>
    callable multidispatch(const ndt::type &tp, const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const callable &default_child,
                           const std::vector<intptr_t> &permutation)
    {
      return detail::multidispatch(tp, begin_child, end_child, permutation,
                                   [default_child]() -> callable & {
        return const_cast<callable &>(default_child);
      });
    }

    template <typename IteratorType>
    callable multidispatch(const IteratorType &begin_child,
                           const IteratorType &end_child,
                           const callable &default_child,
                           const std::vector<intptr_t> &permutation)
    {
      return multidispatch(ndt::type("(...) -> Any"), begin_child, end_child,
                           default_child, permutation);
    }

    inline callable
    multidispatch(const std::initializer_list<callable> &children,
                  const callable &default_child,
                  const std::vector<intptr_t> &permutation)
    {
      return multidispatch(std::begin(children), std::end(children),
                           default_child, permutation);
    }

    inline callable multidispatch(
        const ndt::type &tp, const std::initializer_list<callable> &children,
        const callable &default_child, const std::vector<intptr_t> &permutation)
    {
      return multidispatch(tp, std::begin(children), std::end(children),
                           default_child, permutation);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd