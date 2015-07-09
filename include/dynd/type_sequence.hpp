//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>
#include <type_traits>

namespace dynd {

template <typename T>
struct front;

/**
 * Metafunction that returns the last type of a type_sequence<T...>.
 *
 * "back<type_sequence<int, float, double>>::type" is "double"
 */
template <typename T>
struct back;

/**
 * Metafunction that returns the second last type of a type_sequence<T...>.
 *
 * "second_back<type_sequence<int, float, double>>::type" is "float"
 */
template <typename T>
struct second_back;

template <typename I, size_t J>
struct at;

template <typename I, size_t J>
struct from;

template <typename I, size_t J>
struct to;

template <typename I, typename J>
struct join;

template <typename I, typename J>
struct take;

template <typename... I>
struct alternate;

template <typename T, T... I>
struct integer_sequence {
  //  static_assert(std::is_integral<T>::value,
  //              "integer_sequence must be instantiated with an integral
  //              type");

  enum { size = sizeof...(I) };
  typedef T type;

  static const std::array<T, size> &vals()
  {
    static const std::array<T, size> vals = {{I...}};
    return vals;
  }
};

template <size_t... I>
using index_sequence = integer_sequence<size_t, I...>;

template <typename T, T I0, T... I>
struct at<integer_sequence<T, I0, I...>, 0> {
  enum { value = I0 };
};

template <typename T, T I0, T... I, size_t J>
struct at<integer_sequence<T, I0, I...>, J> {
  enum { value = at<integer_sequence<T, I...>, J - 1>::value };
};

template <typename T, T... I, T... J>
struct join<integer_sequence<T, I...>, integer_sequence<T, J...>> {
  typedef integer_sequence<T, I..., J...> type;
};

namespace detail {

  template <int flags, typename T, T...>
  struct make_integer_sequence;

  template <int flags, typename T, T stop>
  struct make_integer_sequence<flags, T, stop>
      : make_integer_sequence<flags, T, 0, stop> {
  };

  template <int flags, typename T, T start, T stop>
  struct make_integer_sequence<flags, T, start, stop>
      : make_integer_sequence<flags, T, start, stop, 1> {
  };

  template <typename T, T start, T stop, T step>
  struct make_integer_sequence<-1, T, start, stop, step>
      : make_integer_sequence<(start < stop), T, start, stop, step> {
  };

  template <typename T, T start, T stop, T step>
  struct make_integer_sequence<0, T, start, stop, step> {
    typedef integer_sequence<T> type;
  };

  template <typename T, T start, T stop, T step>
  struct make_integer_sequence<1, T, start, stop, step> {
    enum { next = start + step };

    typedef typename join < integer_sequence<T, start>,
        typename make_integer_sequence<
            next<stop, T, next, stop, step>::type>::type type;
  };

} // namespace detail

template <typename T, T... I>
using make_integer_sequence =
    typename detail::make_integer_sequence<-1, T, I...>::type;

// Fails in MSVC 2015 CTP 6 with two levels of alias, therefore don't go through
// make_integer_sequence alias, use detail::make_integer_sequence directly.
// https://connect.microsoft.com/VisualStudio/feedback/details/1200294/code-with-two-levels-of-alias-templates-and-variadic-packs-fails-to-compile
// template <size_t... I>
// using make_index_sequence = make_integer_sequence<size_t, I...>;
template <size_t... I>
using make_index_sequence =
    typename detail::make_integer_sequence<-1, size_t, I...>::type;

template <typename T, T I0>
struct front<integer_sequence<T, I0>> {
  enum { value = I0 };
};

template <typename T, T I0, T... I>
struct front<integer_sequence<T, I0, I...>> {
  enum { value = front<integer_sequence<T, I...>>::value };
};

template <typename T, T I0, T... I>
struct from<integer_sequence<T, I0, I...>, 0> {
  typedef integer_sequence<T, I0, I...> type;
};

template <typename T>
struct from<integer_sequence<T>, 0> {
  typedef integer_sequence<T> type;
};

template <typename T, T I0, T... I>
struct from<integer_sequence<T, I0, I...>, 1> {
  typedef integer_sequence<T, I...> type;
};

template <typename T, T I0, T... I, size_t J>
struct from<integer_sequence<T, I0, I...>, J> {
  typedef typename from<integer_sequence<T, I...>, J - 1>::type type;
};

template <typename... T>
struct type_sequence {
  enum { size = sizeof...(T) };
};

template <typename T>
struct back<type_sequence<T>> {
  typedef T type;
};

template <typename T0, typename... T>
struct back<type_sequence<T0, T...>> {
  typedef typename back<type_sequence<T...>>::type type;
};

template <typename S, typename T>
struct second_back<type_sequence<S, T>> {
  typedef S type;
};

template <typename T0, typename T1, typename... T>
struct second_back<type_sequence<T0, T1, T...>> {
  typedef typename second_back<type_sequence<T1, T...>>::type type;
};

template <typename T0, typename... T>
struct at<type_sequence<T0, T...>, 0> {
  typedef T0 type;
};

template <size_t I, typename T0, typename... T>
struct at<type_sequence<T0, T...>, I> {
  typedef typename at<type_sequence<T...>, I - 1>::type type;
};

template <typename T0, typename... T>
struct from<type_sequence<T0, T...>, 0> {
  typedef type_sequence<T0, T...> type;
};

template <>
struct from<type_sequence<>, 0> {
  typedef type_sequence<> type;
};

template <typename T0, typename... T>
struct from<type_sequence<T0, T...>, 1> {
  typedef type_sequence<T...> type;
};

template <typename T0, typename... T, size_t I>
struct from<type_sequence<T0, T...>, I> {
  typedef typename from<type_sequence<T...>, I - 1>::type type;
};

template <typename T0, typename... T>
struct to<type_sequence<T0, T...>, 0> {
  typedef type_sequence<> type;
};

template <>
struct to<type_sequence<>, 0> {
  typedef type_sequence<> type;
};

template <typename T0, typename... T>
struct to<type_sequence<T0, T...>, 1> {
  typedef type_sequence<T0> type;
};

template <typename T0, typename... T, size_t I>
struct to<type_sequence<T0, T...>, I> {
  typedef typename join<type_sequence<T0>, typename to<type_sequence<T...>,
                                                       I - 1>::type>::type type;
};

template <typename... T, typename... U>
struct join<type_sequence<T...>, type_sequence<U...>> {
  typedef type_sequence<T..., U...> type;
};

template <typename... T, size_t... I>
struct take<type_sequence<T...>, index_sequence<I...>> {
  typedef type_sequence<typename at<type_sequence<T...>, I>::type...> type;
};

template <typename T>
struct alternate<integer_sequence<T>, integer_sequence<T>> {
  typedef integer_sequence<T> type;
};

// This case shouldn't be necessary, but was added to work around bug:
// https://connect.microsoft.com/VisualStudio/feedback/details/1045397/recursive-variadic-template-metaprogram-ice-when-pack-gets-to-empty-size
template <typename T, T I0, T J0>
struct alternate<integer_sequence<T, I0>, integer_sequence<T, J0>> {
  typedef integer_sequence<T, I0, J0> type;
};

template <typename T, T I0, T... I, T J0, T... J>
struct alternate<integer_sequence<T, I0, I...>, integer_sequence<T, J0, J...>> {
  typedef typename join<
      integer_sequence<T, I0, J0>,
      typename alternate<integer_sequence<T, I...>,
                         integer_sequence<T, J...>>::type>::type type;
};

template <typename T>
struct alternate<integer_sequence<T>, integer_sequence<T>, integer_sequence<T>,
                 integer_sequence<T>> {
  typedef integer_sequence<T> type;
};

// Another workaround
template <typename T, T I0, T J0, T K0, T L0>
struct alternate<integer_sequence<T, I0>, integer_sequence<T, J0>,
                 integer_sequence<T, K0>, integer_sequence<T, L0>> {
  typedef integer_sequence<T, I0, J0, K0, L0> type;
};

template <typename T, T I0, T... I, T J0, T... J, T K0, T... K, T L0, T... L>
struct alternate<integer_sequence<T, I0, I...>, integer_sequence<T, J0, J...>,
                 integer_sequence<T, K0, K...>, integer_sequence<T, L0, L...>> {
  typedef typename join<
      integer_sequence<T, I0, J0, K0, L0>,
      typename alternate<integer_sequence<T, I...>, integer_sequence<T, J...>,
                         integer_sequence<T, K...>,
                         integer_sequence<T, L...>>::type>::type type;
};

template <template <typename...> class F, typename T, bool flatten = false>
struct for_each;

template <template <typename...> class F, typename T0>
struct for_each<F, type_sequence<T0>, false> {
  typedef type_sequence<F<T0>> type;
};

template <template <typename...> class F, typename T0, typename... T>
struct for_each<F, type_sequence<T0, T...>, false> {
  typedef type_sequence<F<T0>, F<T>...> type;
};

template <template <typename...> class F, typename... T0>
struct for_each<F, type_sequence<type_sequence<T0...>>, true> {
  typedef type_sequence<F<T0...>> type;
};

template <template <typename...> class F, typename... T0, typename... T>
struct for_each<F, type_sequence<type_sequence<T0...>, T...>, true> {
  typedef typename join<
      type_sequence<F<T0...>>,
      typename for_each<F, type_sequence<T...>, true>::type>::type type;
};

template <typename T>
struct type_proxy;

#if defined(_MSC_VER) && (_MSC_VER == 1800) && !defined(__CUDACC__)
// Workaround empty case for MSVC 2013.
template <>
struct type_proxy<type_sequence<>> {
  template <typename F, typename... A>
  static typename std::result_of<F(A...)>::type apply(F f, A &&... a)
  {
    return f.apply(std::forward<A>(a)...);
  }
};
#endif // _MSC_VER

template <typename... T>
struct type_proxy<type_sequence<T...>> {
  template <typename F, typename... A>
  static typename std::result_of<F(A...)>::type apply(F f, A &&... a)
  {
    return f.template apply<T...>(std::forward<A>(a)...);
  }
};

namespace detail {
  template <size_t I, typename... A>
  struct get_impl;

  template <typename A0, typename... A>
  struct get_impl<0, A0, A...> {
    typedef A0 result_type;
    static result_type get(A0 &&a0, A &&...) { return a0; }
  };

  template <size_t I, typename A0, typename... A>
  struct get_impl<I, A0, A...> {
    typedef typename get_impl<I - 1, A...>::result_type result_type;
    static result_type get(A0 &&, A &&... a)
    {
      return get_impl<I - 1, A...>::get(std::forward<A>(a)...);
    }
  };
} // namespace detail

template <size_t I, typename... A>
typename detail::get_impl<I, A...>::result_type get(A &&... a)
{
  return detail::get_impl<I, A...>::get(std::forward<A>(a)...);
}

} // namespace dynd
