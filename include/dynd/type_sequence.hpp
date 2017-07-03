//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>
#include <type_traits>

namespace dynd {

template <typename T>
struct front;

template <typename I, size_t J>
struct at;

template <typename I, size_t J>
struct from;

template <typename I, size_t J>
struct to;

template <typename I, typename J>
struct join;

template <typename... T>
struct type_sequence {
  static constexpr size_t size() { return sizeof...(T); }
};

template <typename T0, typename... T>
struct front<type_sequence<T0, T...>> {
  typedef T0 type;
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
  typedef typename join<type_sequence<T0>, typename to<type_sequence<T...>, I - 1>::type>::type type;
};

template <typename... T, typename... U>
struct join<type_sequence<T...>, type_sequence<U...>> {
  typedef type_sequence<T..., U...> type;
};

template <typename T>
struct pop_front {
  typedef typename from<T, 1>::type type;
};

template <typename... S>
struct outer;

template <typename T0, typename... T1>
struct outer<type_sequence<T0>, type_sequence<T1...>> {
  typedef type_sequence<type_sequence<T0, T1>...> type;
};

template <typename... T0, typename... T1>
struct outer<type_sequence<type_sequence<T0...>>, type_sequence<T1...>> {
  typedef type_sequence<type_sequence<T0..., T1>...> type;
};

template <typename S0, typename S1>
struct outer<S0, S1> {
  typedef typename join<typename outer<typename to<S0, 1>::type, S1>::type,
                        typename outer<typename pop_front<S0>::type, S1>::type>::type type;
};

template <typename S0, typename S1, typename... S>
struct outer<S0, S1, S...> {
  typedef typename outer<typename outer<S0, S1>::type, S...>::type type;
};

template <typename S, typename A0, typename... A>
std::enable_if_t<S::size() == 1, void> for_each(A0 &&a0, A &&... a) {
  a0.template on_each<typename front<S>::type>(std::forward<A>(a)...);
}

template <typename S, typename... A>
std::enable_if_t<(S::size() > 1), void> for_each(A &&... a) {
  for_each<typename to<S, 1>::type>(std::forward<A>(a)...);
  for_each<typename pop_front<S>::type>(std::forward<A>(a)...);
}

template <typename S, size_t I, typename A0, typename... A>
std::enable_if_t<S::size() == 1, void> for_each(A0 &&a0, A &&... a) {
#if defined(_MSC_VER) && !defined(__clang__)
  a0.operator()<typename front<S>::type, I>(std::forward<A>(a)...);
#else
  a0.template operator()<typename front<S>::type, I>(std::forward<A>(a)...);
#endif
}

template <typename S, size_t I, typename... A>
std::enable_if_t<(S::size() > 1), void> for_each(A &&... a) {
  for_each<typename to<S, 1>::type, I>(std::forward<A>(a)...);
  for_each<typename pop_front<S>::type, I + 1>(std::forward<A>(a)...);
}

template <typename... A>
using outer_t = typename outer<A...>::type;

template <template <typename...> class T, typename TypeSequence>
struct instantiate;

template <template <typename...> class T, typename... Types>
struct instantiate<T, type_sequence<Types...>> {
  typedef T<Types...> type;
};

template <template <typename...> class T, typename TypeSequence>
using instantiate_t = typename instantiate<T, TypeSequence>::type;

} // namespace dynd
