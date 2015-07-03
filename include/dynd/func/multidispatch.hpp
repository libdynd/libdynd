//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <numeric>

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/multidispatch_kernel.hpp>

namespace dynd {

template <typename T>
struct ndim {
  static const int value = nd::detail::ndim_from_array<T>::value;
};

template <typename ContainerType, int N = ndim<ContainerType>::value>
class flat_iterator;

template <typename T>
class flat_iterator<T, 1> {
public:
  T m_current;
  T m_end;

  typedef decltype(*m_current) value_type;

  flat_iterator(T current, T end) : m_current(current), m_end(end) {}

  flat_iterator &operator++()
  {
    ++m_current;

    return *this;
  }

  flat_iterator operator++(int)
  {
    flat_iterator tmp(*this);
    operator++();
    return tmp;
  }

  value_type operator*() const { return *m_current; }
};

template <typename T, int N>
class flat_iterator {
  // typedef typename std::remove_reference<
  //  decltype(std::declval<ContainerType>()[0])>::type value_type;

  //  flat_iterator<value_type, N - 1> m_current;
  //  const ContainerType &m_data;
public:
  //  typedef std::is_array<std::remove_pointer<T>

  T m_current;
  T m_end;
  flat_iterator<decltype(std::begin(*std::declval<T>())), N - 1> m_child;
 
  typedef typename decltype(m_child)::value_type value_type;

  flat_iterator(T begin, T end)
      : m_current(begin), m_end(end),
        m_child(std::begin(*m_current), std::end(*m_current))
  {
  }

  flat_iterator &operator++()
  {
    ++m_child;
    if (m_child.m_current == m_child.m_end) {
      ++m_current;
      m_child.m_current = std::begin(*m_current);
      m_child.m_end = std::end(*m_current);
    }

    return *this;
  }

  flat_iterator operator++(int)
  {
    flat_iterator tmp(*this);
    operator++();
    return tmp;
  }

  value_type operator*() const { return *m_child; }

  bool operator==(const T &other) const { return m_current == other; }

  bool operator!=(const T &other) const { return m_current != other; }

  void test() {}
};

/*
template <typename T, int N = ndim<T>::value>
flat_iterator<T, N> make_flat_iterator(T &data) {
  return flat_iterator<T, N>(std::begin(data), std::end(data));
}
*/

namespace nd {
  namespace functional {

    /**
     * Creates a multiple dispatch arrfunc out of a set of arrfuncs. The
     * input arrfuncs must have concrete signatures.
     *
     * \param naf  The number of arrfuncs provided.
     * \param af  The array of input arrfuncs, sized ``naf``.
     */
    arrfunc multidispatch(intptr_t naf, const arrfunc *af);

    inline arrfunc multidispatch(const std::initializer_list<arrfunc> &children)
    {
      return multidispatch(children.size(), children.begin());
    }

    arrfunc multidispatch(const ndt::type &self_tp,
                          const std::vector<arrfunc> &children,
                          const std::vector<std::string> &ignore_vars);

    arrfunc multidispatch(const ndt::type &self_tp,
                          const std::vector<arrfunc> &children);

    arrfunc multidispatch(const ndt::type &self_tp, intptr_t size,
                          const arrfunc *children, const arrfunc &default_child,
                          bool own_children, intptr_t i0 = 0);

    arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                     const std::vector<arrfunc> &children);

    inline arrfunc multidispatch_by_type_id(const ndt::type &self_tp,
                                            intptr_t size,
                                            const arrfunc *children,
                                            const arrfunc &default_child,
                                            bool own_children, intptr_t i0 = 0)
    {
      return multidispatch(self_tp, size, children, default_child, own_children,
                           i0);
    }

    template <int N0>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const arrfunc (&children)[N0],
                          const arrfunc &default_child, intptr_t i0 = 0)
    {
      return multidispatch(self_tp, N0, children, default_child, false, i0);
    }

    template <typename ContainerType, int N = ndim<ContainerType>::value>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const ContainerType &children,
                          const arrfunc &DYND_UNUSED(default_child),
                          const std::vector<intptr_t> &permutation)
    {
      flat_iterator<decltype(std::begin(children)), N> it(std::begin(children),
                                                          std::end(children));

      for (; it != std::end(children); ++it) {
        const arrfunc &child = *it;
        if (!child.is_null()) {
          std::cout << child << std::endl;
        }
      }

      for (auto &row : children) {
        for (auto &child : row) {
          if (!child.is_null()) {
            std::map<string, ndt::type> tp_vars;
            if (!self_tp.match(child.get_array_type(), tp_vars)) {
              throw std::invalid_argument("could not match arrfuncs");
            }
          }
        }
      }

      struct static_data {
        const ContainerType &children;
        intptr_t permutation[N];

        static_data(const ContainerType &children, const intptr_t *permutation)
            : children(children)
        {
          std::memcpy(this->permutation, permutation,
                      sizeof(this->permutation));
        }

        arrfunc operator()(const ndt::type &dst_tp, intptr_t nsrc,
                           const ndt::type *src_tp) const
        {
          std::vector<ndt::type> tp;
          tp.push_back(dst_tp);
          for (int j = 0; j < nsrc; ++j) {
            tp.push_back(src_tp[j]);
          }
          ndt::type *new_src_tp = tp.data() + 1;

          intptr_t index[N];
          for (intptr_t j = 0; j < N; ++j) {
            index[j] = new_src_tp[permutation[j]].get_type_id();
          }

          return at(children, index);
        }
      };

      return arrfunc::make<multidispatch_kernel<static_data>>(
          self_tp, static_data(children, permutation.data()), 0);
    }

    template <typename ContainerType, int N = ndim<ContainerType>::value>
    arrfunc multidispatch(const ndt::type &self_tp,
                          const ContainerType &children,
                          const arrfunc &default_child)
    {
      std::vector<intptr_t> permutation(N);
      std::iota(permutation.begin(), permutation.end(), 0);

      return multidispatch(self_tp, children, default_child, permutation);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
