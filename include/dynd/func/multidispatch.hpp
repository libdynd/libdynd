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

template <typename ContainerType, int N = ndim<ContainerType>::value,
          typename... ParentIteratorType>
class flat_iterator;

template <typename ContainerType>
class flat_iterator<ContainerType, 1> {
  typedef decltype(std::begin(std::declval<ContainerType>())) iterator_type;

protected:
  iterator_type m_current;
  iterator_type m_end;

public:
  typedef decltype(*std::declval<iterator_type>()) value_type;

  flat_iterator(ContainerType &c) : m_current(std::begin(c)), m_end(std::end(c))
  {
  }

  const iterator_type &end() const { return m_end; }

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

template <typename ContainerType, int N>
class flat_iterator<ContainerType, N> {
  typedef decltype(std::begin(std::declval<ContainerType>())) iterator_type;
  typedef flat_iterator<decltype(*std::declval<iterator_type>()), N - 1,
                        flat_iterator> child_iterator_type;

protected:
  iterator_type m_current;
  iterator_type m_end;
  child_iterator_type m_child;

public:
  typedef typename child_iterator_type::value_type value_type;

  flat_iterator(ContainerType &c)
      : m_current(std::begin(c)), m_end(std::end(c)), m_child(*m_current)
  {
  }

  const iterator_type &end() const { return m_end; }

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

  bool operator==(const iterator_type &other) const
  {
    return m_current == other;
  }

  bool operator!=(const iterator_type &other) const
  {
    return m_current != other;
  }
};

template <typename C, int N, typename ParentIteratorType>
class flat_iterator<C, N, ParentIteratorType> : public flat_iterator<C, N> {
  friend ParentIteratorType;

public:
  using flat_iterator<C, N>::flat_iterator;
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
      for (flat_iterator<const ContainerType, N> it(children); it != it.end();
           ++it) {
        const arrfunc &child = *it;
        if (!child.is_null()) {
          std::map<string, ndt::type> tp_vars;
          if (!self_tp.match(child.get_array_type(), tp_vars)) {
            throw std::invalid_argument("could not match arrfuncs");
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
