//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {

template <typename T>
struct ndim {
  static const int value = nd::detail::ndim_from_array<T>::value;
};

template <typename ContainerType, int N = ndim<ContainerType>::value>
class flat_iterator;

template <int N = 1, typename ContainerType>
flat_iterator<ContainerType, N> begin(ContainerType &c);

template <int N = 1, typename ContainerType>
flat_iterator<ContainerType, N> end(ContainerType &c);

template <typename ContainerType>
class flat_iterator<ContainerType, 1> {
  typedef decltype(std::begin(std::declval<ContainerType &>())) iterator_type;

  friend flat_iterator<ContainerType, 1>
  begin<1, ContainerType>(ContainerType &c);
  friend flat_iterator<ContainerType, 1>
  end<1, ContainerType>(ContainerType &c);

protected:
  iterator_type m_current;
  iterator_type m_end;

  flat_iterator(const iterator_type &current, const iterator_type &end)
      : m_current(current), m_end(end)
  {
  }

public:
  typedef decltype(*std::declval<iterator_type>()) value_type;

  flat_iterator() = default;

  flat_iterator(ContainerType &c) : m_current(std::begin(c)), m_end(std::end(c))
  {
  }

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

  bool operator==(const flat_iterator &other) const
  {
    return m_current == other.m_current;
  }

  bool operator!=(const flat_iterator &other) const
  {
    return m_current != other.m_current;
  }
};

template <typename T>
struct element_of {
  typedef decltype(std::declval<T &>()[0]) type;
};

template <typename ContainerType, int N>
class flat_iterator
    : public flat_iterator<typename element_of<ContainerType>::type, N - 1> {
  typedef flat_iterator<typename element_of<ContainerType>::type, N - 1>
      parent_type;
  typedef decltype(std::begin(std::declval<ContainerType &>())) iterator_type;

  friend flat_iterator<ContainerType, N>
  begin<N, ContainerType>(ContainerType &c);
  friend flat_iterator<ContainerType, N>
  end<N, ContainerType>(ContainerType &c);

protected:
  iterator_type m_current;
  iterator_type m_end;

  flat_iterator(const iterator_type &current, const iterator_type &end)
      : parent_type(std::begin(*current), std::end(*current)),
        m_current(current), m_end(end)
  {
  }

public:
  typedef typename parent_type::value_type value_type;

  flat_iterator() = default;

  flat_iterator(ContainerType &c)
      : parent_type(*std::begin(c)), m_current(std::begin(c)),
        m_end(std::end(c))
  {
  }

  const iterator_type &end() const { return m_end; }

  flat_iterator &operator++()
  {
    parent_type::operator++();
    if (parent_type::m_current == parent_type::m_end) {
      ++m_current;
      parent_type::m_current = std::begin(*m_current);
      parent_type::m_end = std::end(*m_current);
    }

    return *this;
  }

  flat_iterator operator++(int)
  {
    flat_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool operator==(const flat_iterator &other) const
  {
    return m_current == other.m_current && parent_type::operator==(other);
  }

  bool operator!=(const flat_iterator &other) const
  {
    return m_current != other.m_current || parent_type::operator!=(other);
  }

  bool operator==(const iterator_type &other) const
  {
    return m_current == other;
  }

  bool operator!=(const iterator_type &other) const
  {
    return m_current != other;
  }
};

template <int N, typename ContainerType>
flat_iterator<ContainerType, N> begin(ContainerType &c)
{
  return flat_iterator<ContainerType, N>(std::begin(c), std::end(c));
}

template <int N, typename ContainerType>
flat_iterator<ContainerType, N> end(ContainerType &c)
{
  return flat_iterator<ContainerType, N>(std::end(c), std::end(c));
}

} // namespace dynd