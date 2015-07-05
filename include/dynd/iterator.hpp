//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {

template <class T>
struct iterator_for {
  typedef typename T::iterator type;
};

template <class T>
struct iterator_for<T *> {
  typedef T *type;
};

template <class T, std::size_t N>
struct iterator_for<T (&)[N]> {
  typedef T *type;
};

template <typename ContainerType, int N>
class iterator;

template <int N = 1, typename ContainerType>
iterator<ContainerType, N> begin(ContainerType &c);

template <int N = 1, typename ContainerType>
iterator<ContainerType, N> end(ContainerType &c);

template <typename ContainerType>
class iterator<ContainerType, 1> {
public:
//  typedef typename std::remove_reference<
  //    decltype(std::begin(std::declval<ContainerType>()))>::type iterator_type;

  typedef typename iterator_for<ContainerType &>::type iterator_type;

  friend iterator<ContainerType, 1> begin<1, ContainerType>(ContainerType &c);
  friend iterator<ContainerType, 1> end<1, ContainerType>(ContainerType &c);

protected:
  iterator_type m_current;
  iterator_type m_end;

  iterator(const iterator_type &begin, const iterator_type &end)
      : m_current(begin), m_end(end)
  {
  }

  iterator(const iterator_type &end) : m_current(end), m_end(end) {}

public:
  typedef decltype(*std::declval<iterator_type>()) value_type;

  iterator() = default;

  iterator &operator++()
  {
    ++m_current;

    return *this;
  }

  iterator operator++(int)
  {
    iterator tmp(*this);
    operator++();
    return tmp;
  }

  value_type operator*() const { return *m_current; }

  bool operator==(const iterator &other) const
  {
    return m_current == other.m_current;
  }

  bool operator!=(const iterator &other) const
  {
    return m_current != other.m_current;
  }
};

template <typename ContainerType, int N>
class iterator
    : public iterator<typename iterator<ContainerType, 1>::value_type, N - 1> {
public:
  typedef iterator<typename iterator<ContainerType, 1>::value_type, N - 1>
      parent_type;
//  typedef decltype(std::begin(std::declval<ContainerType &>())) iterator_type;
  typedef typename iterator_for<ContainerType &>::type iterator_type;

  friend iterator<ContainerType, N> begin<N, ContainerType>(ContainerType &c);
  friend iterator<ContainerType, N> end<N, ContainerType>(ContainerType &c);

protected:
  iterator_type m_current;
  iterator_type m_end;

  iterator(const iterator_type &begin, const iterator_type &end,
           const typename iterator<ContainerType, 1>::value_type &front)
      : parent_type(std::begin(front), std::end(front)), m_current(begin),
        m_end(end)
  {
  }

  iterator(const iterator_type &begin, const iterator_type &end)
      : iterator(begin, end, *begin)
  {
  }

  iterator(const iterator_type &end,
           const typename iterator<ContainerType, 1>::value_type &back)
      : parent_type(std::end(back)), m_current(end), m_end(end)
  {
  }

  iterator(const iterator_type &end) : iterator(end, *std::prev(end)) {}

public:
  typedef typename parent_type::value_type value_type;

  iterator() = default;

  iterator &operator++()
  {
    parent_type::operator++();
    if (parent_type::m_current == parent_type::m_end) {
      ++m_current;
      parent_type::m_current = std::begin(*m_current);
      parent_type::m_end = std::end(*m_current);
    }

    return *this;
  }

  iterator operator++(int)
  {
    iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool operator==(const iterator &other) const
  {
    return m_current == other.m_current && parent_type::operator==(other);
  }

  bool operator!=(const iterator &other) const
  {
    return m_current != other.m_current || parent_type::operator!=(other);
  }
};

template <int N, typename ContainerType>
iterator<ContainerType, N> begin(ContainerType &c)
{
  return iterator<ContainerType, N>(std::begin(c), std::end(c));
}

template <int N, typename ContainerType>
iterator<ContainerType, N> end(ContainerType &c)
{
  return iterator<ContainerType, N>(std::end(c));
}

} // namespace dynd