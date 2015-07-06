//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

template <typename ContainerType, int N>
class iterator;

template <int N = 1, typename ContainerType>
iterator<ContainerType, N> begin(ContainerType &c);

template <int N = 1, typename ContainerType>
iterator<ContainerType, N> end(ContainerType &c);

template <typename ContainerType>
class iterator<ContainerType, 1> {
  friend iterator<ContainerType, 1> begin<1, ContainerType>(ContainerType &c);
  friend iterator<ContainerType, 1> end<1, ContainerType>(ContainerType &c);

protected:
  typedef decltype(std::begin(std::declval<ContainerType &>())) std_iterator_type;

  std_iterator_type m_current;
  std_iterator_type m_end;

  iterator(const std_iterator_type &begin, const std_iterator_type &end)
      : m_current(begin), m_end(end)
  {
  }

  iterator(const std_iterator_type &end) : m_current(end), m_end(end) {}

public:
  typedef typename std::iterator_traits<std_iterator_type>::value_type value_type;

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

  value_type &operator*() const { return *m_current; }

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
  typedef iterator<typename iterator<ContainerType, 1>::value_type, N - 1>
      base_type;

  friend iterator<ContainerType, N> begin<N, ContainerType>(ContainerType &c);
  friend iterator<ContainerType, N> end<N, ContainerType>(ContainerType &c);

protected:
  typedef decltype(std::begin(std::declval<ContainerType &>())) std_iterator_type;

  std_iterator_type m_current;
  std_iterator_type m_end;

  iterator(const std_iterator_type &begin, const std_iterator_type &end,
           typename std::iterator_traits<std_iterator_type>::value_type &front)
      : base_type(std::begin(front), std::end(front)), m_current(begin),
        m_end(end)
  {
  }

  iterator(const std_iterator_type &begin, const std_iterator_type &end)
      : iterator(begin, end, *begin)
  {
  }

  iterator(const std_iterator_type &end,
           typename std::iterator_traits<std_iterator_type>::value_type &back)
      : base_type(std::end(back)), m_current(end), m_end(end)
  {
  }

  iterator(const std_iterator_type &end) : iterator(end, *std::prev(end)) {}

public:
  typedef typename base_type::value_type value_type;

  iterator() = default;

  iterator &operator++()
  {
    base_type::operator++();
    if (base_type::m_current == base_type::m_end) {
      ++m_current;
      base_type::m_current = std::begin(*m_current);
      base_type::m_end = std::end(*m_current);
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
    return m_current == other.m_current && base_type::operator==(other);
  }

  bool operator!=(const iterator &other) const
  {
    return m_current != other.m_current || base_type::operator!=(other);
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

template <typename ContainerType, int N>
class const_iterator;

template <int N = 1, typename ContainerType>
const_iterator<ContainerType, N> begin(const ContainerType &c);

template <int N = 1, typename ContainerType>
const_iterator<ContainerType, N> end(const ContainerType &c);

template <typename ContainerType>
class const_iterator<ContainerType, 1> {
  friend const_iterator<ContainerType, 1> begin<1, ContainerType>(const ContainerType &c);
  friend const_iterator<ContainerType, 1> end<1, ContainerType>(const ContainerType &c);

protected:
  typedef decltype(std::begin(std::declval<const ContainerType &>())) std_iterator_type;

  std_iterator_type m_current;
  std_iterator_type m_end;

  const_iterator(const std_iterator_type &begin, const std_iterator_type &end)
      : m_current(begin), m_end(end)
  {
  }

  const_iterator(const std_iterator_type &end) : m_current(end), m_end(end) {}

public:
  typedef typename std::iterator_traits<std_iterator_type>::value_type value_type;

  const_iterator() = default;

  const_iterator &operator++()
  {
    ++m_current;

    return *this;
  }

  const_iterator operator++(int)
  {
    const_iterator tmp(*this);
    operator++();
    return tmp;
  }

  const value_type &operator*() const { return *m_current; }

  bool operator==(const const_iterator &other) const
  {
    return m_current == other.m_current;
  }

  bool operator!=(const const_iterator &other) const
  {
    return m_current != other.m_current;
  }
};

template <typename ContainerType, int N>
class const_iterator
    : public const_iterator<typename iterator<ContainerType, 1>::value_type, N - 1> {
  typedef const_iterator<typename const_iterator<ContainerType, 1>::value_type, N - 1>
      base_type;

  friend const_iterator<ContainerType, N> begin<N, ContainerType>(const ContainerType &c);
  friend const_iterator<ContainerType, N> end<N, ContainerType>(const ContainerType &c);

protected:
  typedef decltype(std::begin(std::declval<const ContainerType &>())) std_iterator_type;

  std_iterator_type m_current;
  std_iterator_type m_end;

  const_iterator(const std_iterator_type &begin, const std_iterator_type &end,
           const typename std::iterator_traits<std_iterator_type>::value_type &front)
      : base_type(std::begin(front), std::end(front)), m_current(begin),
        m_end(end)
  {
  }

  const_iterator(const std_iterator_type &begin, const std_iterator_type &end)
      : const_iterator(begin, end, *begin)
  {
  }

  const_iterator(const std_iterator_type &end,
           const typename std::iterator_traits<std_iterator_type>::value_type &back)
      : base_type(std::end(back)), m_current(end), m_end(end)
  {
  }

  const_iterator(const std_iterator_type &end) : const_iterator(end, *std::prev(end)) {}

public:
  typedef typename base_type::value_type value_type;

  const_iterator() = default;

  const_iterator &operator++()
  {
    base_type::operator++();
    if (base_type::m_current == base_type::m_end) {
      ++m_current;
      base_type::m_current = std::begin(*m_current);
      base_type::m_end = std::end(*m_current);
    }

    return *this;
  }

  const_iterator operator++(int)
  {
    const_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool operator==(const const_iterator &other) const
  {
    return m_current == other.m_current && base_type::operator==(other);
  }

  bool operator!=(const const_iterator &other) const
  {
    return m_current != other.m_current || base_type::operator!=(other);
  }
};

template <int N, typename ContainerType>
const_iterator<ContainerType, N> begin(const ContainerType &c)
{
  return const_iterator<ContainerType, N>(std::begin(c), std::end(c));
}

template <int N, typename ContainerType>
const_iterator<ContainerType, N> end(const ContainerType &c)
{
  return const_iterator<ContainerType, N>(std::end(c));
}

} // namespace dynd