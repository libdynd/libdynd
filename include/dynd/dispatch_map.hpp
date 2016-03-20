//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// #include <sparsehash/dense_hash_map>

namespace dynd {

bool supercedes(type_id_t lhs, type_id_t rhs) { return is_base_id_of(rhs, lhs); }

template <size_t N>
bool supercedes(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs)
{
  for (size_t i = 0; i < N; ++i) {
    if (!is_base_id_of(rhs[i], lhs[i])) {
      return false;
    }
  }

  return true;
}

bool supercedes(const std::vector<type_id_t> &lhs, const std::vector<type_id_t> &rhs)
{
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!is_base_id_of(rhs[i], lhs[i])) {
      return false;
    }
  }

  return true;
}

template <size_t N>
bool supercedes(const type_id_t(&lhs)[N], const std::vector<type_id_t> &rhs)
{
  if (rhs.size() != N) {
    return false;
  }

  for (size_t i = 0; i < N; ++i) {
    if (!is_base_id_of(rhs[i], lhs[i])) {
      return false;
    }
  }

  return true;
}

class topological_sort_marker {
  char m_mark;

public:
  topological_sort_marker() : m_mark(0) {}

  void temporarily_mark() { m_mark = 1; }
  void mark() { m_mark = 2; }

  bool is_temporarily_marked() { return m_mark == 1; }
  bool is_marked() { return m_mark == 2; }
};

template <typename ValueType, typename ResIteratorType>
void topological_sort_visit(intptr_t i, const std::vector<ValueType> &values,
                            const std::vector<std::vector<intptr_t>> &edges,
                            std::vector<topological_sort_marker> &markers, ResIteratorType &res)
{
  if (markers[i].is_temporarily_marked()) {
    throw std::runtime_error("not a dag");
  }

  if (!markers[i].is_marked()) {
    markers[i].temporarily_mark();
    for (size_t j : edges[i]) {
      topological_sort_visit(j, values, edges, markers, res);
    }
    markers[i].mark();
    *res = values[i];
    ++res;
  }
}

template <typename ValueType, typename OutputIterator>
void topological_sort(const std::vector<ValueType> &values, const std::vector<std::vector<intptr_t>> &edges,
                      OutputIterator res)
{
  size_t size = values.size();

  std::vector<topological_sort_marker> markers(values.size());
  for (size_t i = 0; i < size; ++i) {
    topological_sort_visit(i, values, edges, markers, res);
  }
  std::reverse(res - values.size(), res);
}

template <typename T>
class dispatcher {
public:
  typedef T mapped_type;
  typedef std::pair<std::vector<type_id_t>, mapped_type> value_type;

  typedef typename std::vector<value_type>::iterator iterator;
  typedef typename std::vector<value_type>::const_iterator const_iterator;

private:
  std::vector<value_type> m_pairs;
  //  google::dense_hash_map<size_t, T> m_map;
  std::map<size_t, T> m_map;

  static size_t combine(size_t seed) { return seed; }

  static size_t combine(size_t seed, type_id_t id0) { return seed ^ (id0 + (seed << 6) + (seed >> 2)); }

public:
  dispatcher() = default;

  template <typename IteratorType>
  dispatcher(IteratorType begin, IteratorType end)
      : m_pairs(end - begin)
  {
    //    m_map.set_empty_key(uninitialized_id);

    std::vector<std::vector<type_id_t>> signatures;

    std::vector<value_type> vertices;
    while (begin != end) {
      signatures.push_back(begin->first);
      vertices.push_back(*begin);
      ++begin;
    }

    std::vector<std::vector<intptr_t>> edges(signatures.size());
    for (size_t i = 0; i < signatures.size(); ++i) {
      for (size_t j = 0; j < signatures.size(); ++j) {
        if (edge(signatures[i], signatures[j])) {
          edges[i].push_back(j);
        }
      }
    }

    topological_sort(vertices, edges, m_pairs.begin());
  }

  dispatcher(const std::initializer_list<value_type> &pairs) : dispatcher(pairs.begin(), pairs.end()) {}

  iterator begin() { return m_pairs.begin(); }
  const_iterator begin() const { return m_pairs.begin(); }
  const_iterator cbegin() const { return m_pairs.cbegin(); }

  iterator end() { return m_pairs.end(); }
  const_iterator end() const { return m_pairs.end(); }
  const_iterator cend() const { return m_pairs.cend(); }

  template <typename... U>
  mapped_type operator()(type_id_t id0, U... ids)
  {
    size_t key = combine(static_cast<size_t>(id0), ids...);

    const auto &it = m_map.find(key);
    if (it != m_map.end()) {
      return it->second;
    }

    type_id_t signature[sizeof...(U) + 1] = {id0, ids...};
    for (const auto &pair : m_pairs) {
      if (supercedes(signature, pair.first)) {
        return m_map[key] = pair.second;
      }
    }

    throw std::out_of_range("signature not found");
  }

  static bool edge(const std::vector<type_id_t> &u, const std::vector<type_id_t> &v)
  {
    if (supercedes(u, v)) {
      if (supercedes(v, u)) {
        return false;
      }
      else {
        return true;
      }
    }

    return false;
  }
};

} // namespace dynd
