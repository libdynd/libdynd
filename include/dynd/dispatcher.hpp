//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

// #include <sparsehash/dense_hash_map>

#include <dynd/types/type_id.hpp>

namespace dynd {

inline bool supercedes(const std::vector<type_id_t> &lhs, const std::vector<type_id_t> &rhs)
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

inline bool supercedes(size_t N, const type_id_t *lhs, const std::vector<type_id_t> &rhs)
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

namespace detail {

  class topological_sort_marker {
    char m_mark;

  public:
    topological_sort_marker() : m_mark(0) {}

    void temporarily_mark() { m_mark = 1; }
    void mark() { m_mark = 2; }

    bool is_temporarily_marked() { return m_mark == 1; }
    bool is_marked() { return m_mark == 2; }
  };

  template <typename VertexIterator, typename EdgeIterator, typename MarkerIterator, typename Iterator>
  void topological_sort_visit(size_t i, VertexIterator vertices, EdgeIterator edges, MarkerIterator markers,
                              Iterator &res)
  {
    if (markers[i].is_temporarily_marked()) {
      throw std::runtime_error("not a dag");
    }

    if (!markers[i].is_marked()) {
      markers[i].temporarily_mark();
      for (auto j : edges[i]) {
        topological_sort_visit(j, vertices, edges, markers, res);
      }
      markers[i].mark();
      *res = vertices[i];
      --res;
    }
  }

} // namespace dynd::detail

template <typename VertexIterator, typename EdgeIterator, typename Iterator>
void topological_sort(VertexIterator begin, VertexIterator end, EdgeIterator edges, Iterator res)
{
  size_t size = end - begin;
  res += size - 1;

  std::unique_ptr<detail::topological_sort_marker[]> markers =
      std::make_unique<detail::topological_sort_marker[]>(size);
  for (size_t i = 0; i < size; ++i) {
    detail::topological_sort_visit(i, begin, edges, markers.get(), res);
  }
}

template <typename VertexType, typename Iterator>
void topological_sort(std::initializer_list<VertexType> vertices,
                      std::initializer_list<std::initializer_list<size_t>> edges, Iterator res)
{
  topological_sort(vertices.begin(), vertices.end(), edges.begin(), res);
}

template <typename T, typename Map = std::map<size_t, T>>
class dispatcher {
public:
  typedef T value_type;

  typedef std::pair<std::vector<type_id_t>, value_type> pair_type;
  typedef Map map_type;
  //  typedef google::dense_hash_map<size_t, value_type> map_type;

  typedef typename std::vector<pair_type>::iterator iterator;
  typedef typename std::vector<pair_type>::const_iterator const_iterator;

private:
  std::vector<pair_type> m_pairs;
  map_type m_map;

  static size_t hash_combine(size_t seed, type_id_t id) { return seed ^ (id + (seed << 6) + (seed >> 2)); }

  template <typename... IDTypes>
  static size_t hash_combine(size_t seed, type_id_t id0, IDTypes... ids)
  {
    return hash_combine(hash_combine(seed, id0), ids...);
  }

public:
  dispatcher() = default;

  template <typename Iterator>
  dispatcher(Iterator begin, Iterator end, const map_type &map = map_type())
      : m_map(map)
  {
    //    m_map.set_empty_key(uninitialized_id);

    assign(begin, end);
  }

  dispatcher(std::initializer_list<pair_type> pairs, const map_type &map = map_type())
      : dispatcher(pairs.begin(), pairs.end(), map)
  {
  }

  template <typename Iterator>
  void assign(Iterator begin, Iterator end)
  {
    m_pairs.resize(end - begin);

    std::vector<std::vector<size_t>> edges(m_pairs.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      for (size_t j = i + 1; j < edges.size(); ++j) {
        if (edge(begin[i].first, begin[j].first)) {
          edges[i].push_back(j);
        }
        else if (edge(begin[j].first, begin[i].first)) {
          edges[j].push_back(i);
        }
      }
    }

    topological_sort(begin, end, edges, m_pairs.begin());

    m_map.clear();
  }

  void assign(std::initializer_list<pair_type> pairs) { assign(pairs.begin(), pairs.end()); }

  template <typename Iterator>
  void insert(Iterator begin, Iterator end)
  {
    std::vector<pair_type> vertices = m_pairs;
    vertices.insert(vertices.end(), begin, end);

    assign(vertices.begin(), vertices.end());
  }

  void insert(const pair_type &pair) { insert(&pair, &pair + 1); }

  void insert(std::initializer_list<pair_type> pairs) { insert(pairs.begin(), pairs.end()); }

  iterator begin() { return m_pairs.begin(); }
  const_iterator begin() const { return m_pairs.begin(); }
  const_iterator cbegin() const { return m_pairs.cbegin(); }

  iterator end() { return m_pairs.end(); }
  const_iterator end() const { return m_pairs.end(); }
  const_iterator cend() const { return m_pairs.cend(); }

  template <typename... IDTypes>
  const value_type &operator()(IDTypes... ids)
  {
    size_t key = hash(ids...);

    const auto &it = m_map.find(key);
    if (it != m_map.end()) {
      return it->second;
    }

    for (const pair_type &pair : m_pairs) {
      if (supercedes({ids...}, pair.first)) {
        return m_map[key] = pair.second;
      }
    }

    throw std::out_of_range("signature not found");
  }

  const value_type &operator()(size_t nids, const type_id_t *ids)
  {
    size_t key = static_cast<size_t>(ids[0]);
    for (size_t i = 1; i < nids; ++i) {
      key = hash_combine(key, ids[i]);
    }

    const auto &it = m_map.find(key);
    if (it != m_map.end()) {
      return it->second;
    }

    for (const pair_type &pair : m_pairs) {
      if (supercedes(nids, ids, pair.first)) {
        return m_map[key] = pair.second;
      }
    }

    throw std::out_of_range("signature not found");
  }

  const value_type &operator()(std::initializer_list<type_id_t> ids) { return operator()(ids.size(), ids.begin()); }

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

  static size_t hash(type_id_t id) { return static_cast<size_t>(id); }

  template <typename... IDTypes>
  static size_t hash(type_id_t id0, IDTypes... ids)
  {
    return hash_combine(hash(id0), ids...);
  }
};

} // namespace dynd
