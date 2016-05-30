//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <memory>

// #include <sparsehash/dense_hash_map>

#include <dynd/type_registry.hpp>

namespace dynd {

typedef std::vector<ndt::type> (*dispatch_t)(const ndt::type &, size_t, const ndt::type *);

template <size_t N>
std::array<ndt::type, N> as_array(const std::vector<ndt::type> &in) {
  std::array<ndt::type, N> out;
  for (size_t i = 0; i < N; ++i) {
    out[i] = in[i];
  }

  return out;
}

template <size_t N>
bool consistent(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!is_base_id_of(lhs[i], rhs[i]) && !is_base_id_of(rhs[i], lhs[i])) {
      return false;
    }
  }

  return true;
}

template <size_t N>
bool supercedes(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!is_base_id_of(rhs[i], lhs[i])) {
      return false;
    }
  }

  return true;
}

template <size_t N>
bool ambiguous(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs) {
  return consistent(lhs, rhs) && !(supercedes(lhs, rhs) || supercedes(rhs, lhs));
}

template <size_t N>
bool consistent(const std::array<ndt::type, N> &lhs, const std::array<ndt::type, N> &rhs) {
  std::array<type_id_t, N> lhs_ids, rhs_ids;
  for (size_t i = 0; i < N; ++i) {
    lhs_ids[i] = lhs[i].get_id();
    rhs_ids[i] = rhs[i].get_id();
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!is_base_id_of(lhs_ids[i], rhs_ids[i]) && !is_base_id_of(rhs_ids[i], lhs_ids[i])) {
      return false;
    }
  }

  return true;
}

template <size_t N>
bool supercedes(const std::array<ndt::type, N> &lhs, const std::array<ndt::type, N> &rhs) {
  std::array<type_id_t, N> lhs_ids, rhs_ids;
  for (size_t i = 0; i < N; ++i) {
    lhs_ids[i] = lhs[i].get_id();
    rhs_ids[i] = rhs[i].get_id();
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!is_base_id_of(rhs_ids[i], lhs_ids[i])) {
      return false;
    }
  }

  return true;
}

template <size_t N>
bool ambiguous(const std::array<ndt::type, N> &lhs, const std::array<ndt::type, N> &rhs) {
  return consistent(lhs, rhs) && !(supercedes(lhs, rhs) || supercedes(rhs, lhs));
}

/*
template <size_t N>
bool supercedes(const type_id_t (&lhs)[N], const std::vector<type_id_t> &rhs) {
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

inline bool supercedes(size_t N, const type_id_t *lhs, const std::vector<type_id_t> &rhs) {
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
*/

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
                              Iterator &res, Iterator &res_begin) {
    if (markers[i].is_temporarily_marked()) {
      throw std::runtime_error("not a dag");
    }

    if (!markers[i].is_marked()) {
      markers[i].temporarily_mark();
      for (auto j : edges[i]) {
        topological_sort_visit(j, vertices, edges, markers, res, res_begin);
      }
      markers[i].mark();
      *res = vertices[i];
      if (res != res_begin) {
        --res;
      }
    }
  }

} // namespace dynd::detail

template <typename VertexIterator, typename EdgeIterator, typename Iterator>
void topological_sort(VertexIterator begin, VertexIterator end, EdgeIterator edges, Iterator res) {
  size_t size = end - begin;
  Iterator res_begin = res;
  res += size - 1;

  std::unique_ptr<detail::topological_sort_marker[]> markers =
      std::make_unique<detail::topological_sort_marker[]>(size);
  for (size_t i = 0; i < size; ++i) {
    detail::topological_sort_visit(i, begin, edges, markers.get(), res, res_begin);
  }
}

template <typename VertexType, typename Iterator>
void topological_sort(std::initializer_list<VertexType> vertices,
                      std::initializer_list<std::initializer_list<size_t>> edges, Iterator res) {
  topological_sort(vertices.begin(), vertices.end(), edges.begin(), res);
}

inline std::ostream &print_ids(std::ostream &o, size_t nids, const type_id_t *ids) {
  o << "(" << ids[0];
  for (size_t i = 1; i < nids; ++i) {
    o << ", " << ids[i];
  }
  o << ")";
  return o;
}

inline std::ostream &print_ids(std::ostream &o, size_t nids, const ndt::type *tps) {
  o << "(" << tps[0].get_id();
  for (size_t i = 1; i < nids; ++i) {
    o << ", " << tps[i].get_id();
  }
  o << ")";
  return o;
}

template <size_t N, typename T>
class dispatcher {
  typedef std::map<size_t, T> Map;

public:
  typedef T value_type;

  typedef std::pair<std::array<ndt::type, N>, value_type> new_pair_type;
  typedef Map map_type;
  //  typedef google::dense_hash_map<size_t, value_type> map_type;

  typedef typename std::vector<new_pair_type>::iterator iterator;
  typedef typename std::vector<new_pair_type>::const_iterator const_iterator;

private:
  std::vector<new_pair_type> m_pairs;
  map_type m_map;
  dispatch_t m_dispatch;

  static size_t hash_combine(size_t seed, type_id_t id) { return seed ^ (id + (seed << 6) + (seed >> 2)); }

  template <typename... IDTypes>
  static size_t hash_combine(size_t seed, type_id_t id0, IDTypes... ids) {
    return hash_combine(hash_combine(seed, id0), ids...);
  }

  template <size_t M>
  static size_t hash_combine(size_t seed, std::array<type_id_t, M> ids) {
    for (size_t i = 0; i < M; ++i) {
      seed = hash_combine(seed, ids[i]);
    }

    return seed;
  }

public:
  dispatcher(dispatch_t dispatch) : m_dispatch(dispatch) {}

  dispatcher(const dispatcher &other) : m_pairs(other.m_pairs), m_map(other.m_map), m_dispatch(other.m_dispatch) {}

  template <typename Iterator>
  dispatcher(dispatch_t dispatch, Iterator begin, Iterator end, const map_type &map = map_type())
      : m_map(map), m_dispatch(dispatch) {
    //    m_map.set_empty_key(uninitialized_id);

    assign(begin, end);
  }

  dispatcher(dispatch_t dispatch, std::initializer_list<new_pair_type> pairs, const map_type &map = map_type())
      : dispatcher(dispatch, pairs.begin(), pairs.end(), map) {}

  template <typename Iterator>
  void assign(Iterator begin, Iterator end) {
    m_pairs.resize(end - begin);

    std::vector<std::vector<size_t>> edges(m_pairs.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      const auto &f_i = begin[i].second;
      std::array<ndt::type, N> tp_i =
          as_array<N>(m_dispatch(f_i->get_ret_type(), f_i->get_narg(), f_i->get_arg_types().data()));

      for (size_t j = i + 1; j < edges.size(); ++j) {
        const auto &f_j = begin[j].second;
        std::array<ndt::type, N> tp_j =
            as_array<N>(m_dispatch(f_j->get_ret_type(), f_j->get_narg(), f_j->get_arg_types().data()));

        if (ambiguous(tp_i, tp_j)) {
          bool ok = false;
          for (size_t k = 0; k < edges.size(); ++k) {
            const auto &f_k = begin[k].second;
            std::array<ndt::type, N> tp_k =
                as_array<N>(m_dispatch(f_k->get_ret_type(), f_k->get_narg(), f_k->get_arg_types().data()));

            if (supercedes(tp_k, tp_i) && supercedes(tp_k, tp_j)) {
              ok = true;
            }
          }

          /*
                    if (!ok) {
                      std::stringstream ss;
                      print_ids(ss, begin[i].first.size(), begin[i].first.data());
                      ss << " and ";
                      print_ids(ss, begin[j].first.size(), begin[j].first.data());
                      ss << " are ambiguous";
                      throw std::runtime_error(ss.str());
                    }
          */
        }

        if (edge(tp_i, tp_j)) {
          edges[i].push_back(j);
        } else if (edge(tp_j, tp_i)) {
          edges[j].push_back(i);
        }
      }
    }

    topological_sort(begin, end, edges, m_pairs.begin());

    m_map.clear();
  }

  void assign(std::initializer_list<new_pair_type> pairs) { assign(pairs.begin(), pairs.end()); }

  template <typename Iterator>
  void insert(Iterator begin, Iterator end) {
    std::vector<new_pair_type> vertices = m_pairs;
    vertices.insert(vertices.end(), begin, end);

    assign(vertices.begin(), vertices.end());
  }

  void insert(const new_pair_type &pair) { insert(&pair, &pair + 1); }

  void insert(std::initializer_list<new_pair_type> pairs) { insert(pairs.begin(), pairs.end()); }

  iterator begin() { return m_pairs.begin(); }
  const_iterator begin() const { return m_pairs.begin(); }
  const_iterator cbegin() const { return m_pairs.cbegin(); }

  iterator end() { return m_pairs.end(); }
  const_iterator end() const { return m_pairs.end(); }
  const_iterator cend() const { return m_pairs.cend(); }

  template <typename... Types>
  const value_type &operator()(Types... args) {
    std::array<ndt::type, sizeof...(Types)> tps = {args...};
    std::array<type_id_t, sizeof...(Types)> ids;
    for (size_t i = 0; i < sizeof...(Types); ++i) {
      ids[i] = tps[i].get_id();
    }

    size_t key = hash(ids);

    const auto &it = m_map.find(key);
    if (it != m_map.end()) {
      return it->second;
    }

    for (const new_pair_type &pair : m_pairs) {
      if (supercedes(tps, pair.first)) {
        return m_map[key] = pair.second;
      }
    }

    std::stringstream ss;
    ss << "signature not found for (";
    for (size_t i = 0; i < sizeof...(Types); ++i) {
      ss << tps[i] << ", ";
    }
    ss << ")";

    throw std::out_of_range(ss.str());
  }

  const value_type &operator()(size_t nids, const type_id_t *ids) {
    size_t key = static_cast<size_t>(ids[0]);
    for (size_t i = 1; i < nids; ++i) {
      key = hash_combine(key, ids[i]);
    }

    const auto &it = m_map.find(key);
    if (it != m_map.end()) {
      return it->second;
    }

    for (const new_pair_type &pair : m_pairs) {
      if (supercedes(nids, ids, pair.first)) {
        return m_map[key] = pair.second;
      }
    }

    throw std::out_of_range("signature not found");
  }

  const value_type &operator()(std::initializer_list<type_id_t> ids) { return operator()(ids.size(), ids.begin()); }

  static bool edge(const std::array<type_id_t, N> &u, const std::array<type_id_t, N> &v) {
    if (supercedes(u, v)) {
      if (supercedes(v, u)) {
        return false;
      } else {
        return true;
      }
    }
    return false;
  }

  static bool edge(const std::array<ndt::type, N> &u, const std::array<ndt::type, N> &v) {
    if (supercedes(u, v)) {
      if (supercedes(v, u)) {
        return false;
      } else {
        return true;
      }
    }
    return false;
  }

  static size_t hash(type_id_t id) { return static_cast<size_t>(id); }

  template <typename... IDTypes>
  static size_t hash(type_id_t id0, IDTypes... ids) {
    return hash_combine(hash(id0), ids...);
  }

  template <size_t M>
  static size_t hash(std::array<type_id_t, M> ids) {
    std::array<type_id_t, M - 1> new_ids;
    for (size_t i = 1; i < M; ++i) {
      new_ids[i - 1] = ids[i];
    }

    return hash_combine(hash(ids[0]), new_ids);
  }
};

} // namespace dynd
