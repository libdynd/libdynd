
namespace dynd {

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

template <typename MappedType, size_t N>
class dispatch_map {
  static bool supercedes(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs)
  {
    for (size_t i = 0; i < N; ++i) {
      if (!(lhs[i] == rhs[i] || is_base_id_of(rhs[i], lhs[i]))) {
        return false;
      }
    }

    return true;
  }

  static bool edge(const std::array<type_id_t, N> &u, const std::array<type_id_t, N> &v)
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

public:
  struct key_compare {
    bool operator()(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs) const
    {
      return edge(lhs, rhs);
    }
  };

  std::vector<std::pair<std::array<type_id_t, N>, MappedType>> m_sorted;

  typedef std::map<std::array<type_id_t, N>, MappedType> map_type;
  typedef typename map_type::key_type key_type;
  typedef typename map_type::value_type value_type;
  typedef typename std::vector<std::pair<std::array<type_id_t, N>, MappedType>>::iterator iterator;

  dispatch_map() = default;

  dispatch_map(const std::initializer_list<value_type> &values) { init(values.begin(), values.end()); }

  template <typename Iter>
  void init(Iter begin, Iter end)
  {
    std::vector<std::array<type_id_t, N>> signatures;
    std::vector<std::pair<std::array<type_id_t, N>, MappedType>> m;
    while (begin != end) {
      signatures.push_back(begin->first);
      m.push_back(*begin);
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

    decltype(m) res(m.size());
    topological_sort(m, edges, res.begin());

    for (auto &val : res) {
      m_map[val.first] = val.second;
    }

    m_sorted = res;
  }

  const MappedType &at(const std::array<type_id_t, N> &key) const { return m_map.at(key); }

  iterator find(const std::array<type_id_t, N> &key)
  {
    return std::find_if(m_sorted.begin(), m_sorted.end(), [key]());
//    auto it = m_map.find(key);
  //  if (it != m_map.end()) {
    //  return it;
    //}

    /*
        for (const auto &pair : m_sorted) {
          if (supercedes(key, pair.first)) {
            //        m_map[key] = pair.second;
            return pair.second;
          }
        }
    */

    return m_sorted.end();
  }

  void insert(const value_type &key) { m_map.insert(key); }

  const MappedType &operator[](const std::array<type_id_t, N> &key) { return find(key); }

  // operator() -- insert if not found
  // operator[] -- insert
  // at() -- lookup exactly, no
  // find() -- equivalent lookup

//  decltype(auto) begin() { return m_map.begin(); }

//  decltype(auto) end() { return m_map.end(); }

private:
  map_type m_map;
};

} // namespace dynd
