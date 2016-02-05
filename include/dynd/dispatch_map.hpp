
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
  bool supercedes(const std::array<type_id_t, N> &lhs, const std::array<type_id_t, N> &rhs)
  {
    for (size_t i = 0; i < N; ++i) {
      if (!(lhs[i] == rhs[i] || is_base_id_of(rhs[i], lhs[i]))) {
        return false;
      }
    }

    return true;
  }

  bool edge(const std::array<type_id_t, N> &u, const std::array<type_id_t, N> &v)
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
  typedef std::map<std::array<type_id_t, N>, MappedType> map_type;
  typedef typename map_type::key_type key_type;
  typedef typename map_type::value_type value_type;
  typedef typename map_type::iterator iterator;

  dispatch_map() = default;

  dispatch_map(const std::initializer_list<value_type> &values) { init(values.begin(), values.end()); }

  // pair(type_id[N], MappedType)
  template <typename Iter>
  void init(Iter begin, Iter end)
  {
    std::vector<std::array<type_id_t, N>> signatures;
    std::vector<MappedType> m;
    while (begin != end) {
      signatures.push_back(begin->first);
      m.push_back(begin->second);
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

    std::vector<MappedType> res(m.size());
    topological_sort(m, edges, res.begin());
  }

  const MappedType &at(const std::array<type_id_t, N> &key) const { return m_map.at(key); }

  void insert(const value_type &key) { m_map.insert(key); }

private:
  map_type m_map;
};

} // namespace dynd
