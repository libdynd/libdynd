
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

namespace detail {

  template <typename KeyType, typename MappedType>
  class dispatch_map {
  public:
    typedef KeyType key_type;
    typedef MappedType mapped_type;
    typedef std::pair<key_type, mapped_type> value_type;

    typedef typename std::vector<value_type>::iterator iterator;
    typedef typename std::vector<value_type>::const_iterator const_iterator;

  private:
    std::vector<value_type> m_values;
    std::map<key_type, iterator> m_cache;

  public:
    dispatch_map() = default;

    dispatch_map(const std::initializer_list<value_type> &values) { init(values.begin(), values.end()); }

    template <typename Iter>
    void init(Iter begin, Iter end)
    {
      std::vector<key_type> signatures;
      std::vector<value_type> m;
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

      m_values = res;
    }

    iterator find(const key_type &key)
    {

      auto it = m_cache.find(key);
      if (it != m_cache.end()) {
        return it->second;
      }

      return m_cache[key] = std::find_if(m_values.begin(), m_values.end(),
                                         [key](const value_type &value) { return supercedes(key, value.first); });
    }

    iterator begin() { return m_values.begin(); }
    const_iterator begin() const { return m_values.begin(); }
    const_iterator cbegin() const { return m_values.cbegin(); }

    iterator end() { return m_values.end(); }
    const_iterator end() const { return m_values.end(); }
    const_iterator cend() const { return m_values.cend(); }

    mapped_type &operator[](const key_type &key) { return find(key)->second; }

    static bool edge(const key_type &u, const key_type &v)
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

} // namespace dynd::detail

template <typename MappedType, size_t...>
class dispatch_map;

template <typename MappedType>
class dispatch_map<MappedType, 1> : public detail::dispatch_map<type_id_t, MappedType> {
  using detail::dispatch_map<type_id_t, MappedType>::dispatch_map;
};

template <typename MappedType, size_t N>
class dispatch_map<MappedType, N> : public detail::dispatch_map<std::array<type_id_t, N>, MappedType> {
  using detail::dispatch_map<std::array<type_id_t, N>, MappedType>::dispatch_map;
};

} // namespace dynd
