//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {

class registry_entry {
public:
  typedef nd::callable value_type;
  typedef std::map<std::string, registry_entry> namespace_type;

  typedef void (*observer)(registry_entry *, const char *, registry_entry *);

  typedef typename namespace_type::iterator iterator;
  typedef typename namespace_type::const_iterator const_iterator;

private:
  std::string m_name;
  bool m_is_namespace;
  value_type m_value;
  namespace_type m_namespace;
  std::vector<observer> m_observers;

public:
  registry_entry() = default;

  registry_entry(const value_type &entry) : m_is_namespace(false), m_value(entry) {}

  registry_entry(std::initializer_list<std::pair<const std::string, registry_entry>> values)
      : m_is_namespace(true), m_namespace(values) {
    for (auto &pair : m_namespace) {
      pair.second.absolute(pair.first);
    }
  }

  value_type &value() { return m_value; }
  const value_type &value() const { return m_value; }

  const std::string &path() const { return m_name; }

  void absolute(const std::string &name) {
    if (m_name.empty()) {
      m_name = name;
    } else {
      m_name = name + "." + m_name;
    }

    if (m_is_namespace) {
      for (auto &pair : m_namespace) {
        pair.second.absolute(name);
      }
    }
  }

  bool is_namespace() const { return m_is_namespace; }

  void emit(const char *name, registry_entry *entry) {
    for (observer observer : m_observers) {
      observer(this, name, entry);
    }
  }

  void insert(const std::pair<std::string, const registry_entry &> &entry) {
    iterator it = m_namespace.find(entry.first);
    if (it != m_namespace.end()) {
      throw std::runtime_error("entry already exists");
    }

    it = m_namespace.emplace(entry).first;
    emit(entry.first.c_str(), &it->second);
  }

  iterator find(const std::string &name) { return m_namespace.find(name); }

  void observe(observer obs) { m_observers.emplace_back(obs); }

  registry_entry &operator[](const std::string &path) {
    size_t i = path.find(".");
    std::string name = path.substr(0, i);

    iterator it = find(name);
    if (it == end()) {
      std::stringstream ss;
      ss << "No dynd function ";
      print_escaped_utf8_string(ss, name);
      ss << " has been registered";
      throw std::invalid_argument(ss.str());
    }

    if (i == std::string::npos) {
      return it->second;
    }

    return it->second[path.substr(i + 1)];
  }

  iterator begin() { return m_namespace.begin(); }
  const_iterator begin() const { return m_namespace.begin(); }

  iterator end() { return m_namespace.end(); }
  const_iterator end() const { return m_namespace.end(); }

  const_iterator cbegin() const { return m_namespace.cbegin(); }

  const_iterator cend() const { return m_namespace.cend(); }
};

/**
 * Returns a reference to the map of registered callables.
 */
DYND_API registry_entry &registered();

inline registry_entry &registered(const std::string &path) {
  registry_entry &entry = registered();
  return entry[path];
}

namespace nd {

  template <typename... ArgTypes>
  array array::f(const char *name, ArgTypes &&... args) const {
    registry_entry &entry = registered("dynd.nd");

    callable &f = entry[name].value();
    return f(*this, std::forward<ArgTypes>(args)...);
  }

} // namespace dynd::nd
} // namespace dynd
