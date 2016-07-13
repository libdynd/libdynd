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

  typedef void (*observer)(registry_entry *, const char *);

  typedef typename namespace_type::iterator iterator;
  typedef typename namespace_type::const_iterator const_iterator;

private:
  registry_entry *m_parent;
  std::string m_name;
  bool m_is_namespace;
  value_type m_value;
  namespace_type m_namespace;
  std::vector<observer> m_observers;

public:
  registry_entry *parent() { return m_parent; }

  registry_entry() = default;

  registry_entry(const value_type &entry) : m_parent(nullptr), m_is_namespace(false), m_value(entry) {}

  registry_entry(std::initializer_list<typename namespace_type::value_type> values)
      : m_parent(nullptr), m_is_namespace(true), m_namespace(values) {
    for (auto &x : m_namespace) {
      x.second.m_parent = this;
      x.second.m_name = x.first;
    }
  }

  registry_entry(const registry_entry &other)
      : m_parent(other.m_parent), m_name(other.m_name), m_is_namespace(other.m_is_namespace), m_value(other.m_value),
        m_namespace(other.m_namespace), m_observers(other.m_observers) {
    for (auto &x : m_namespace) {
      x.second.m_parent = this;
      x.second.m_name = x.first;
    }
  }

  registry_entry(registry_entry &&other) = delete;

  registry_entry &operator=(const registry_entry &other) {
    m_parent = other.m_parent;
    m_is_namespace = other.m_is_namespace;
    m_value = other.m_value;
    m_namespace = other.m_namespace;
    m_observers = other.m_observers;
    m_name = other.m_name;

    for (auto x : m_namespace) {
      x.second.m_parent = this;
    }

    return *this;
  }

  registry_entry &operator=(registry_entry &&other) = delete;

  value_type &value() { return m_value; }
  const value_type &value() const { return m_value; }

  bool is_namespace() const { return m_is_namespace; }

  const std::string &name() const { return m_name; }

  void emit(const char *path) {
    for (observer observer : m_observers) {
      observer(this, path);
    }

  //  if (m_parent) {
    //  std::string new_path = m_name + "." + std::string(path);
//      m_parent->emit(new_path.c_str());
//    }
  }

  void insert(const typename namespace_type::value_type &child) {
    auto subentry = m_namespace.find(child.first);
    if (subentry == m_namespace.end()) {
      auto res = m_namespace.emplace(child);
      res.first->second.m_parent = this;
      res.first->second.m_name = child.first;
    } else {
      for (const auto &pair : child.second) {
        subentry->second.insert(pair);
      }
    }

    emit(child.first.c_str());
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
