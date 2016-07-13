//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {

class registry_entry {
public:
  typedef void (*observer)(const char *, registry_entry *);
  typedef typename std::map<std::string, registry_entry>::iterator iterator;
  typedef typename std::map<std::string, registry_entry>::const_iterator const_iterator;

private:
  bool m_is_namespace;
  nd::callable m_value;
  std::map<std::string, registry_entry> m_namespace;
  std::vector<observer> m_observers;

public:
  registry_entry() = default;

  registry_entry(const nd::callable &entry) : m_is_namespace(false), m_value(entry) {}

  registry_entry(std::initializer_list<std::pair<const std::string, registry_entry>> values)
      : m_is_namespace(true), m_namespace(values) {}

  nd::callable &value() { return m_value; }
  const nd::callable &value() const { return m_value; }

  bool is_namespace() const { return m_is_namespace; }

  void insert(const std::pair<const std::string, registry_entry> &entry) {
    auto subentry = m_namespace.find(entry.first);
    if (subentry == m_namespace.end()) {
      m_namespace.emplace(entry);
    } else {
      for (const auto &pair : entry.second) {
        subentry->second.insert(pair);
      }
    }

    for (observer obs : m_observers) {
      obs(entry.first.c_str(), this);
    }
  }

  iterator find(const std::string &name) { return m_namespace.find(name); }

  void observe(observer obs) { m_observers.emplace_back(obs); }

  registry_entry &operator=(const registry_entry &rhs) {
    m_is_namespace = rhs.m_is_namespace;
    if (m_is_namespace) {
      m_namespace = rhs.m_namespace;
    } else {
      m_value = rhs.m_value;
    }

    return *this;
  }

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
