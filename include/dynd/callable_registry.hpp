//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API class callable_registry {
    /**
     * Returns a reference to the map of registered callables.
     * NOTE: The internal representation will change, this
     *       function will change.
     */
    DYND_API std::map<std::string, callable> &get_regfunctions();

    typedef callable mapped_type;
    typedef std::map<std::string, mapped_type> map_type;

  public:
    typedef std::string key_type;
    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;

    /**
      * Looks up a named callable from the registry.
      */
    DYND_API callable &operator[](const std::string &name);

    iterator find(const key_type &key) { return get_regfunctions().find(key); }

    iterator begin() { return get_regfunctions().begin(); }
    const_iterator cbegin() { return get_regfunctions().cbegin(); }

    iterator end() { return get_regfunctions().end(); }
    const_iterator cend() { return get_regfunctions().cend(); }
  } callable_registry;

} // namespace dynd::nd
} // namespace dynd
