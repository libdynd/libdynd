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
    DYND_API std::map<std::string, nd::callable> &get_regfunctions();

  public:
    /**
      * Looks up a named callable from the registry.
      */
    DYND_API callable &operator[](const std::string &name);

    std::map<std::string, callable>::iterator begin() { return get_regfunctions().begin(); }
    std::map<std::string, callable>::const_iterator cbegin() { return get_regfunctions().cbegin(); }

    std::map<std::string, callable>::iterator end() { return get_regfunctions().end(); }
    std::map<std::string, callable>::const_iterator cend() { return get_regfunctions().cend(); }
  } callable_registry;

} // namespace dynd::nd
} // namespace dynd
