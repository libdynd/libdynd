//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace func {

  /**
   * Returns a reference to the map of registered callables.
   * NOTE: The internal representation will change, this
   *       function will change.
   */
  DYND_API std::map<std::string, nd::callable> &get_regfunctions();

  /**
    * Looks up a named callable from the registry.
    */
  DYND_API nd::callable get_regfunction(const std::string &name);
  /**
    * Sets a named callable in the registry.
    */
  DYND_API void set_regfunction(const std::string &name, const nd::callable &af);

} // namespace func
} // namespace dynd
