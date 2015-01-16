//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace func {

/**
 * Returns a reference to the map of registered arrfuncs.
 * NOTE: The internal representation will change, this
 *       function will change.
 */
const std::map<nd::string, nd::arrfunc>& get_regfunctions();

/**
  * Looks up a named arrfunc from the registry.
  */
nd::arrfunc get_regfunction(const nd::string &name);
/**
  * Sets a named arrfunc in the registry.
  */
void set_regfunction(const nd::string &name, const nd::arrfunc &af);

} // namespace func

namespace init {
void arrfunc_registry_init();
void arrfunc_registry_cleanup();
} // namespace init

} // namespace dynd
