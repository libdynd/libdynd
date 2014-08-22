//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/arrfunc_registry.hpp>

#include <map>

using namespace std;
using namespace dynd;

static map<nd::string, nd::arrfunc>& get_registry()
{
  static map<nd::string, nd::arrfunc> reg;
  return reg;
}

nd::arrfunc func::get_regfunction(const nd::string &name)
{
}

void func::set_regfunction(const nd::string &name, const nd::arrfunc &af)
{
}
