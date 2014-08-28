//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/arrfunc_registry.hpp>
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>

#include <map>

using namespace std;
using namespace dynd;

// Probably want to use a concurrent_hash_map, like
// http://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/concurrent_hash_map_cls.htm
static map<nd::string, nd::arrfunc> *registry;

void init::arrfunc_registry_init()
{
  registry = new map<nd::string, nd::arrfunc>;

  func::set_regfunction("sin", lift_arrfunc(nd::make_functor_arrfunc(
                                   static_cast<double (*)(double)>(&::sin))));
  func::set_regfunction("cos", lift_arrfunc(nd::make_functor_arrfunc(
                                   static_cast<double (*)(double)>(&::cos))));
  func::set_regfunction("exp", lift_arrfunc(nd::make_functor_arrfunc(
                                   static_cast<double (*)(double)>(&::exp))));
}

void init::arrfunc_registry_cleanup()
{
  delete registry;
  registry = NULL;
}

nd::arrfunc func::get_regfunction(const nd::string &name)
{
  map<nd::string, nd::arrfunc>::const_iterator it = registry->find(name);
  if (it != registry->end()) {
    return it->second;
  } else {
    stringstream ss;
    ss << "No dynd function ";
    print_escaped_utf8_string(ss, name);
    ss << " has been registered";
    throw invalid_argument(ss.str());
  }
}

void func::set_regfunction(const nd::string &name, const nd::arrfunc &af)
{
  (*registry)[name] = af;
}
