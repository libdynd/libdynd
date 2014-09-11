//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/arrfunc_registry.hpp>
#include <dynd/func/functor_arrfunc.hpp>
#include <dynd/func/multidispatch_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>

#include <map>

using namespace std;
using namespace dynd;

// Probably want to use a concurrent_hash_map, like
// http://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/concurrent_hash_map_cls.htm
static map<nd::string, nd::arrfunc> *registry;

template<typename T0, typename T1>
static nd::arrfunc make_ufunc(T0 f0, T1 f1)
{
  vector<nd::arrfunc> af;
  af.push_back(nd::make_functor_arrfunc(f0));
  af.push_back(nd::make_functor_arrfunc(f1));
  return lift_arrfunc(make_multidispatch_arrfunc((intptr_t)af.size(), &af[0]));
}

template<typename T0, typename T1, typename T2>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2)
{
  vector<nd::arrfunc> af;
  af.push_back(nd::make_functor_arrfunc(f0));
  af.push_back(nd::make_functor_arrfunc(f1));
  af.push_back(nd::make_functor_arrfunc(f2));
  return lift_arrfunc(make_multidispatch_arrfunc((intptr_t)af.size(), &af[0]));
}

void init::arrfunc_registry_init()
{
  registry = new map<nd::string, nd::arrfunc>;

  func::set_regfunction("sin",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::sin),
#else
                                       &::sinf),
#endif
                                   static_cast<double (*)(double)>(&::sin)));
  func::set_regfunction("cos",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::cos),
#else
                                       &::cosf),
#endif
                                   static_cast<double (*)(double)>(&::cos)));
  func::set_regfunction("tan",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::tan),
#else
                                       &::tanf),
#endif
                                   static_cast<double (*)(double)>(&::tan)));
  func::set_regfunction("exp",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::exp),
#else
                                       &::expf),
#endif
                                   static_cast<double (*)(double)>(&::exp)));
  func::set_regfunction("arcsin",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::asin),
#else
                                       &::asinf),
#endif
                                   static_cast<double (*)(double)>(&::asin)));
  func::set_regfunction("arccos",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::acos),
#else
                                       &::acosf),
#endif
                                   static_cast<double (*)(double)>(&::acos)));
  func::set_regfunction("arctan",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::atan),
#else
                                       &::atanf),
#endif
                                   static_cast<double (*)(double)>(&::atan)));
  func::set_regfunction("arctan2",
                        make_ufunc(static_cast<float (*)(float, float)>(
#ifdef WIN32
                                       &::atan2),
#else
                                       &::atan2f),
#endif
                                   static_cast<double (*)(double, double)>(&::atan2)));
  func::set_regfunction("hypot",
                        make_ufunc(static_cast<float (*)(float, float)>(
#ifdef WIN32
                                       &::hypot),
#else
                                       &::hypotf),
#endif
                                   static_cast<double (*)(double, double)>(&::hypot)));
  func::set_regfunction("sinh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::sinh),
#else
                                       &::sinhf),
#endif
                                   static_cast<double (*)(double)>(&::sinh)));
  func::set_regfunction("cosh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::cosh),
#else
                                       &::coshf),
#endif
                                   static_cast<double (*)(double)>(&::cosh)));
  func::set_regfunction("tanh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::tanh),
#else
                                       &::tanhf),
#endif
                                   static_cast<double (*)(double)>(&::tanh)));
  func::set_regfunction("asinh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::asinh),
#else
                                       &::asinhf),
#endif
                                   static_cast<double (*)(double)>(&::asinh)));
  func::set_regfunction("acosh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::acosh),
#else
                                       &::acoshf),
#endif
                                   static_cast<double (*)(double)>(&::acosh)));
  func::set_regfunction("atanh",
                        make_ufunc(static_cast<float (*)(float)>(
#ifdef WIN32
                                       &::atanh),
#else
                                       &::atanhf),
#endif
                                   static_cast<double (*)(double)>(&::atanh)));


  func::set_regfunction(
      "power",
      make_ufunc(&powf, static_cast<double (*)(double, double)>(&::pow)));

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
