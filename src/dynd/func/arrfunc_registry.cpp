//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <math.h>

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/arrfunc_registry.hpp>
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/multidispatch_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>

using namespace std;
using namespace dynd;

// Probably want to use a concurrent_hash_map, like
// http://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/concurrent_hash_map_cls.htm
static map<nd::string, nd::arrfunc> *registry;

template<typename T0, typename T1>
static nd::arrfunc make_ufunc(T0 f0, T1 f1)
{
  nd::arrfunc af[2] = {nd::make_apply_arrfunc(f0),
                       nd::make_apply_arrfunc(f1)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

template<typename T0, typename T1, typename T2>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2)
{
  nd::arrfunc af[3] = {nd::make_apply_arrfunc(f0),
                       nd::make_apply_arrfunc(f1),
                       nd::make_apply_arrfunc(f2)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4)
{
  nd::arrfunc af[5] = {
      nd::make_apply_arrfunc(f0), nd::make_apply_arrfunc(f1),
      nd::make_apply_arrfunc(f2), nd::make_apply_arrfunc(f3),
      nd::make_apply_arrfunc(f4)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6)
{
  nd::arrfunc af[7] = {
      nd::make_apply_arrfunc(f0), nd::make_apply_arrfunc(f1),
      nd::make_apply_arrfunc(f2), nd::make_apply_arrfunc(f3),
      nd::make_apply_arrfunc(f4), nd::make_apply_arrfunc(f5),
      nd::make_apply_arrfunc(f6)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6,
                              T7 f7)
{
  nd::arrfunc af[8] = {
      nd::make_apply_arrfunc(f0), nd::make_apply_arrfunc(f1),
      nd::make_apply_arrfunc(f2), nd::make_apply_arrfunc(f3),
      nd::make_apply_arrfunc(f4), nd::make_apply_arrfunc(f5),
      nd::make_apply_arrfunc(f6), nd::make_apply_arrfunc(f7)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5, typename T6, typename T7, typename T8, typename T9>
static nd::arrfunc make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6,
                              T7 f7, T8 f8, T9 f9)
{
  nd::arrfunc af[10] = {
      nd::make_apply_arrfunc(f0), nd::make_apply_arrfunc(f1),
      nd::make_apply_arrfunc(f2), nd::make_apply_arrfunc(f3),
      nd::make_apply_arrfunc(f4), nd::make_apply_arrfunc(f5),
      nd::make_apply_arrfunc(f6), nd::make_apply_arrfunc(f7),
      nd::make_apply_arrfunc(f8), nd::make_apply_arrfunc(f9)};
  return lift_arrfunc(
      make_multidispatch_arrfunc(sizeof(af) / sizeof(af[0]), af));
}

namespace {
template <typename T>
struct add {
  inline T operator()(T x, T y) const { return x + y; }
};
template <typename T>
struct subtract {
  inline T operator()(T x, T y) const { return x - y; }
};
template <typename T>
struct multiply {
  inline T operator()(T x, T y) const { return x * y; }
};
template <typename T>
struct divide {
  inline T operator()(T x, T y) const { return x / y; }
};
template <typename T>
struct negative {
  inline T operator()(T x) const { return -x; }
};
template <typename T>
struct sign {
  // Note that by returning `x` in the else case, NaN will pass
  // through
  inline T operator()(T x) const { return x > T(0) ? 1 : (x < T(0) ? -1 : x); }
};
template <>
struct sign<dynd_int128> {
  inline dynd_int128 operator()(const dynd_int128 &x) const
  {
    return x.is_negative() ? -1 : (x == 0 ? 0 : 1);
  }
};
template <>
struct sign<dynd_uint128> {
  inline dynd_uint128 operator()(const dynd_uint128 &x) const
  {
    return x == 0 ? 0 : 1;
  }
};
template <typename T>
struct conj_fn {
  inline T operator()(T x) const { return conj(x); };
};
#if !(defined(_MSC_VER) && _MSC_VER < 1700)
template <typename T>
struct logaddexp {
  inline T operator()(T x, T y) const
  {
    // log(exp(x) + exp(y))
    if (x > y) {
      return x + log1p(exp(y - x));
    } else if (x <= y) {
      return y + log1p(exp(x - y));
    } else {
      // a NaN, +inf/+inf, or -inf/-inf
      return x + y;
    }
  }
};
template <typename T>
struct logaddexp2 {
  inline T operator()(T x, T y) const
  {
    const T log2_e = T(1.442695040888963407359924681001892137);
    // log2(exp2(x) + exp2(y))
    if (x > y) {
      return x + log2_e * log1p(exp2(y - x));
    } else if (x <= y) {
      return y + log2_e * log1p(exp2(x - y));
    } else {
      // a NaN, +inf/+inf, or -inf/-inf
      return x + y;
    }
  }
};
#endif
} // anonymous namespace

void init::arrfunc_registry_init()
{
  registry = new map<nd::string, nd::arrfunc>;

  // Arithmetic
  func::set_regfunction(
      "add", make_ufunc(add<int32_t>(), add<int64_t>(), add<dynd_int128>(),
                        add<uint32_t>(), add<uint64_t>(), add<dynd_uint128>(),
                        add<float>(), add<double>(), add<complex<float> >(),
                        add<complex<double> >()));
  func::set_regfunction(
      "subtract",
      make_ufunc(subtract<int32_t>(), subtract<int64_t>(),
                 subtract<dynd_int128>(), subtract<float>(), subtract<double>(),
                 subtract<complex<float> >(), subtract<complex<double> >()));
  func::set_regfunction(
      "multiply",
      make_ufunc(multiply<int32_t>(), multiply<int64_t>(),
                 /*multiply<dynd_int128>(),*/ multiply<uint32_t>(),
                 multiply<uint64_t>(), /*multiply<dynd_uint128>(),*/
                 multiply<float>(), multiply<double>(),
                 multiply<complex<float> >(), multiply<complex<double> >()));
  func::set_regfunction(
      "divide",
      make_ufunc(
          divide<int32_t>(), divide<int64_t>(),   /*divide<dynd_int128>(),*/
          divide<uint32_t>(), divide<uint64_t>(), /*divide<dynd_uint128>(),*/
          divide<float>(), divide<double>(), divide<complex<float> >(),
          divide<complex<double> >()));
  func::set_regfunction(
      "negative",
      make_ufunc(negative<int32_t>(), negative<int64_t>(),
                 negative<dynd_int128>(), negative<float>(), negative<double>(),
                 negative<complex<float> >(), negative<complex<double> >()));
  func::set_regfunction("sign", make_ufunc(sign<int32_t>(), sign<int64_t>(),
                                           sign<dynd_int128>(), sign<float>(),
                                           sign<double>()));
  func::set_regfunction("conj", make_ufunc(conj_fn<complex<float> >(), conj_fn<complex<double> >()));

#if !(defined(_MSC_VER) && _MSC_VER < 1700)
  func::set_regfunction("logaddexp",
                        make_ufunc(logaddexp<float>(), logaddexp<double>()));
  func::set_regfunction("logaddexp2",
                        make_ufunc(logaddexp2<float>(), logaddexp2<double>()));
#endif

  // Trig functions
  func::set_regfunction(
      "sin", make_ufunc(&::sinf, static_cast<double (*)(double)>(&::sin)));
  func::set_regfunction(
      "cos", make_ufunc(&::cosf, static_cast<double (*)(double)>(&::cos)));
  func::set_regfunction(
      "tan", make_ufunc(&::tanf, static_cast<double (*)(double)>(&::tan)));
  func::set_regfunction(
      "exp", make_ufunc(&::expf, static_cast<double (*)(double)>(&::exp)));
  func::set_regfunction(
      "arcsin", make_ufunc(&::asinf, static_cast<double (*)(double)>(&::asin)));
  func::set_regfunction(
      "arccos", make_ufunc(&::acosf, static_cast<double (*)(double)>(&::acos)));
  func::set_regfunction(
      "arctan", make_ufunc(&::atanf, static_cast<double (*)(double)>(&::atan)));
  func::set_regfunction(
      "arctan2",
      make_ufunc(&::atan2f, static_cast<double (*)(double, double)>(&::atan2)));
  func::set_regfunction(
      "hypot",
      make_ufunc(&::hypotf, static_cast<double (*)(double, double)>(&::hypot)));
  func::set_regfunction(
      "sinh", make_ufunc(&::sinhf, static_cast<double (*)(double)>(&::sinh)));
  func::set_regfunction(
      "cosh", make_ufunc(&::coshf, static_cast<double (*)(double)>(&::cosh)));
  func::set_regfunction(
      "tanh", make_ufunc(&::tanhf, static_cast<double (*)(double)>(&::tanh)));
#if !(defined(_MSC_VER) && _MSC_VER < 1700)
  func::set_regfunction(
      "asinh",
      make_ufunc(&::asinhf, static_cast<double (*)(double)>(&::asinh)));
  func::set_regfunction(
      "acosh",
      make_ufunc(&::acoshf, static_cast<double (*)(double)>(&::acosh)));
  func::set_regfunction(
      "atanh",
      make_ufunc(&::atanhf, static_cast<double (*)(double)>(&::atanh)));
#endif

  func::set_regfunction(
      "power",
      make_ufunc(&powf, static_cast<double (*)(double, double)>(&::pow)));
}

void init::arrfunc_registry_cleanup()
{
  delete registry;
  registry = NULL;
}

const std::map<nd::string, nd::arrfunc> &func::get_regfunctions()
{
  return *registry;
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
