//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <cmath>

#include <dynd/func/apply.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/callable_registry.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/take.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/func/min.hpp>
#include <dynd/func/max.hpp>
#include <dynd/func/option.hpp>

using namespace std;
using namespace dynd;

template <typename T0, typename T1>
static nd::callable make_ufunc(T0 f0, T1 f1)
{
  return nd::functional::elwise(
      nd::functional::old_multidispatch({nd::functional::apply(f0), nd::functional::apply(f1)}));
}

template <typename T0, typename T1, typename T2>
static nd::callable make_ufunc(T0 f0, T1 f1, T2 f2)
{
  return nd::functional::elwise(nd::functional::old_multidispatch(
      {nd::functional::apply(f0), nd::functional::apply(f1), nd::functional::apply(f2)}));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
static nd::callable make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4)
{
  return nd::functional::elwise(nd::functional::old_multidispatch({nd::functional::apply(f0), nd::functional::apply(f1),
                                                                   nd::functional::apply(f2), nd::functional::apply(f3),
                                                                   nd::functional::apply(f4)}));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
static nd::callable make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6)
{
  return nd::functional::elwise(nd::functional::old_multidispatch(
      {nd::functional::apply(f0), nd::functional::apply(f1), nd::functional::apply(f2), nd::functional::apply(f3),
       nd::functional::apply(f4), nd::functional::apply(f5), nd::functional::apply(f6)}));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
static nd::callable make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6, T7 f7)
{
  return nd::functional::elwise(nd::functional::old_multidispatch(
      {nd::functional::apply(f0), nd::functional::apply(f1), nd::functional::apply(f2), nd::functional::apply(f3),
       nd::functional::apply(f4), nd::functional::apply(f5), nd::functional::apply(f6), nd::functional::apply(f7)}));
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
static nd::callable make_ufunc(T0 f0, T1 f1, T2 f2, T3 f3, T4 f4, T5 f5, T6 f6, T7 f7, T8 f8, T9 f9)
{
  return nd::functional::elwise(nd::functional::old_multidispatch(
      {nd::functional::apply(f0), nd::functional::apply(f1), nd::functional::apply(f2), nd::functional::apply(f3),
       nd::functional::apply(f4), nd::functional::apply(f5), nd::functional::apply(f6), nd::functional::apply(f7),
       nd::functional::apply(f8), nd::functional::apply(f9)}));
}

namespace {
template <typename T>
struct add {
  inline T operator()(T x, T y) const
  {
    return x + y;
  }
};
template <typename T>
struct subtract {
  inline T operator()(T x, T y) const
  {
    return x - y;
  }
};
template <typename T>
struct multiply {
  inline T operator()(T x, T y) const
  {
    return x * y;
  }
};
template <typename T>
struct divide {
  inline T operator()(T x, T y) const
  {
    return x / y;
  }
};
template <typename T>
struct negative {
  inline T operator()(T x) const
  {
    return -x;
  }
};
template <typename T>
struct sign {
  // Note that by returning `x` in the else case, NaN will pass
  // through
  inline T operator()(T x) const
  {
    return x > T(0) ? 1 : (x < T(0) ? -1 : x);
  }
};
template <>
struct sign<int128> {
  inline int128 operator()(const int128 &x) const
  {
    return x.is_negative() ? -1 : (x == 0 ? 0 : 1);
  }
};
template <>
struct sign<uint128> {
  inline uint128 operator()(const uint128 &x) const
  {
    return x == 0 ? 0 : 1;
  }
};
template <typename T>
struct conj_fn {
  inline T operator()(T x) const
  {
    return conj(x);
  };
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

std::map<std::string, nd::callable> &func::get_regfunctions()
{
  // Probably want to use a concurrent_hash_map, like
  // http://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/concurrent_hash_map_cls.htm
  static map<std::string, nd::callable> registry;
  if (registry.empty()) {
    // Arithmetic
    /*
      func::set_regfunction(
          "add", make_ufunc(add<int32_t>(), add<int64_t>(), add<int128>(),
                            add<uint32_t>(), add<uint64_t>(),
      add<dynd_uint128>(),
                            add<float>(), add<double>(), add<complex<float>>(),
                            add<complex<double>>()));
    */
    registry["add"] = nd::add;
    registry["subtract"] = make_ufunc(subtract<int32_t>(), subtract<int64_t>(), subtract<int128>(), subtract<float>(),
                                      subtract<double>(), subtract<complex<float>>(), subtract<complex<double>>());
    registry["multiply"] =
        make_ufunc(multiply<int32_t>(), multiply<int64_t>(),
                   /*multiply<int128>(),*/ multiply<uint32_t>(), multiply<uint64_t>(), /*multiply<dynd_uint128>(),*/
                   multiply<float>(), multiply<double>(), multiply<complex<float>>(), multiply<complex<double>>());
    registry["divide"] =
        make_ufunc(divide<int32_t>(), divide<int64_t>(),   /*divide<int128>(),*/
                   divide<uint32_t>(), divide<uint64_t>(), /*divide<dynd_uint128>(),*/
                   divide<float>(), divide<double>(), divide<complex<float>>(), divide<complex<double>>());
    registry["negative"] = make_ufunc(negative<int32_t>(), negative<int64_t>(), negative<int128>(), negative<float>(),
                                      negative<double>(), negative<complex<float>>(), negative<complex<double>>());
    registry["sign"] = make_ufunc(sign<int32_t>(), sign<int64_t>(), sign<int128>(), sign<float>(), sign<double>());
    registry["conj"] = make_ufunc(conj_fn<std::complex<float>>(), conj_fn<std::complex<double>>());

#if !(defined(_MSC_VER) && _MSC_VER < 1700)
    registry["logaddexp"] = make_ufunc(logaddexp<float>(), logaddexp<double>());
    registry["logaddexp2"] = make_ufunc(logaddexp2<float>(), logaddexp2<double>());
#endif

    // Trig functions
    registry["sin"] = make_ufunc(&::sinf, static_cast<double (*)(double)>(&::sin));
    registry["cos"] = make_ufunc(&::cosf, static_cast<double (*)(double)>(&::cos));
    registry["tan"] = make_ufunc(&::tanf, static_cast<double (*)(double)>(&::tan));
    registry["exp"] = make_ufunc(&::expf, static_cast<double (*)(double)>(&::exp));
    registry["arcsin"] = make_ufunc(&::asinf, static_cast<double (*)(double)>(&::asin));
    registry["arccos"] = make_ufunc(&::acosf, static_cast<double (*)(double)>(&::acos));
    registry["arctan"] = make_ufunc(&::atanf, static_cast<double (*)(double)>(&::atan));
    registry["arctan2"] = make_ufunc(&::atan2f, static_cast<double (*)(double, double)>(&::atan2));
    registry["hypot"] = make_ufunc(&::hypotf, static_cast<double (*)(double, double)>(&::hypot));
    registry["sinh"] = make_ufunc(&::sinhf, static_cast<double (*)(double)>(&::sinh));
    registry["cosh"] = make_ufunc(&::coshf, static_cast<double (*)(double)>(&::cosh));
    registry["tanh"] = make_ufunc(&::tanhf, static_cast<double (*)(double)>(&::tanh));
#if !(defined(_MSC_VER) && _MSC_VER < 1700)
    registry["asinh"] = make_ufunc(&::asinhf, static_cast<double (*)(double)>(&::asinh));
    registry["acosh"] = make_ufunc(&::acoshf, static_cast<double (*)(double)>(&::acosh));
    registry["atanh"] = make_ufunc(&::atanhf, static_cast<double (*)(double)>(&::atanh));
#endif

    registry["power"] = make_ufunc(&powf, static_cast<double (*)(double, double)>(&::pow));

    registry["uniform"] = nd::random::uniform;
    registry["take"] = nd::take;
    registry["sum"] = nd::sum;
    registry["is_avail"] = nd::is_avail;
    registry["min"] = nd::min;
    registry["max"] = nd::max;
  }

  return registry;
}

nd::callable func::get_regfunction(const std::string &name)
{
  const std::map<std::string, nd::callable> &registry = get_regfunctions();

  map<std::string, nd::callable>::const_iterator it = registry.find(name);
  if (it != registry.end()) {
    return it->second;
  } else {
    stringstream ss;
    ss << "No dynd function ";
    print_escaped_utf8_string(ss, name);
    ss << " has been registered";
    throw invalid_argument(ss.str());
  }
}

void func::set_regfunction(const std::string &name, const nd::callable &af)
{
  std::map<std::string, nd::callable> &registry = get_regfunctions();

  registry[name] = af;
}
