//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <cmath>

#include <dynd/callable_registry.hpp>
#include <dynd/functional.hpp>
#include <dynd/io.hpp>
#include <dynd/option.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/func/take.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/func/min.hpp>
#include <dynd/func/max.hpp>
#include <dynd/func/complex.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename... T>
nd::callable make_ufunc(T... f)
{
  return nd::functional::elwise(nd::functional::old_multidispatch({nd::functional::apply(f)...}));
}

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
struct sign<int128> {
  inline int128 operator()(const int128 &x) const { return x.is_negative() ? -1 : (x == 0 ? 0 : 1); }
};
template <>
struct sign<uint128> {
  inline uint128 operator()(const uint128 &x) const { return x == 0 ? 0 : 1; }
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
    }
    else if (x <= y) {
      return y + log1p(exp(x - y));
    }
    else {
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
    }
    else if (x <= y) {
      return y + log2_e * log1p(exp2(x - y));
    }
    else {
      // a NaN, +inf/+inf, or -inf/-inf
      return x + y;
    }
  }
};
#endif
} // anonymous namespace

std::map<std::string, nd::callable> &nd::callable_registry::get_regfunctions()
{
  static map<std::string, nd::callable> registry;
  if (registry.empty()) {
    registry["add"] = nd::add::get();
    registry["subtract"] = nd::subtract::get();
    registry["multiply"] = nd::multiply::get();
    registry["divide"] =
        make_ufunc(::divide<int32_t>(), ::divide<int64_t>(), ::divide<uint32_t>(), ::divide<uint64_t>(),
                   ::divide<float>(), ::divide<double>(), ::divide<complex<float>>(), ::divide<complex<double>>());
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

    registry["take"] = nd::take::get();
    registry["sum"] = nd::sum::get();
    registry["min"] = nd::min::get();
    registry["max"] = nd::max::get();
    registry["real"] = real::get();
    registry["imag"] = imag::get();
    registry["conj"] = conj::get();

    // assign.cpp
    registry["assign"] = assign::get();

    // io.cpp
    registry["serialize"] = serialize::get();

    // option.cpp
    registry["assign_na"] = assign_na::get();
    registry["is_na"] = is_na::get();

    // random.cpp
    registry["uniform"] = random::uniform::get();
  }

  return registry;
}

nd::callable &nd::callable_registry::operator[](const std::string &name)
{
  std::map<std::string, nd::callable> &registry = get_regfunctions();

  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second;
  }
  else {
    stringstream ss;
    ss << "No dynd function ";
    print_escaped_utf8_string(ss, name);
    ss << " has been registered";
    throw invalid_argument(ss.str());
  }
}

DYND_API class nd::callable_registry nd::callable_registry;
