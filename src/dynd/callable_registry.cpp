//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <map>
#include <cmath>

#include <dynd/callable_registry.hpp>
#include <dynd/functional.hpp>
#include <dynd/func/arithmetic.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/io.hpp>
#include <dynd/math.hpp>
#include <dynd/option.hpp>
#include <dynd/func/random.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/func/take.hpp>
#include <dynd/func/min.hpp>
#include <dynd/func/max.hpp>
#include <dynd/func/complex.hpp>
#include <dynd/func/pointer.hpp>

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
    registry["negative"] = make_ufunc(negative<int32_t>(), negative<int64_t>(), negative<int128>(), negative<float>(),
                                      negative<double>(), negative<complex<float>>(), negative<complex<double>>());
    registry["sign"] = make_ufunc(sign<int32_t>(), sign<int64_t>(), sign<int128>(), sign<float>(), sign<double>());

    registry["take"] = take::get();
    registry["sum"] = sum::get();
    registry["min"] = min::get();
    registry["max"] = max::get();

    // dynd/arithmetic.hpp
    registry["add"] = add::get();
    registry["subtract"] = subtract::get();
    registry["multiply"] = multiply::get();
    registry["divide"] = divide::get();

    // dynd/assign.hpp
    registry["assign"] = assign::get();

    // dynd/complex.hpp
    registry["real"] = real::get();
    registry["imag"] = imag::get();
    registry["conj"] = conj::get();

    // dynd/io.hpp
    registry["serialize"] = serialize::get();

    // dynd/math.hpp
    registry["sin"] = sin::get();
    registry["cos"] = cos::get();
    registry["tan"] = tan::get();
    registry["exp"] = exp::get();

    // dynd/option.hpp
    registry["assign_na"] = assign_na::get();
    registry["is_na"] = is_na::get();

    // dynd/pointer.hpp
    registry["dereference"] = dereference::get();

    // dynd/random.hpp
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
