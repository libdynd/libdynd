//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "../dynd_assertions.hpp"
#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/types/callable_type.hpp>

using namespace std;
using namespace dynd;

TEST(CallableType, Basic) {
  const ndt::type &callable_tp = ndt::make_type<int(double, float)>();

  vector<ndt::type> bases{ndt::make_type<ndt::scalar_kind_type>(), ndt::make_type<ndt::any_kind_type>()};
  EXPECT_EQ(bases, callable_tp.bases());
}

TEST(CallableType, Repr) {
  std::vector<const char *> roundtrip{
      // positional-only
      "() -> int32", "(int32) -> int32", "(int32, float64) -> int32",
      // keyword-only
      "(scale: float64) -> int32", "(scale: float64, color: float64) -> int32",
      // positional+keyword
      "(int32, scale: float64) -> int32", "(int32, scale: float64, color: float64) -> int32",
      "(int32, float32, scale: float64, color: float64) -> int32",
      // positional-variadic
      "(...) -> int32", "(int32, ...) -> int32", "(int32, float32, ...) -> int32",
      // keyword-variadic
      "(scale: float64, ...) -> int32", "(scale: float64, color: float64, ...) -> int32",
      // positional-variadic+keyword
      "(..., scale: float64) -> int32", "(int32, ..., scale: float64) -> int32",
      "(int32, float32, ..., scale: float64) -> int32",
      // positional+keyword-variadic
      "(int32, scale: float64, ...) -> int32", "(int32, float32, scale: float64, color: float64, ...) -> int32",
      // positional-variadic+keyword-variadic
      "(..., scale: float64, ...) -> int32", "(int32, ..., scale: float64, color: float64, ...) -> int32",
      "(int32, float32, ..., scale: float64, color: float64, ...) -> int32",
  };

  for (auto s : roundtrip) {
    EXPECT_TYPE_REPR_EQ(s, ndt::type(s));
  }
}

TEST(CallableType, IDOf) { EXPECT_EQ(callable_id, ndt::id_of<ndt::callable_type>::value); }
