//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/types/callable_type.hpp>


using namespace std;
using namespace dynd;

TEST(CallableType, Repr)
{
  std::vector<const char *> roundtrip {
    // positional-only
    "() -> int32",
    "(int32) -> int32",
    "(int32, float64) -> int32",
    // keyword-only
    "(scale: float64) -> int32",
    "(scale: float64, color: float64) -> int32",
    // positional+keyword
    "(int32, scale: float64) -> int32",
    "(int32, scale: float64, color: float64) -> int32",
    "(int32, float32, scale: float64, color: float64) -> int32",
    // positional-variadic
    "(...) -> int32",
    "(int32, ...) -> int32",
    "(int32, float32, ...) -> int32",
    // keyword-variadic
    "(scale: float64, ...) -> int32",
    "(scale: float64, color: float64, ...) -> int32",
    // positional-variadic+keyword
    "(..., scale: float64) -> int32",
    "(int32, ..., scale: float64) -> int32",
    "(int32, float32, ..., scale: float64) -> int32",
    // positional+keyword-variadic
    "(int32, scale: float64, ...) -> int32",
    "(int32, float32, scale: float64, color: float64, ...) -> int32",
    // positional-variadic+keyword-variadic
    "(..., scale: float64, ...) -> int32",
    "(int32, ..., scale: float64, color: float64, ...) -> int32",
    "(int32, float32, ..., scale: float64, color: float64, ...) -> int32",
  };

  for (auto s : roundtrip) {
    EXPECT_TYPE_REPR_EQ(s, ndt::type(s));
  }
}



