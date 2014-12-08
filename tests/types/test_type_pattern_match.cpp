//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/pow_dimsym_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypePatternMatch, Simple)
{
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"), ndt::type("int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"), ndt::type("T")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("int32"), ndt::type("A... * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"), ndt::type("A... * T")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("4 * int32"), ndt::type("A... * 4 * M")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("3 * int32"), ndt::type("3 * A... * M")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed**3 * int32"),
                                 ndt::type("fixed**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed**2 * var * int32"),
                                 ndt::type("A**N * var * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed * int32"),
                                 ndt::type("fixed**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("4 * 4 * 3 * int32"),
                                 ndt::type("4**N * 3 * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("4 * 4 * 2 * int32"),
                                 ndt::type("4**N * M * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("4 * 4 * 2 * int32"),
                                 ndt::type("4**N * N * int32")));

  EXPECT_FALSE(ndt::pattern_match(ndt::type("int32"), ndt::type("int64")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("3 * int32"), ndt::type("T")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("int16"), ndt::type("A... * int32")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("4 * int32"), ndt::type("A... * 3 * M")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("fixed**3 * int32"),
                                  ndt::type("fixed**N * var * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("var * fixed**3 * int32"),
                                  ndt::type("fixed**N * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("fixed**3 * var * int32"),
                                  ndt::type("A**N * A * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("4 * 4 * 3 * int32"),
                                  ndt::type("4**N * N * int32")));

  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("... * int32"), ndt::type("... * T")));
}

TEST(TypePatternMatch, Struct)
{
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("{x: int32}"), ndt::type("{x: int32}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("{x: int32}"), ndt::type("{x: T}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("{x: int32}"),
                                 ndt::type("A... * {x: B... * T}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("100 * {x: int32, y: int32}"),
                                 ndt::type("A... * {x: T, y: T}")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("100 * {x: int32, y: int32, u: int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: S, v: S}")));

  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("100 * {x: int32, y: int32, u: int16, v: int32}"),
      ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, Option)
{
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"), ndt::type("?int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"), ndt::type("?T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"), ndt::type("T")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("100 * {x: ?int32, y: ?int32, u: int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: S, v: S}")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: ?S, v: S}")));

  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, ArrFuncProto)
{
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("() -> void"), ndt::type("() -> void")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("() -> void"), ndt::type("() -> T")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("() -> void"), ndt::type("() -> int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("() -> void"),
                                  ndt::type("(int32) -> void")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> void"),
                                  ndt::type("() -> void")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("() -> float32"), ndt::type("() -> T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("(float32) -> float32"),
                                 ndt::type("(T) -> T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                 ndt::type("(S) -> T")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("(int32, 5 * var * 2 * int16) -> float32"),
                         ndt::type("(S, A... * 2 * int16) -> T")));

  EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                  ndt::type("() -> T")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("() -> float32"), ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                  ndt::type("(T) -> T")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("(int32, 5 * var * 2 * int16) -> float32"),
                         ndt::type("(S, M * 2 * int16) -> T")));

}

TEST(TypePatternMatch, NestedArrFuncProto)
{
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("(3 * int32, (2 * int32) -> float64) -> 3 * float64"),
      ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("(3 * int32, (2 * float64) -> float64) -> 3 * float64"),
      ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("(3 * int32, (2 * int32) -> float64) -> 2 * float64"),
      ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("(3 * int32, (2 * int32) -> float64) -> 3 * float64"),
      ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("(3 * int32, func: (2 * int32) -> float64) -> 3 * float64"),
      ndt::type("(N * S, func: (M * S) -> T) -> N * T")));
  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("(3 * int32, func: (2 * int32) -> float64) -> 3 * float64"),
      ndt::type("(N * S, funk: (M * S) -> T) -> N * T")));
}

TEST(TypePatternMatch, Strided)
{
  // cfixed and fixed can match against strided
  EXPECT_TRUE(ndt::pattern_match(ndt::type("cfixed[3] * int32"),
                                 ndt::type("fixed * int32")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("3 * int32"), ndt::type("fixed * int32")));
  // cfixed can match against strided if the sizes match
  EXPECT_TRUE(ndt::pattern_match(ndt::type("cfixed[3] * int32"),
                                 ndt::type("3 * int32")));
  // Things do not hold in other cases
  EXPECT_FALSE(ndt::pattern_match(ndt::type("3 * int32"),
                                  ndt::type("cfixed[3] * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("cfixed[3] * int32"),
                                  ndt::type("4 * int32")));
}

TEST(TypePatternMatch, Pow)
{
  // Match pow_dimsym against itself
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed**N * int32"),
                                 ndt::type("fixed**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed**N * int32"),
                                 ndt::type("fixed**M * int32")));
  // Match fixed_dim against fixed_dimsym within the power's base
  EXPECT_TRUE(ndt::pattern_match(ndt::type("3**N * int32"),
                                 ndt::type("fixed**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("3**N * int32"),
                                 ndt::type("fixed**M * int32")));
  // Ensure that the typevar D is constrained in the power's base
  EXPECT_TRUE(ndt::pattern_match(ndt::type("fixed * fixed * fixed * int32"),
                                 ndt::type("D * D**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("3 * 3 * 3 * int32"),
                                 ndt::type("D * D**N * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("2 * 3 * 3 * int32"),
                                  ndt::type("D * D**N * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("2 * 2 * 3 * int32"),
                                  ndt::type("D * D**N * int32")));
  // Make sure an exponent of zero works as expected
  EXPECT_TRUE(ndt::pattern_match(ndt::type("5 * int32"),
                                 ndt::type("D**N * fixed * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("2 * int32"),
                                 ndt::type("3**N * fixed * int32")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("int32"), ndt::type("D**N * int32")));
  EXPECT_TRUE(
      ndt::pattern_match(ndt::type("int32"), ndt::type("3**N * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("3 * 4 * int32"),
                                 ndt::type("D**N * 3 * D * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("0 * 4 * int32"),
                                 ndt::type("D**N * N * D * int32")));
  // Can't have a negative exponent
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("int32"), ndt::type("D**N * E * int32")));
}

TEST(TypePatternMatchDims, Simple)
{
  // ndt::pattern_match_dims just matches the dims, it does not match the dtype,
  // and as part of the matching, it returns the two dtypes.
  std::map<nd::string, ndt::type> typevars;
  ndt::type cdt, pdt;
  EXPECT_TRUE(ndt::pattern_match_dims(ndt::type("3 * 4 * fixed * int32"),
                                      ndt::type("fixed * 4 * fixed * {x: int}"),
                                      typevars, cdt, pdt));
  EXPECT_EQ(ndt::type("int32"), cdt);
  EXPECT_EQ(ndt::type("{x: int}"), pdt);
}
