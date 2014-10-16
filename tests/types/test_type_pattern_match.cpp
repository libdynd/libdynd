//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/pow_dimsym_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypePatternMatch, Simple) {
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"),
                                      ndt::type("int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"),
                                      ndt::type("T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"),
                                      ndt::type("A... * int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("int32"),
                                      ndt::type("A... * T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("4 * int32"),
                                      ndt::type("A... * 4 * M")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("3 * int32"),
                                      ndt::type("3 * A... * M")));
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

  EXPECT_FALSE(ndt::pattern_match(ndt::type("int32"),
                                        ndt::type("int64")));
  EXPECT_FALSE(
      ndt::pattern_match(ndt::type("3 * int32"), ndt::type("T")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("int16"),
                                        ndt::type("A... * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("4 * int32"),
                                        ndt::type("A... * 3 * M")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("fixed**3 * int32"),
                                      ndt::type("fixed**N * var * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("var * fixed**3 * int32"),
                                      ndt::type("fixed**N * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("fixed**3 * var * int32"),
                                      ndt::type("A**N * A * int32")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("4 * 4 * 3 * int32"),
                                      ndt::type("4**N * N * int32")));
}

TEST(TypePatternMatch, Struct) {
  EXPECT_TRUE(ndt::pattern_match(ndt::type("{x: int32}"),
                                      ndt::type("{x: int32}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("{x: int32}"),
                                      ndt::type("{x: T}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("{x: int32}"),
                                      ndt::type("A... * {x: B... * T}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("100 * {x: int32, y: int32}"),
                                      ndt::type("A... * {x: T, y: T}")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("100 * {x: int32, y: int32, u: int16, v: int16}"),
                                      ndt::type("M * {x: T, y: T, u: S, v: S}")));

  EXPECT_FALSE(ndt::pattern_match(ndt::type("100 * {x: int32, y: int32, u: int16, v: int32}"),
                                      ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, Option) {
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"),
                                      ndt::type("?int32")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"),
                                      ndt::type("?T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("?int32"),
                                      ndt::type("T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("100 * {x: ?int32, y: ?int32, u: int16, v: int16}"),
                                      ndt::type("M * {x: T, y: T, u: S, v: S}")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: ?S, v: S}")));

  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}"),
      ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, FuncProto) {
  EXPECT_TRUE(ndt::pattern_match(ndt::type("() -> void"),
                                      ndt::type("() -> void")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("() -> float32"),
                                      ndt::type("() -> T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("(float32) -> float32"),
                                      ndt::type("(T) -> T")));
  EXPECT_TRUE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                      ndt::type("(S) -> T")));
  EXPECT_TRUE(ndt::pattern_match(
      ndt::type("(int32, 5 * var * 2 * int16) -> float32"),
      ndt::type("(S, A... * 2 * int16) -> T")));

  EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                        ndt::type("() -> T")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("() -> float32"),
                                        ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                        ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::pattern_match(
      ndt::type("(int32, 5 * var * 2 * int16) -> float32"),
      ndt::type("(S, M * 2 * int16) -> T")));
}

TEST(TypePatternMatch, Strided) {
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
