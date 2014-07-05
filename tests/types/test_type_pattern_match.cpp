//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/type_pattern_match.hpp>

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

    EXPECT_FALSE(ndt::pattern_match(ndt::type("int32"),
                                         ndt::type("int64")));
    EXPECT_FALSE(
        ndt::pattern_match(ndt::type("3 * int32"), ndt::type("T")));
    EXPECT_FALSE(ndt::pattern_match(ndt::type("int16"),
                                         ndt::type("A... * int32")));
    EXPECT_FALSE(ndt::pattern_match(ndt::type("4 * int32"),
                                         ndt::type("A... * 3 * M")));
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
        ndt::type("(int32, strided * var * 2 * int16) -> float32"),
        ndt::type("(S, A... * 2 * int16) -> T")));

    EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                         ndt::type("() -> T")));
    EXPECT_FALSE(ndt::pattern_match(ndt::type("() -> float32"),
                                         ndt::type("(T) -> T")));
    EXPECT_FALSE(ndt::pattern_match(ndt::type("(int32) -> float32"),
                                         ndt::type("(T) -> T")));
    EXPECT_FALSE(ndt::pattern_match(
        ndt::type("(int32, strided * var * 2 * int16) -> float32"),
        ndt::type("(S, M * 2 * int16) -> T")));
}
