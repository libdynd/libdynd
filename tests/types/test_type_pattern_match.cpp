//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>

#include <dynd/types/categorical_kind_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/pow_dimsym_type.hpp>
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

TEST(TypePatternMatch, Simple) {
  EXPECT_TYPE_MATCH("int32", "int32");
  EXPECT_TYPE_MATCH("T", "int32");
  EXPECT_TYPE_MATCH("A... * int32", "int32");
  EXPECT_TYPE_MATCH("A... * T", "int32");
  EXPECT_TYPE_MATCH("A... * 4 * M", "4 * int32");
  EXPECT_TYPE_MATCH("3 * A... * M", "3 * int32");
  EXPECT_TYPE_MATCH("Fixed**N * int32", "Fixed**3 * int32");
  EXPECT_TYPE_MATCH("A**N * var * int32", "Fixed**2 * var * int32");
  EXPECT_TYPE_MATCH("Fixed**N * int32", "Fixed * int32");
  EXPECT_TYPE_MATCH("4**N * 3 * int32", "4 * 4 * 3 * int32");

  EXPECT_TYPE_MATCH("4**N * M * int32", "4 * 4 * 2 * int32");
  EXPECT_TYPE_MATCH("4**N * N * int32", "4 * 4 * 2 * int32");

  EXPECT_FALSE(ndt::type("int32").match(ndt::type("int64")));
  EXPECT_FALSE(ndt::type("T").match(ndt::type("3 * int32")));
  EXPECT_FALSE(ndt::type("A... * int32").match(ndt::type("int16")));
  EXPECT_FALSE(ndt::type("A... * 3 * M").match(ndt::type("4 * int32")));
  EXPECT_FALSE(ndt::type("Fixed**N * var * int32").match(ndt::type("Fixed**3 * int32")));
  EXPECT_FALSE(ndt::type("Fixed**N * int32").match(ndt::type("var * Fixed**3 * int32")));
  EXPECT_FALSE(ndt::type("A**N * A * int32").match(ndt::type("Fixed**3 * var * int32")));
  EXPECT_FALSE(ndt::type("4**N * N * int32").match(ndt::type("4 * 4 * 3 * int32")));

  EXPECT_TYPE_MATCH("... * T", "... * int32");
}

TEST(TypePatternMatch, Any) {
  // Match various dtypes against "Any"
  EXPECT_TYPE_MATCH("Any", "Any");
  EXPECT_TYPE_MATCH("Any", "int32");
  EXPECT_TYPE_MATCH("Any", "(float32, Any)");
  EXPECT_TYPE_MATCH("Any", "{x: Any, y: bool}");
  EXPECT_TYPE_MATCH("Any", "pointer[complex]");
  EXPECT_TYPE_MATCH("Any", "?float64");

  // Match various dimensions + dtypes against "Any"
  EXPECT_TYPE_MATCH("Any", "Fixed * T");
  EXPECT_TYPE_MATCH("Any", "D * T");
  EXPECT_TYPE_MATCH("Any", "... * T");
  EXPECT_TYPE_MATCH("Any", "Dims... * float64");
  EXPECT_TYPE_MATCH("Any", "2 * 3 * complex[float32]");
  EXPECT_TYPE_MATCH("Any", "3 * {x: 2 * int32, y: var * int16}");
  EXPECT_TYPE_MATCH("Any", "3 * 5 * var * (int32, float16, 2 * int8)");

  // On the other hand, "Any" doesn't in general match against anything,
  // because a match with a symbolic candidate is saying "For all types that
  // match the candidate, they also match the pattern".
  EXPECT_FALSE(ndt::type("int32").match(ndt::type("Any")));
  EXPECT_FALSE(ndt::type("T").match(ndt::type("Any")));
  // TODO: This should fail to match
  //  EXPECT_FALSE(ndt::type("Fixed**2 * T").match(ndt::type("Any")));
  // TODO: This should fail to match
  //  EXPECT_FALSE(ndt::type("... * float32").match(ndt::type("Any")));

  // TODO: This should match
  EXPECT_TYPE_MATCH("... * T", "Any");
}

TEST(TypePatternMatch, VariadicTuple) {
  EXPECT_TYPE_MATCH("(...)", "(...)");
  EXPECT_TYPE_MATCH("(...)", "()");
  EXPECT_FALSE(ndt::type("()").match(ndt::type("(...)")));
  EXPECT_TYPE_MATCH("(...)", "(int32)");
  EXPECT_TYPE_MATCH("(int32, ...)", "(int32)");
  EXPECT_TYPE_MATCH("(...)", "(int32, int64)");
  EXPECT_TYPE_MATCH("(...)", "(int32, int64, float32)");
  EXPECT_TYPE_MATCH("(int32, ...)", "(int32, int64, float32)");
  EXPECT_TYPE_MATCH("(int32, int64, ...)", "(int32, int64, float32)");
  EXPECT_TYPE_MATCH("(int32, int64, float32, ...)", "(int32, int64, float32)");
  EXPECT_TYPE_MATCH("(int32, T, ...)", "(int32, int64, float32)");
  EXPECT_FALSE(ndt::type("(T, T, ...)").match(ndt::type("(int32, int64, float32)")));
  EXPECT_TYPE_MATCH("(T, T, ...)", "(int32, int32, float32)");
}

TEST(TypePatternMatch, Struct) {
  EXPECT_TYPE_MATCH("{x: int32}", "{x: int32}");
  EXPECT_TYPE_MATCH("{x: T}", "{x: int32}");
  EXPECT_TYPE_MATCH("A... * {x: B... * T}", "{x: int32}");
  EXPECT_TYPE_MATCH("A... * {x: T, y: T}", "100 * {x: int32, y: int32}");
  EXPECT_TYPE_MATCH("M * {x: T, y: T, u: S, v: S}", "100 * {x: int32, y: int32, u: int16, v: int16}");
  EXPECT_TYPE_MATCH("T", "{x: int32}");

  EXPECT_FALSE(
      ndt::type("100 * {x: int32, y: int32, u: int16, v: int32}").match(ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, VariadicStruct) {
  EXPECT_TYPE_MATCH("{...}", "{...}");
  EXPECT_TYPE_MATCH("{...}", "{}");
  EXPECT_FALSE(ndt::type("{}").match(ndt::type("{...}")));
  EXPECT_TYPE_MATCH("{...}", "{x: int32}");
  EXPECT_TYPE_MATCH("{x: int32, ...}", "{x: int32}");
  EXPECT_TYPE_MATCH("{...}", "{x: int32, y: int64}");
  EXPECT_TYPE_MATCH("{...}", "{x: int32, y: int64, z: float32}");
  EXPECT_TYPE_MATCH("{x: int32, ...}", "{x: int32, y: int64, z: float32}");
  EXPECT_FALSE(ndt::type("{x: int32, y: int64, z: float32}").match(ndt::type("{y: int32, ...}")));
  EXPECT_FALSE(ndt::type("{y: int64, ...}").match(ndt::type("{x: int32, y: int64, z: float32}")));
  EXPECT_TYPE_MATCH("{x: int32, y: int64, ...}", "{x: int32, y: int64, z: float32}");
  EXPECT_TYPE_MATCH("{x: int32, y: int64, z: float32, ...}", "{x: int32, y: int64, z: float32}");
  EXPECT_TYPE_MATCH("{x: int32, y: T, ...}", "{x: int32, y: int64, z: float32}");
  EXPECT_FALSE(ndt::type("{x: T, y: T, ...}").match(ndt::type("{x: int32, y: int64, z: float32}")));
  EXPECT_TYPE_MATCH("{x: T, y: T, ...}", "{x: int32, y: int32, z: float32}");
}

TEST(TypePatternMatch, Option) {
  EXPECT_TYPE_MATCH("?int32", "?int32");
  EXPECT_TYPE_MATCH("?T", "?int32");
  EXPECT_TYPE_MATCH("T", "?int32");
  EXPECT_TYPE_MATCH("M * {x: T, y: T, u: S, v: S}", "100 * {x: ?int32, y: ?int32, u: int16, v: int16}");
  EXPECT_TYPE_MATCH("M * {x: T, y: T, u: ?S, v: S}", "100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}");

  EXPECT_FALSE(
      ndt::type("M * {x: T, y: T, u: S, v: S}").match(ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}")));
}

/*
TEST(TypePatternMatch, Categorical)
{
  const char *a_vals[] = {"foo", "bar", "baz"};
  nd::array a = nd::empty(3, ndt::fixed_string_type::make(3, string_encoding_ascii));
  a.vals() = a_vals;

  EXPECT_TRUE(ndt::type("Categorical").match(ndt::categorical_type::make(a)));
  EXPECT_TRUE(ndt::type("Categorical").match(ndt::type("Categorical")));
  EXPECT_FALSE(ndt::type("Categorical").match(ndt::type("int32")));
}
*/

TEST(TypePatternMatch, FixedBytes) {
  EXPECT_TRUE(ndt::type("FixedBytes").match(ndt::type("fixed_bytes[8]")));
  EXPECT_TRUE(ndt::type("FixedBytes").match(ndt::type("fixed_bytes[16]")));
  EXPECT_FALSE(ndt::type("FixedBytes").match(ndt::type("bytes")));
  EXPECT_FALSE(ndt::type("FixedBytes").match(ndt::type("int32")));
}

TEST(TypePatternMatch, FixedString) {
  EXPECT_TRUE(ndt::type("FixedString").match(ndt::type("fixed_string[8]")));
  EXPECT_TRUE(ndt::type("FixedString").match(ndt::type("fixed_string[16]")));
  EXPECT_FALSE(ndt::type("FixedString").match(ndt::type("string")));
  EXPECT_FALSE(ndt::type("FixedString").match(ndt::type("int32")));
}

TEST(TypePatternMatch, ArrFuncProto) {
  EXPECT_TYPE_MATCH("() -> void", "() -> void");
  EXPECT_TYPE_MATCH("() -> T", "() -> void");
  EXPECT_FALSE(ndt::type("() -> void").match(ndt::type("() -> int32")));
  EXPECT_FALSE(ndt::type("() -> void").match(ndt::type("(int32) -> void")));
  EXPECT_FALSE(ndt::type("(int32) -> void").match(ndt::type("() -> void")));
  EXPECT_TYPE_MATCH("() -> T", "() -> float32");
  EXPECT_TYPE_MATCH("(T) -> T", "(float32) -> float32");
  EXPECT_TYPE_MATCH("(S) -> T", "(int32) -> float32");
  EXPECT_TYPE_MATCH("(S, A... * 2 * int16) -> T", "(int32, 5 * var * 2 * int16) -> float32");

  EXPECT_FALSE(ndt::type("(int32) -> float32").match(ndt::type("() -> T")));
  EXPECT_FALSE(ndt::type("() -> float32").match(ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::type("(int32) -> float32").match(ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::type("(int32, 5 * var * 2 * int16) -> float32").match(ndt::type("(S, M * 2 * int16) -> T")));
}

TEST(TypePatternMatch, VariadicArrFuncProto) {
  EXPECT_TYPE_MATCH("(...) -> void", "() -> void");
  EXPECT_FALSE(ndt::type("() -> void").match(ndt::type("(...) -> void")));
  EXPECT_TYPE_MATCH("(...) -> void", "(kw: int32) -> void");
  EXPECT_TYPE_MATCH("(int32, kw: T, ...) -> void", "(int32, kw: int32, x: float64) -> void");
  EXPECT_FALSE(ndt::type("(int32, wk: T, ...) -> void").match(ndt::type("(int32, kw: int32, x: float64) -> void")));
  EXPECT_TYPE_MATCH("(...) -> void", "(int32, ..., kw: int32, ...) -> void");
  EXPECT_TYPE_MATCH("(S, ...) -> T", "(int32, float32) -> float32");
  EXPECT_TYPE_MATCH("(Fixed**N * T, ..., func: (N * T) -> R) -> R",
                    "(2 * 3 * 4 * int32, float64, func: (3 * int32) -> bool) -> bool");
}

TEST(TypePatternMatch, NestedArrFuncProto) {
  EXPECT_TYPE_MATCH("(N * S, (M * S) -> T) -> N * T", "(3 * int32, (2 * int32) -> float64) -> 3 * float64");
  EXPECT_FALSE(ndt::type("(N * S, (M * S) -> T) -> N * T")
                   .match(ndt::type("(3 * int32, (2 * float64) -> float64) -> 3 * float64")));
  EXPECT_FALSE(ndt::type("(3 * int32, (2 * int32) -> float64) -> 2 * float64")
                   .match(ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_TYPE_MATCH("(N * S, (M * S) -> T) -> N * T", "(3 * int32, (2 * int32) -> float64) -> 3 * float64");
  EXPECT_TYPE_MATCH("(N * S, func: (M * S) -> T) -> N * T", "(3 * int32, func: (2 * int32) -> float64) -> 3 * float64");
  EXPECT_FALSE(ndt::type("(N * S, funk: (M * S) -> T) -> N * T")
                   .match(ndt::type("(3 * int32, func: (2 * int32) -> float64) -> 3 * float64")));
}

TEST(TypePatternMatch, Pow) {
  // Match pow_dimsym against itself
  EXPECT_TYPE_MATCH("Fixed**N * int32", "Fixed**N * int32");
  EXPECT_TYPE_MATCH("Fixed**N * int32", "Fixed**M * int32");
  // Match fixed_dim against fixed_dimsym within the power's base
  EXPECT_TYPE_MATCH("Fixed**N * int32", "3**N * int32");
  EXPECT_TYPE_MATCH("Fixed**M * int32", "3**N * int32");
  // Ensure that the typevar D is constrained in the power's base
  EXPECT_TYPE_MATCH("D * D**N * int32", "Fixed * Fixed * Fixed * int32");

  EXPECT_TYPE_MATCH("D * D**N * int32", "3 * 3 * 3 * int32");
  EXPECT_FALSE(ndt::type("D * D**N * int32").match(ndt::type("2 * 3 * 3 * int32")));
  EXPECT_FALSE(ndt::type("2 * 2 * 3 * int32").match(ndt::type("D * D**N * int32")));
  // Make sure an exponent of zero works as expected
  EXPECT_TYPE_MATCH("D**N * Fixed * int32", "5 * int32");
  EXPECT_TYPE_MATCH("3**N * Fixed * int32", "2 * int32");
  EXPECT_TYPE_MATCH("D**N * int32", "int32");
  EXPECT_TYPE_MATCH("3**N * int32", "int32");
  EXPECT_TYPE_MATCH("D**N * 3 * D * int32", "3 * 4 * int32");
  EXPECT_TYPE_MATCH("D**N * N * D * int32", "0 * 4 * int32");
  // Can't have a negative exponent
  EXPECT_FALSE(ndt::type("int32").match(ndt::type("D**N * E * int32")));
}

TEST(TypePatternMatch, TypeVarConstructed) {
#ifdef DYND_CUDA
  EXPECT_TYPE_MATCH("M[int32]", "cuda_device[int32]");
// EXPECT_TYPE_MATCH("M[10 * 5 * int32]", "cuda_device[10 * 5 * int32]");
// EXPECT_TYPE_MATCH("M[Dims... * T]", "cuda_device[7 * int32]");
#endif
}

TEST(TypePatternMatch, Broadcast) {
  // Confirm that "T..." type variables broadcast together as they match
  std::map<std::string, ndt::type> tp_vars;
  EXPECT_TRUE(ndt::type("Dims... * int32").match(ndt::type("3 * 1 * int32"), tp_vars));
  EXPECT_TRUE(ndt::type("Dims... * float32").match(ndt::type("1 * 2 * float32"), tp_vars));
  EXPECT_EQ(ndt::type("3 * 2 * bool"), ndt::substitute(ndt::type("Dims... * bool"), tp_vars, true));
}

TEST(TypePatternMatch, TypeVar) {
  EXPECT_TYPE_MATCH("T", "T");
  EXPECT_TYPE_MATCH("T", "S");
  EXPECT_TYPE_MATCH("(T)", "(T)");
  EXPECT_TYPE_MATCH("(T)", "(S)");
  EXPECT_TYPE_MATCH("(T)", "(S)");
  EXPECT_TYPE_MATCH("(T,T)", "(T,T)");
  EXPECT_TYPE_MATCH("(S,T)", "(T,T)");
  EXPECT_FALSE(ndt::type("(T,T)").match(ndt::type("(S,T)")));
  EXPECT_TYPE_MATCH("(T,T)->T", "(T,T)->T");
  EXPECT_TYPE_MATCH("(S,T)->T", "(T,T)->T");
  EXPECT_FALSE(ndt::type("(T,T)->T").match(ndt::type("(S,T)->T")));
}
