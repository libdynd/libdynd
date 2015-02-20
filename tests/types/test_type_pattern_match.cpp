//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/pow_dimsym_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypePatternMatch, Simple)
{
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("int32")));
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("T")));
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("A... * int32")));
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("A... * T")));
  EXPECT_TRUE(ndt::type("A... * 4 * M").matches(ndt::type("4 * int32")));
  EXPECT_TRUE(ndt::type("3 * int32").matches(ndt::type("3 * A... * M")));
  EXPECT_TRUE(
      ndt::type("Fixed**N * int32").matches(ndt::type("Fixed**3 * int32")));
  EXPECT_TRUE(ndt::type("A**N * var * int32")
                  .matches(ndt::type("Fixed**2 * var * int32")));
  EXPECT_TRUE(
      ndt::type("Fixed**N * int32").matches(ndt::type("Fixed * int32")));
  EXPECT_TRUE(
      ndt::type("4**N * 3 * int32").matches(ndt::type("4 * 4 * 3 * int32")));

  // TODO: FIX these
  //  EXPECT_TRUE(
  //    ndt::type("4**N * M * int32").matches(ndt::type("4 * 4 * 2 * int32")));
  // EXPECT_TRUE(
  //   ndt::type("4**N * N * int32").matches(ndt::type("4 * 4 * 2 * int32")));

  EXPECT_FALSE(ndt::type("int32").matches(ndt::type("int64")));
  EXPECT_FALSE(ndt::type("3 * int32").matches(ndt::type("T")));
  EXPECT_FALSE(ndt::type("int16").matches(ndt::type("A... * int32")));
  EXPECT_FALSE(ndt::type("4 * int32").matches(ndt::type("A... * 3 * M")));
  EXPECT_FALSE(ndt::type("Fixed**3 * int32")
                   .matches(ndt::type("Fixed**N * var * int32")));
  EXPECT_FALSE(ndt::type("var * Fixed**3 * int32")
                   .matches(ndt::type("Fixed**N * int32")));
  EXPECT_FALSE(ndt::type("Fixed**3 * var * int32")
                   .matches(ndt::type("A**N * A * int32")));
  EXPECT_FALSE(
      ndt::type("4 * 4 * 3 * int32").matches(ndt::type("4**N * N * int32")));

  EXPECT_TRUE(ndt::type("... * int32").matches(ndt::type("... * T")));
}

TEST(TypePatternMatch, Any)
{
  // Match various dtypes against "Any"
  EXPECT_TRUE(ndt::type("Any").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("Any").matches(ndt::type("int32")));
  EXPECT_TRUE(ndt::type("Any").matches(ndt::type("(float32, Any)")));
  EXPECT_TRUE(ndt::type("{x: Any, y: bool}").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("pointer[complex]").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("?float64").matches(ndt::type("Any")));

  // Match various dimensions + dtypes against "Any"
  EXPECT_TRUE(ndt::type("Fixed * T").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("D * T").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("... * T").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("Dims... * float64").matches(ndt::type("Any")));
  EXPECT_TRUE(ndt::type("Any").matches(ndt::type("2 * 3 * complex[float32]")));
  EXPECT_TRUE(ndt::type("Any")
                  .matches(ndt::type("3 * {x: 2 * int32, y: var * int16}")));
  EXPECT_TRUE(ndt::type("Any").matches(
      ndt::type("3 * 5 * var * (int32, float16, 2 * int8)")));
}

TEST(TypePatternMatch, VariadicTuple)
{
  EXPECT_TRUE(ndt::type("(...)").matches(ndt::type("(...)")));
  EXPECT_TRUE(ndt::type("(...)").matches(ndt::type("()")));
  EXPECT_FALSE(ndt::type("()").matches(ndt::type("(...)")));
  EXPECT_TRUE(ndt::type("(...)").matches(ndt::type("(int32)")));
  EXPECT_TRUE(ndt::type("(int32, ...)").matches(ndt::type("(int32)")));
  EXPECT_TRUE(ndt::type("(...)").matches(ndt::type("(int32, int64)")));
  EXPECT_TRUE(ndt::type("(...)").matches(ndt::type("(int32, int64, float32)")));
  EXPECT_TRUE(
      ndt::type("(int32, ...)").matches(ndt::type("(int32, int64, float32)")));
  EXPECT_TRUE(ndt::type("(int32, int64, ...)")
                  .matches(ndt::type("(int32, int64, float32)")));
  EXPECT_TRUE(ndt::type("(int32, int64, float32, ...)")
                  .matches(ndt::type("(int32, int64, float32)")));
  EXPECT_TRUE(ndt::type("(int32, T, ...)")
                  .matches(ndt::type("(int32, int64, float32)")));
  EXPECT_FALSE(
      ndt::type("(T, T, ...)").matches(ndt::type("(int32, int64, float32)")));
  EXPECT_TRUE(
      ndt::type("(T, T, ...)").matches(ndt::type("(int32, int32, float32)")));
}

TEST(TypePatternMatch, Struct)
{
  EXPECT_TRUE(ndt::type("{x: int32}").matches(ndt::type("{x: int32}")));
  EXPECT_TRUE(ndt::type("{x: int32}").matches(ndt::type("{x: T}")));
  EXPECT_TRUE(
      ndt::type("{x: int32}").matches(ndt::type("A... * {x: B... * T}")));
  EXPECT_TRUE(ndt::type("A... * {x: T, y: T}")
                  .matches(ndt::type("100 * {x: int32, y: int32}")));
  EXPECT_TRUE(ndt::type("M * {x: T, y: T, u: S, v: S}").matches(
      ndt::type("100 * {x: int32, y: int32, u: int16, v: int16}")));

  EXPECT_FALSE(ndt::type("100 * {x: int32, y: int32, u: int16, v: int32}")
                   .matches(ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, VariadicStruct)
{
  EXPECT_TRUE(ndt::type("{...}").matches(ndt::type("{...}")));
  EXPECT_TRUE(ndt::type("{}").matches(ndt::type("{...}")));
  EXPECT_FALSE(ndt::type("{...}").matches(ndt::type("{}")));
  EXPECT_TRUE(ndt::type("{x: int32}").matches(ndt::type("{...}")));
  EXPECT_FALSE(ndt::type("{x: int32, ...}").matches(ndt::type("{x: int32}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64}").matches(ndt::type("{...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64, z: float32}")
                  .matches(ndt::type("{...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64, z: float32}")
                  .matches(ndt::type("{x: int32, ...}")));
  EXPECT_FALSE(ndt::type("{x: int32, y: int64, z: float32}")
                   .matches(ndt::type("{y: int32, ...}")));
  EXPECT_FALSE(ndt::type("{x: int32, y: int64, z: float32}")
                   .matches(ndt::type("{y: int64, ...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64, z: float32}")
                  .matches(ndt::type("{x: int32, y: int64, ...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64, z: float32}")
                  .matches(ndt::type("{x: int32, y: int64, z: float32, ...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int64, z: float32}")
                  .matches(ndt::type("{x: int32, y: T, ...}")));
  EXPECT_FALSE(ndt::type("{x: int32, y: int64, z: float32}")
                   .matches(ndt::type("{x: T, y: T, ...}")));
  EXPECT_TRUE(ndt::type("{x: int32, y: int32, z: float32}")
                  .matches(ndt::type("{x: T, y: T, ...}")));
}

TEST(TypePatternMatch, Option)
{
  EXPECT_TRUE(ndt::type("?int32").matches(ndt::type("?int32")));
  EXPECT_TRUE(ndt::type("?int32").matches(ndt::type("?T")));
  EXPECT_TRUE(ndt::type("?int32").matches(ndt::type("T")));
  EXPECT_TRUE(ndt::type("M * {x: T, y: T, u: S, v: S}").matches(
      ndt::type("100 * {x: ?int32, y: ?int32, u: int16, v: int16}")));
  EXPECT_TRUE(ndt::type("M * {x: T, y: T, u: ?S, v: S}").matches(
      ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}")));

  EXPECT_FALSE(ndt::type("100 * {x: ?int32, y: ?int32, u: ?int16, v: int16}")
                   .matches(ndt::type("M * {x: T, y: T, u: S, v: S}")));
}

TEST(TypePatternMatch, ArrFuncProto)
{
  EXPECT_TRUE(ndt::type("() -> void").matches(ndt::type("() -> void")));
  EXPECT_TRUE(ndt::type("() -> void").matches(ndt::type("() -> T")));
  EXPECT_FALSE(ndt::type("() -> void").matches(ndt::type("() -> int32")));
  EXPECT_FALSE(ndt::type("() -> void").matches(ndt::type("(int32) -> void")));
  EXPECT_FALSE(ndt::type("(int32) -> void").matches(ndt::type("() -> void")));
  EXPECT_TRUE(ndt::type("() -> float32").matches(ndt::type("() -> T")));
  EXPECT_TRUE(ndt::type("(float32) -> float32").matches(ndt::type("(T) -> T")));
  EXPECT_TRUE(ndt::type("(int32) -> float32").matches(ndt::type("(S) -> T")));
  EXPECT_TRUE(ndt::type("(S, A... * 2 * int16) -> T").matches(
      ndt::type("(int32, 5 * var * 2 * int16) -> float32")));

  EXPECT_FALSE(ndt::type("(int32) -> float32").matches(ndt::type("() -> T")));
  EXPECT_FALSE(ndt::type("() -> float32").matches(ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::type("(int32) -> float32").matches(ndt::type("(T) -> T")));
  EXPECT_FALSE(ndt::type("(int32, 5 * var * 2 * int16) -> float32")
                   .matches(ndt::type("(S, M * 2 * int16) -> T")));
}

TEST(TypePatternMatch, VariadicArrFuncProto)
{
  EXPECT_TRUE(ndt::type("(...) -> void").matches(ndt::type("() -> void")));
  EXPECT_FALSE(ndt::type("() -> void").matches(ndt::type("(...) -> void")));
  EXPECT_TRUE(
      ndt::type("(kw: int32) -> void").matches(ndt::type("(...) -> void")));
  EXPECT_TRUE(ndt::type("(int32, kw: int32, x: float64) -> void")
                  .matches(ndt::type("(int32, kw: T, ...) -> void")));
  EXPECT_FALSE(ndt::type("(int32, kw: int32, x: float64) -> void")
                   .matches(ndt::type("(int32, wk: T, ...) -> void")));
  EXPECT_TRUE(ndt::type("(int32, ..., kw: int32, ...) -> void")
                  .matches(ndt::type("(...) -> void")));
  EXPECT_TRUE(ndt::type("(int32, float32) -> float32")
                  .matches(ndt::type("(S, ...) -> T")));

  std::cout << ndt::type("(Fixed**N * T)")
                   .matches(ndt::type("(2 * 3 * 4 * int32)")) << std::endl;
  std::cout << ndt::type("(Fixed**N * T, ...)").matches(
                   ndt::type("(2 * 3 * 4 * int32, float64)")) << std::endl;
  EXPECT_TRUE(ndt::type("(Fixed**N * T, ..., func: (N * T) -> R) -> R").matches(
      ndt::type(
          "(2 * 3 * 4 * int32, float64, func: (3 * int32) -> bool) -> bool")));
}

TEST(TypePatternMatch, NestedArrFuncProto)
{
  EXPECT_TRUE(ndt::type("(3 * int32, (2 * int32) -> float64) -> 3 * float64")
                  .matches(ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_FALSE(ndt::type("(3 * int32, (2 * float64) -> float64) -> 3 * float64")
                   .matches(ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_FALSE(ndt::type("(3 * int32, (2 * int32) -> float64) -> 2 * float64")
                   .matches(ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_TRUE(ndt::type("(3 * int32, (2 * int32) -> float64) -> 3 * float64")
                  .matches(ndt::type("(N * S, (M * S) -> T) -> N * T")));
  EXPECT_TRUE(
      ndt::type("(3 * int32, func: (2 * int32) -> float64) -> 3 * float64")
          .matches(ndt::type("(N * S, func: (M * S) -> T) -> N * T")));
  EXPECT_FALSE(
      ndt::type("(3 * int32, func: (2 * int32) -> float64) -> 3 * float64")
          .matches(ndt::type("(N * S, funk: (M * S) -> T) -> N * T")));
}

TEST(TypePatternMatch, Strided)
{
  // cfixed and fixed can match against strided
  EXPECT_TRUE(
      ndt::type("cfixed[3] * int32").matches(ndt::type("Fixed * int32")));
  EXPECT_TRUE(ndt::type("3 * int32").matches(ndt::type("Fixed * int32")));
  // cfixed can match against strided if the sizes match
  EXPECT_TRUE(ndt::type("cfixed[3] * int32").matches(ndt::type("3 * int32")));
  // Things do not hold in other cases
  EXPECT_FALSE(ndt::type("3 * int32").matches(ndt::type("cfixed[3] * int32")));
  EXPECT_FALSE(ndt::type("cfixed[3] * int32").matches(ndt::type("4 * int32")));
}

TEST(TypePatternMatch, Pow)
{
  // Match pow_dimsym against itself
  EXPECT_TRUE(
      ndt::type("Fixed**N * int32").matches(ndt::type("Fixed**N * int32")));
  EXPECT_TRUE(
      ndt::type("Fixed**N * int32").matches(ndt::type("Fixed**M * int32")));
  // Match fixed_dim against fixed_dimsym within the power's base
  EXPECT_TRUE(ndt::type("3**N * int32").matches(ndt::type("Fixed**N * int32")));
  EXPECT_TRUE(ndt::type("3**N * int32").matches(ndt::type("Fixed**M * int32")));
  // Ensure that the typevar D is constrained in the power's base
  // TODO: FIX THIS
  //  EXPECT_TRUE(ndt::type("D * D**N * int32")
  //                .matches(ndt::type("Fixed * Fixed * Fixed * int32")));

  EXPECT_TRUE(
      ndt::type("3 * 3 * 3 * int32").matches(ndt::type("D * D**N * int32")));
  EXPECT_FALSE(
      ndt::type("2 * 3 * 3 * int32").matches(ndt::type("D * D**N * int32")));
  EXPECT_FALSE(
      ndt::type("2 * 2 * 3 * int32").matches(ndt::type("D * D**N * int32")));
  // Make sure an exponent of zero works as expected
  EXPECT_TRUE(
      ndt::type("5 * int32").matches(ndt::type("D**N * Fixed * int32")));
  EXPECT_TRUE(
      ndt::type("2 * int32").matches(ndt::type("3**N * Fixed * int32")));
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("D**N * int32")));
  EXPECT_TRUE(ndt::type("int32").matches(ndt::type("3**N * int32")));
  EXPECT_TRUE(
      ndt::type("3 * 4 * int32").matches(ndt::type("D**N * 3 * D * int32")));
  EXPECT_TRUE(
      ndt::type("0 * 4 * int32").matches(ndt::type("D**N * N * D * int32")));
  // Can't have a negative exponent
  EXPECT_FALSE(ndt::type("int32").matches(ndt::type("D**N * E * int32")));
}

TEST(TypePatternMatch, TypeVarConstructed)
{
#ifdef DYND_CUDA
  EXPECT_TRUE(ndt::type("cuda_device[int32]").matches(ndt::type("M[int32]")));
//  EXPECT_TRUE(ndt::type("cuda_device[10 * 5 * int32]")
//                .matches(ndt::type("M[10 * 5 * int32]")));
// EXPECT_TRUE(
//   ndt::type("cuda_device[7 * int32]").matches(ndt::type("M[Dims... * T]")));
#endif
}