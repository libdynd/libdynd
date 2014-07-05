//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/type_substitute.hpp>
#include <dynd/types/dim_fragment_type.hpp>

using namespace std;
using namespace dynd;

TEST(TypeSubstitute, SimpleNoSubstitutions) {
  map<nd::string, ndt::type> typevars;
  EXPECT_EQ(ndt::type("int32"),
            ndt::substitute(ndt::type("int32"), typevars, false));
  EXPECT_EQ(ndt::type("int32"),
            ndt::substitute(ndt::type("int32"), typevars, true));
  EXPECT_EQ(ndt::type("T"),
            ndt::substitute(ndt::type("T"), typevars, false));
  EXPECT_THROW(ndt::substitute(ndt::type("T"), typevars, true),
               invalid_argument);
  EXPECT_EQ(ndt::type("A... * int32"),
            ndt::substitute(ndt::type("A... * int32"), typevars, false));
  EXPECT_THROW(ndt::substitute(ndt::type("A... * int32"), typevars, true),
               invalid_argument);
}

TEST(TypeSubstitute, SimpleSubstitution) {
  map<nd::string, ndt::type> typevars;
  typevars["Tint"] = ndt::type("int32");
  typevars["Tsym"] = ndt::type("S");
  typevars["Mstrided"] = ndt::type("strided * void");
  typevars["Mfixed"] = ndt::type("8 * void");
  typevars["Mvar"] = ndt::type("var * void");
  typevars["Msym"] = ndt::type("N * void");
  typevars["Aempty"] = ndt::make_dim_fragment(0, ndt::make_type<void>());
  typevars["Astrided"] =
      ndt::make_dim_fragment(1, ndt::type("strided * void"));
  typevars["Afixed"] =
      ndt::make_dim_fragment(1, ndt::type("5 * void"));
  typevars["Avar"] =
      ndt::make_dim_fragment(1, ndt::type("var * void"));
  typevars["Amulti"] =
      ndt::make_dim_fragment(3, ndt::type("strided * var * 3 * void"));

  EXPECT_EQ(ndt::type("int32"),
            ndt::substitute(ndt::type("Tint"), typevars, false));
  EXPECT_EQ(ndt::type("int32"),
            ndt::substitute(ndt::type("Tint"), typevars, true));
  EXPECT_EQ(ndt::type("?int32"),
            ndt::substitute(ndt::type("?Tint"), typevars, false));
  EXPECT_EQ(ndt::type("?int32"),
            ndt::substitute(ndt::type("?Tint"), typevars, true));
  EXPECT_EQ(ndt::type("S"),
            ndt::substitute(ndt::type("Tsym"), typevars, false));
  EXPECT_THROW(ndt::substitute(ndt::type("Tsym"), typevars, true),
               invalid_argument);
  EXPECT_EQ(ndt::type("strided * int32"),
            ndt::substitute(ndt::type("Mstrided * Tint"), typevars, false));
  EXPECT_EQ(ndt::type("strided * int32"),
            ndt::substitute(ndt::type("Mstrided * Tint"), typevars, true));
  EXPECT_EQ(ndt::type("8 * int32"),
            ndt::substitute(ndt::type("Mfixed * Tint"), typevars, false));
  EXPECT_EQ(ndt::type("8 * int32"),
            ndt::substitute(ndt::type("Mfixed * Tint"), typevars, true));
  EXPECT_EQ(ndt::type("var * int32"),
            ndt::substitute(ndt::type("Mvar * Tint"), typevars, false));
  EXPECT_EQ(ndt::type("var * int32"),
            ndt::substitute(ndt::type("Mvar * Tint"), typevars, true));
  EXPECT_EQ(
      ndt::type("var * int32"),
      ndt::substitute(ndt::type("Aempty... * Mvar * Tint"), typevars, false));
  EXPECT_EQ(
      ndt::type("var * int32"),
      ndt::substitute(ndt::type("Mvar * Aempty... * Tint"), typevars, true));
  EXPECT_EQ(
      ndt::type("strided * var * int32"),
      ndt::substitute(ndt::type("Astrided... * Mvar * Tint"), typevars, false));
  EXPECT_EQ(
      ndt::type("var * strided * int32"),
      ndt::substitute(ndt::type("Mvar * Astrided... * Tint"), typevars, true));
  EXPECT_EQ(
      ndt::type("5 * var * int32"),
      ndt::substitute(ndt::type("Afixed... * Mvar * Tint"), typevars, false));
  EXPECT_EQ(
      ndt::type("var * 5 * int32"),
      ndt::substitute(ndt::type("Mvar * Afixed... * Tint"), typevars, true));
  EXPECT_EQ(
      ndt::type("var * var * int32"),
      ndt::substitute(ndt::type("Avar... * Mvar * Tint"), typevars, false));
  EXPECT_EQ(
      ndt::type("var * var * int32"),
      ndt::substitute(ndt::type("Mvar * Avar... * Tint"), typevars, true));
  EXPECT_EQ(
      ndt::type("strided * var * 3 * var * int32"),
      ndt::substitute(ndt::type("Amulti... * Mvar * Tint"), typevars, false));
  EXPECT_EQ(
      ndt::type("var * strided * var * 3 * int32"),
      ndt::substitute(ndt::type("Mvar * Amulti... * Tint"), typevars, true));
}

TEST(TypeSubstitute, Tuple) {
  map<nd::string, ndt::type> typevars;
  typevars["T"] = ndt::type("int32");
  typevars["M"] = ndt::type("3 * void");
  typevars["A"] =
      ndt::make_dim_fragment(3, ndt::type("var * strided * 4 * void"));

  EXPECT_EQ(ndt::type("(int, real)"),
            ndt::substitute(ndt::type("(int, real)"), typevars, false));
  EXPECT_EQ(ndt::type("(int, real)"),
            ndt::substitute(ndt::type("(int, real)"), typevars, true));
  EXPECT_EQ(ndt::type("(int32, 3 * real)"),
            ndt::substitute(ndt::type("(T, M * real)"), typevars, false));
  EXPECT_EQ(ndt::type("(int32, 3 * real)"),
            ndt::substitute(ndt::type("(T, M * real)"), typevars, true));

  EXPECT_EQ(
      ndt::type("(var * strided * 4 * int32, 3 * real)"),
      ndt::substitute(ndt::type("(A... * T, M * real)"), typevars, false));
  EXPECT_EQ(
      ndt::type("(var * strided * 4 * int32, 3 * real)"),
      ndt::substitute(ndt::type("(A... * T, M * real)"), typevars, true));
}

TEST(TypeSubstitute, Struct) {
  map<nd::string, ndt::type> typevars;
  typevars["T"] = ndt::type("int32");
  typevars["M"] = ndt::type("3 * void");
  typevars["A"] =
      ndt::make_dim_fragment(3, ndt::type("var * strided * 4 * void"));

  EXPECT_EQ(ndt::type("{x: int, y: real}"),
    ndt::substitute(ndt::type("{x: int, y: real}"), typevars, false));
  EXPECT_EQ(ndt::type("{x: int, y: real}"),
    ndt::substitute(ndt::type("{x: int, y: real}"), typevars, true));
  EXPECT_EQ(ndt::type("{x: int32, y: 3 * real}"),
            ndt::substitute(ndt::type("{x: T, y: M * real}"), typevars, false));
  EXPECT_EQ(ndt::type("{x: int32, y: 3 * real}"),
            ndt::substitute(ndt::type("{x: T, y: M * real}"), typevars, true));

  EXPECT_EQ(ndt::type("{x: var * strided * 4 * int32, y: 3 * real}"),
            ndt::substitute(ndt::type("{x: A... * T, y: M * real}"), typevars,
                            false));
  EXPECT_EQ(ndt::type("{x: var * strided * 4 * int32, y: 3 * real}"),
            ndt::substitute(ndt::type("{x: A... * T, y: M * real}"), typevars,
                            true));
}

TEST(TypeSubstitute, FuncProto) {
  map<nd::string, ndt::type> typevars;
  typevars["T"] = ndt::type("int32");
  typevars["M"] = ndt::type("3 * void");
  typevars["A"] =
      ndt::make_dim_fragment(3, ndt::type("var * strided * 4 * void"));

  EXPECT_EQ(ndt::type("(int, real) -> complex"),
    ndt::substitute(ndt::type("(int, real) -> complex"), typevars, false));
  EXPECT_EQ(ndt::type("(int, real) -> complex"),
    ndt::substitute(ndt::type("(int, real) -> complex"), typevars, true));
  EXPECT_EQ(ndt::type("(int32, 3 * real) -> var * strided * 4 * complex"),
    ndt::substitute(ndt::type("(T, M * real) -> A... * complex"), typevars, false));
}
