//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/callable.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/int_kind_sym_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

TEST(SymbolicTypes, CreateFuncProto) {
  ndt::type tp;
  const ndt::callable_type *fpt;

  // Function prototype from C++ template parameter
  tp = ndt::make_type<int64_t(float, int32_t, double)>();
  EXPECT_EQ(callable_id, tp.get_id());
  EXPECT_EQ(sizeof(ndt::callable_type::data_type), tp.get_data_size());
  EXPECT_EQ(alignof(nd::callable), tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  // function prototype is not actually symbolic, it is
  // used to store callable objects
  EXPECT_FALSE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  fpt = tp.extended<ndt::callable_type>();
  ASSERT_EQ(3, fpt->get_narg());
  EXPECT_EQ(ndt::make_type<float>(), fpt->get_argument_types()[0]);
  EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_argument_types()[1]);
  EXPECT_EQ(ndt::make_type<double>(), fpt->get_argument_types()[2]);
  EXPECT_EQ(ndt::make_type<int64_t>(), fpt->get_return_type());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
  EXPECT_EQ("(float32, int32, float64) -> int64", tp.str());

  // Dynamic type properties
  EXPECT_EQ(ndt::make_type<int64_t>(), tp.p<ndt::type>("return_type"));
  nd::array ptp = tp.p<std::vector<ndt::type>>("pos_types");
  EXPECT_EQ(ndt::type("3 * type"), ptp.get_type());
  ASSERT_EQ(3, ptp.get_dim_size());
  EXPECT_EQ(ndt::make_type<float>(), ptp(0).as<ndt::type>());
  EXPECT_EQ(ndt::make_type<int32_t>(), ptp(1).as<ndt::type>());
  EXPECT_EQ(ndt::make_type<double>(), ptp(2).as<ndt::type>());

  // Exercise a few different variations
  tp = ndt::make_type<int8_t()>();
  fpt = tp.extended<ndt::callable_type>();
  ASSERT_EQ(0, fpt->get_narg());
  EXPECT_EQ(ndt::make_type<int8_t>(), fpt->get_return_type());

  tp = ndt::make_type<int16_t(int32_t)>();
  fpt = tp.extended<ndt::callable_type>();
  ASSERT_EQ(1, fpt->get_narg());
  EXPECT_EQ(ndt::make_type<int16_t>(), fpt->get_return_type());
  EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_argument_types()[0]);

  tp = ndt::make_type<int16_t(int32_t, int64_t)>();
  fpt = tp.extended<ndt::callable_type>();
  ASSERT_EQ(2, fpt->get_narg());
  EXPECT_EQ(ndt::make_type<int16_t>(), fpt->get_return_type());
  EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_argument_types()[0]);
  EXPECT_EQ(ndt::make_type<int64_t>(), fpt->get_argument_types()[1]);
}

TEST(SymbolicTypes, CreateTypeVar) {
  ndt::type tp;
  const ndt::typevar_type *tvt;

  // Simple TypeVar
  tp = ndt::typevar_type::make("Blah");
  EXPECT_EQ(typevar_id, tp.get_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ(1u, tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  tvt = tp.extended<ndt::typevar_type>();
  EXPECT_EQ("Blah", tvt->get_name());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
  EXPECT_EQ("Blah", tp.str());

  // Dynamic type properties
  EXPECT_EQ("Blah", tp.p<std::string>("name"));

  // The typevar name must start with a capital
  // and look like an identifier
  EXPECT_THROW(ndt::typevar_type::make(""), type_error);
  EXPECT_THROW(ndt::typevar_type::make("blah"), type_error);
  EXPECT_THROW(ndt::typevar_type::make("T "), type_error);
  EXPECT_THROW(ndt::typevar_type::make("123"), type_error);
  EXPECT_THROW(ndt::typevar_type::make("Two+"), type_error);
}

TEST(SymbolicTypes, CreateTypeVarDim) {
  ndt::type tp;
  const ndt::typevar_dim_type *tvt;

  // Simple Dimension TypeVar
  tp = ndt::typevar_dim_type::make("Blah", ndt::make_type<int>());
  EXPECT_EQ(typevar_dim_id, tp.get_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ(1u, tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  tvt = tp.extended<ndt::typevar_dim_type>();
  EXPECT_EQ("Blah", tvt->get_name());
  EXPECT_EQ(ndt::make_type<int>(), tvt->get_element_type());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
  EXPECT_EQ("Blah * int32", tp.str());

  // Dynamic type properties
  EXPECT_EQ("Blah", tp.p<std::string>("name"));

  // The typevar name must start with a capital
  // and look like an identifier
  EXPECT_THROW(ndt::typevar_dim_type::make("", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::typevar_dim_type::make("blah", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::typevar_dim_type::make("T ", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::typevar_dim_type::make("123", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::typevar_dim_type::make("Two+", ndt::make_type<int>()), type_error);
}

TEST(SymbolicTypes, CreateEllipsisDim) {
  ndt::type tp;
  const ndt::ellipsis_dim_type *et;

  // Named Ellipsis Dimension
  tp = ndt::make_ellipsis_dim("Blah", ndt::make_type<int>());
  EXPECT_EQ(ellipsis_dim_id, tp.get_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ(1u, tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  et = tp.extended<ndt::ellipsis_dim_type>();
  EXPECT_EQ("Blah", et->get_name());
  EXPECT_EQ(ndt::make_type<int>(), et->get_element_type());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
  EXPECT_EQ("Blah... * int32", tp.str());

  // Dynamic type properties
  //  Todo: Fix this
  //  EXPECT_EQ("Blah", tp.p("name").as<std::string>());

  // Unnamed Ellipsis Dimension
  tp = ndt::make_ellipsis_dim(ndt::make_type<int>());
  EXPECT_EQ(ellipsis_dim_id, tp.get_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ(1u, tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  et = tp.extended<ndt::ellipsis_dim_type>();
  EXPECT_TRUE(et->get_name().empty());
  EXPECT_EQ(ndt::make_type<int>(), et->get_element_type());
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
  EXPECT_EQ("... * int32", tp.str());
  // Construction from empty string is ok
  EXPECT_EQ(tp, ndt::make_ellipsis_dim("", ndt::make_type<int>()));

  // Dynamic type properties
  //  Todo: Fix this
  //  EXPECT_TRUE(tp.p("name").is_null());

  // The ellipsis name must start with a capital
  // and look like an identifier
  EXPECT_THROW(ndt::make_ellipsis_dim("blah", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::make_ellipsis_dim("T ", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::make_ellipsis_dim("123", ndt::make_type<int>()), type_error);
  EXPECT_THROW(ndt::make_ellipsis_dim("Two+", ndt::make_type<int>()), type_error);
}

TEST(SymbolicTypes, AnySym) {
  ndt::type tp;

  tp = ndt::any_kind_type::make();
  EXPECT_EQ(any_kind_id, tp.get_id());
  EXPECT_EQ("Any", tp.str());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  // The "Any" type's variadic-ness should propagate through dimension types
  tp = ndt::type("3 * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("var * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("Fixed * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("?3 * Any");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("pointer[3 * Any]");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  // The "Any" type's variadic-ness should not propagate through struct/tuple
  // types
  tp = ndt::type("(Any, Any)");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("{x: Any, y: Any}");
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_FALSE(tp.is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(SymbolicTypes, TypeTypeWithPattern) {
  ndt::type tp;

  tp = ndt::make_type<ndt::type_type>(ndt::type("N * int32"));
  EXPECT_FALSE(tp.is_symbolic());
  EXPECT_EQ(ndt::type("N * int32"), tp.extended<ndt::type_type>()->get_pattern_type());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  // The pattern type must be symbolic
  EXPECT_THROW(ndt::type("type | 4 * int32"), type_error);
}

TEST(SymbolicTypes, VariadicTuple) {
  ndt::type tp;

  tp = ndt::make_type<ndt::tuple_type>({ndt::make_type<int>(), ndt::make_type<float>()}, true);
  EXPECT_EQ(tuple_id, tp.get_id());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.extended<ndt::tuple_type>()->is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("(type, int32, T, ...)");
  EXPECT_EQ(tuple_id, tp.get_id());
  EXPECT_JSON_EQ_ARR("[\"type\", \"int32\", \"T\"]", tp.p<std::vector<ndt::type>>("field_types"));
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.extended<ndt::tuple_type>()->is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(SymbolicTypes, VariadicStruct) {
  ndt::type tp;

  tp = ndt::make_type<ndt::struct_type>({{ndt::make_type<int>(), "x"}, {ndt::make_type<float>(), "y"}}, true);
  EXPECT_EQ(struct_id, tp.get_id());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.extended<ndt::struct_type>()->is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("{x: type, y: int32, z: T, ...}");
  EXPECT_JSON_EQ_ARR("[\"type\", \"int32\", \"T\"]", tp.p<std::vector<ndt::type>>("field_types"));
  EXPECT_JSON_EQ_ARR("[\"x\", \"y\", \"z\"]", tp.p<std::vector<std::string>>("field_names"));
  EXPECT_EQ(struct_id, tp.get_id());
  EXPECT_TRUE(tp.is_symbolic());
  EXPECT_TRUE(tp.extended<ndt::struct_type>()->is_variadic());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(SymbolicTypes, VariadicCallable) {
  ndt::type tp;

  tp = ndt::type("(int32, ...) -> float32");
  EXPECT_JSON_EQ_ARR("[\"int32\"]", tp.p<std::vector<ndt::type>>("pos_types"));
  EXPECT_TRUE(tp.extended<ndt::callable_type>()->has_variadic_arguments());
  EXPECT_EQ(ndt::type("(int32, ...)"), tp.extended<ndt::callable_type>()->get_pos_tuple());
  EXPECT_EQ(tp, ndt::type(tp.str()));

  tp = ndt::type("(int32, ..., shape: 3 * intptr) -> float32");
  EXPECT_JSON_EQ_ARR("[\"int32\"]", tp.p<std::vector<ndt::type>>("pos_types"));
  EXPECT_JSON_EQ_ARR("[\"3 * intptr\"]", tp.p<std::vector<ndt::type>>("kwd_types"));
  EXPECT_EQ(ndt::type("(int32, ...)"), tp.extended<ndt::callable_type>()->get_pos_tuple());
  EXPECT_EQ(ndt::type("{shape: 3 * intptr}"), tp.extended<ndt::callable_type>()->get_kwd_struct());
  EXPECT_EQ(tp, ndt::type(tp.str()));
}
