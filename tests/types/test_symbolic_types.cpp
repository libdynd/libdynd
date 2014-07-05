//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/types/funcproto_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(SymbolicTypes, CreateFuncProto) {
    ndt::type tp;
    const funcproto_type *fpt;

    // Function prototype from C++ template parameter
    tp = ndt::make_funcproto<int64_t (float, int32_t, double)>();
    EXPECT_EQ(funcproto_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    fpt = tp.tcast<funcproto_type>();
    ASSERT_EQ(3, fpt->get_param_count());
    EXPECT_EQ(ndt::make_type<float>(), fpt->get_param_type(0));
    EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_param_type(1));
    EXPECT_EQ(ndt::make_type<double>(), fpt->get_param_type(2));
    EXPECT_EQ(ndt::make_type<int64_t>(), fpt->get_return_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("(float32, int32, float64) -> int64", tp.str());

    // Dynamic type properties
    EXPECT_EQ(ndt::make_type<int64_t>(), tp.p("return_type").as<ndt::type>());
    nd::array ptp = tp.p("param_types");
    EXPECT_EQ(ndt::type("strided * type"), ptp.get_type());
    ASSERT_EQ(3, ptp.get_dim_size());
    EXPECT_EQ(ndt::make_type<float>(), ptp(0).as<ndt::type>());
    EXPECT_EQ(ndt::make_type<int32_t>(), ptp(1).as<ndt::type>());
    EXPECT_EQ(ndt::make_type<double>(), ptp(2).as<ndt::type>());

    // Exercise a few different variations
    tp = ndt::make_funcproto<int8_t ()>();
    fpt = tp.tcast<funcproto_type>();
    ASSERT_EQ(0, fpt->get_param_count());
    EXPECT_EQ(ndt::make_type<int8_t>(), fpt->get_return_type());

    tp = ndt::make_funcproto<int16_t (int32_t)>();
    fpt = tp.tcast<funcproto_type>();
    ASSERT_EQ(1, fpt->get_param_count());
    EXPECT_EQ(ndt::make_type<int16_t>(), fpt->get_return_type());
    EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_param_type(0));

    tp = ndt::make_funcproto<int16_t (int32_t, int64_t)>();
    fpt = tp.tcast<funcproto_type>();
    ASSERT_EQ(2, fpt->get_param_count());
    EXPECT_EQ(ndt::make_type<int16_t>(), fpt->get_return_type());
    EXPECT_EQ(ndt::make_type<int32_t>(), fpt->get_param_type(0));
    EXPECT_EQ(ndt::make_type<int64_t>(), fpt->get_param_type(1));
}

TEST(SymbolicTypes, CreateTypeVar) {
    ndt::type tp;
    const typevar_type *tvt;

    // Simple TypeVar
    tp = ndt::make_typevar("Blah");
    EXPECT_EQ(typevar_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    tvt = tp.tcast<typevar_type>();
    EXPECT_EQ("Blah", tvt->get_name_str());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("Blah", tp.str());

    // Dynamic type properties
    EXPECT_EQ("Blah", tp.p("name").as<std::string>());

    // The typevar name must start with a capital
    // and look like an identifier
    EXPECT_THROW(ndt::make_typevar(""), type_error);
    EXPECT_THROW(ndt::make_typevar("blah"), type_error);
    EXPECT_THROW(ndt::make_typevar("T "), type_error);
    EXPECT_THROW(ndt::make_typevar("123"), type_error);
    EXPECT_THROW(ndt::make_typevar("Two+"), type_error);
}
 
TEST(SymbolicTypes, CreateTypeVarDim) {
    ndt::type tp;
    const typevar_dim_type *tvt;

    // Simple Dimension TypeVar
    tp = ndt::make_typevar_dim("Blah", ndt::make_type<int>());
    EXPECT_EQ(typevar_dim_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    tvt = tp.tcast<typevar_dim_type>();
    EXPECT_EQ("Blah", tvt->get_name_str());
    EXPECT_EQ(ndt::make_type<int>(), tvt->get_element_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("Blah * int32", tp.str());

    // Dynamic type properties
    EXPECT_EQ("Blah", tp.p("name").as<std::string>());

    // The typevar name must start with a capital
    // and look like an identifier
    EXPECT_THROW(ndt::make_typevar_dim("", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("blah", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("T ", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("123", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_typevar_dim("Two+", ndt::make_type<int>()),
                 type_error);
}
 
TEST(SymbolicTypes, CreateEllipsisDim) {
    ndt::type tp;
    const ellipsis_dim_type *et;

    // Named Ellipsis Dimension
    tp = ndt::make_ellipsis_dim("Blah", ndt::make_type<int>());
    EXPECT_EQ(ellipsis_dim_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    EXPECT_TRUE(tp.is_symbolic());
    et = tp.tcast<ellipsis_dim_type>();
    EXPECT_EQ("Blah", et->get_name_str());
    EXPECT_EQ(ndt::make_type<int>(), et->get_element_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("Blah... * int32", tp.str());

    // Dynamic type properties
    EXPECT_EQ("Blah", tp.p("name").as<std::string>());

    // Unnamed Ellipsis Dimension
    tp = ndt::make_ellipsis_dim(ndt::make_type<int>());
    EXPECT_EQ(ellipsis_dim_type_id, tp.get_type_id());
    EXPECT_EQ(0u, tp.get_data_size());
    EXPECT_EQ(1u, tp.get_data_alignment());
    EXPECT_FALSE(tp.is_pod());
    EXPECT_TRUE(tp.is_symbolic());
    et = tp.tcast<ellipsis_dim_type>();
    EXPECT_TRUE(et->get_name().is_null());
    EXPECT_EQ(ndt::make_type<int>(), et->get_element_type());
    // Roundtripping through a string
    EXPECT_EQ(tp, ndt::type(tp.str()));
    EXPECT_EQ("... * int32", tp.str());
    // Construction from empty string is ok
    EXPECT_EQ(tp, ndt::make_ellipsis_dim("", ndt::make_type<int>()));

    // Dynamic type properties
    EXPECT_TRUE(tp.p("name").is_null());

    // The ellipsis name must start with a capital
    // and look like an identifier
    EXPECT_THROW(ndt::make_ellipsis_dim("blah", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_ellipsis_dim("T ", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_ellipsis_dim("123", ndt::make_type<int>()),
                 type_error);
    EXPECT_THROW(ndt::make_ellipsis_dim("Two+", ndt::make_type<int>()),
                 type_error);
}
