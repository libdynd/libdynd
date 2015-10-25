//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/func/callable.hpp>
#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/c_contiguous_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeParser, Basic)
{
  EXPECT_EQ(ndt::type::make<void>(), ndt::type("void"));
  EXPECT_EQ(ndt::type::make<bool1>(), ndt::type("bool"));
  EXPECT_EQ(ndt::type::make<int8>(), ndt::type("int8"));
  EXPECT_EQ(ndt::type::make<int16>(), ndt::type("int16"));
  EXPECT_EQ(ndt::type::make<int32>(), ndt::type("int32"));
  EXPECT_EQ(ndt::type::make<int64>(), ndt::type("int64"));
  EXPECT_EQ(ndt::type::make<int128>(), ndt::type("int128"));
  EXPECT_EQ(ndt::type::make<intptr_t>(), ndt::type("intptr"));
  EXPECT_EQ(ndt::type::make<uint8>(), ndt::type("uint8"));
  EXPECT_EQ(ndt::type::make<uint16>(), ndt::type("uint16"));
  EXPECT_EQ(ndt::type::make<uint32>(), ndt::type("uint32"));
  EXPECT_EQ(ndt::type::make<uint64>(), ndt::type("uint64"));
  EXPECT_EQ(ndt::type::make<uint128>(), ndt::type("uint128"));
  EXPECT_EQ(ndt::type::make<uintptr_t>(), ndt::type("uintptr"));
  EXPECT_EQ(ndt::type::make<float16>(), ndt::type("float16"));
  EXPECT_EQ(ndt::type::make<float32>(), ndt::type("float32"));
  EXPECT_EQ(ndt::type::make<float64>(), ndt::type("float64"));
  EXPECT_EQ(ndt::type::make<float128>(), ndt::type("float128"));
  EXPECT_EQ(ndt::type::make<dynd::complex<float>>(), ndt::type("complex64"));
  EXPECT_EQ(ndt::type::make<dynd::complex<double>>(), ndt::type("complex128"));
  EXPECT_EQ(ndt::type::make<dynd::complex<float>>(), ndt::type("complex[float32]"));
  EXPECT_EQ(ndt::type::make<dynd::complex<double>>(), ndt::type("complex[float64]"));
  EXPECT_EQ(ndt::date_type::make(), ndt::type("date"));
  EXPECT_EQ(ndt::time_type::make(), ndt::type("time"));
  EXPECT_EQ(ndt::datetime_type::make(), ndt::type("datetime"));
  // Aliases for some of the above types
  EXPECT_EQ(ndt::type::make<int32_t>(), ndt::type("int"));
  EXPECT_EQ(ndt::type::make<double>(), ndt::type("real"));
  EXPECT_EQ(ndt::type::make<dynd::complex<double>>(), ndt::type("complex"));
}

TEST(DataShapeParser, BasicThrow)
{
  EXPECT_THROW(ndt::type("boot"), runtime_error);
  EXPECT_THROW(ndt::type("int33"), runtime_error);
}

TEST(DataShapeParser, StringAtoms)
{
  // Default string
  EXPECT_EQ(ndt::string_type::make(), ndt::type("string"));
  // String with size
  EXPECT_EQ(ndt::fixed_string_type::make(1, string_encoding_utf_8), ndt::type("fixed_string[1]"));
  EXPECT_EQ(ndt::fixed_string_type::make(100, string_encoding_utf_8), ndt::type("fixed_string[100]"));
  // String with size and encoding
  EXPECT_EQ(ndt::fixed_string_type::make(1, string_encoding_ascii), ndt::type("fixed_string[1, 'A']"));
  EXPECT_EQ(ndt::fixed_string_type::make(10, string_encoding_utf_8), ndt::type("fixed_string[10, 'U8']"));
  EXPECT_EQ(ndt::fixed_string_type::make(1000, string_encoding_utf_16), ndt::type("fixed_string[1000,'U16']"));

  // String with encoding -- change to fixed_string or delete?
  /*
    EXPECT_EQ(ndt::string_type::make(string_encoding_ascii),
              ndt::type("string['A']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_ascii),
              ndt::type("string['ascii']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_8),
              ndt::type("string['U8']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_8),
              ndt::type("string['utf8']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_8),
              ndt::type("string['utf-8']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_16),
              ndt::type("string['U16']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_16),
              ndt::type("string['utf16']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_16),
              ndt::type("string['utf-16']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_32),
              ndt::type("string['U32']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_32),
              ndt::type("string['utf32']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_utf_32),
              ndt::type("string['utf-32']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_ucs_2),
              ndt::type("string['ucs2']"));
    EXPECT_EQ(ndt::string_type::make(string_encoding_ucs_2),
              ndt::type("string['ucs-2']"));
  */
}

TEST(DataShapeParser, Unaligned)
{
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::type::make<bool1>()), ndt::type("Fixed * unaligned[bool]"));
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::make_unaligned(ndt::type::make<float>()), 2),
            ndt::type("Fixed * Fixed * unaligned[float32]"));
  EXPECT_EQ(ndt::struct_type::make({"x", "y"}, {ndt::make_unaligned(ndt::type::make<int32_t>()),
                                                ndt::make_unaligned(ndt::type::make<int64_t>())}),
            ndt::type("{x : unaligned[int32], y : unaligned[int64]}"));
}

TEST(DataShapeParser, Option)
{
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::option_type::make(ndt::type::make<bool1>())),
            ndt::type("Fixed * option[bool]"));
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::option_type::make(ndt::type::make<bool1>())), ndt::type("Fixed * ?bool"));
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::option_type::make(ndt::make_fixed_dim_kind(ndt::type::make<float>()))),
            ndt::type("Fixed * option[Fixed * float32]"));
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::option_type::make(ndt::make_fixed_dim_kind(ndt::type::make<float>()))),
            ndt::type("Fixed * ?Fixed * float32"));
  EXPECT_EQ(ndt::struct_type::make({"x", "y"}, {ndt::option_type::make(ndt::type::make<int32_t>()),
                                                ndt::option_type::make(ndt::type::make<int64_t>())}),
            ndt::type("{x : ?int32, y : ?int64}"));
}

TEST(DataShapeParser, StridedDim)
{
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::type::make<bool1>()), ndt::type("Fixed * bool"));
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::type::make<float>(), 2), ndt::type("Fixed * Fixed * float32"));
  EXPECT_EQ(ndt::type("Fixed * Fixed * float32"), ndt::type("Fixed**2 * float32"));
}

TEST(DataShapeParser, FixedDim)
{
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::type::make<bool1>()), ndt::type("3 * bool"));
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::type::make<bool1>()), ndt::type("fixed[3] * bool"));
  EXPECT_EQ(ndt::make_fixed_dim(4, ndt::make_fixed_dim(3, ndt::type::make<float>())), ndt::type("4 * 3 * float32"));
  EXPECT_EQ(ndt::type("7 * 7 * 7 * float32"), ndt::type("7**3 * float32"));
}

TEST(DataShapeParser, CContiguous)
{
  EXPECT_EQ(ndt::c_contiguous_type::make(ndt::make_fixed_dim(3, ndt::type::make<bool1>())), ndt::type("C[3 * bool]"));
  EXPECT_EQ(ndt::c_contiguous_type::make(make_fixed_dim(4, ndt::make_fixed_dim(3, ndt::type::make<float>()))),
            ndt::type("C[4 * 3 * float32]"));
}

TEST(DataShapeParser, VarDim)
{
  EXPECT_EQ(ndt::var_dim_type::make(ndt::type::make<bool1>()), ndt::type("var * bool"));
  EXPECT_EQ(ndt::var_dim_type::make(ndt::var_dim_type::make(ndt::type::make<float>())),
            ndt::type("var * var * float32"));
  EXPECT_EQ(ndt::type("var * var * var * var * var * float32"), ndt::type("var**5 * float32"));
}

TEST(DataShapeParser, StridedFixedDim)
{
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::make_fixed_dim(3, ndt::type::make<float>())),
            ndt::type("Fixed * 3 * float32"));
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_fixed_dim_kind(ndt::type::make<float>())),
            ndt::type("3 * Fixed * float32"));
}

TEST(DataShapeParser, StridedVarFixedDim)
{
  EXPECT_EQ(ndt::make_fixed_dim_kind(ndt::var_dim_type::make(ndt::make_fixed_dim(3, ndt::type::make<float>()))),
            ndt::type("Fixed * var * 3 * float32"));
}

TEST(DataShapeParser, TypeVarConstructed)
{
  EXPECT_EQ(ndt::typevar_constructed_type::make("T", ndt::type("int32")), ndt::type("T[int32]"));
  EXPECT_EQ(ndt::typevar_constructed_type::make("T", ndt::type("10 * A")), ndt::type("T[10 * A]"));
  EXPECT_EQ(ndt::typevar_constructed_type::make("T", ndt::type("Dims... * int32")), ndt::type("T[Dims... * int32]"));
}

TEST(DataShapeParser, RecordOneField)
{
  EXPECT_EQ(ndt::struct_type::make({"val"}, {ndt::type::make<float>()}), ndt::type("{ val : float32 }"));
  EXPECT_EQ(ndt::struct_type::make({"val"}, {ndt::type::make<float>()}), ndt::type("{ val : float32 , }"));
}

TEST(DataShapeParser, RecordTwoFields)
{
  EXPECT_EQ(ndt::struct_type::make({"val", "id"}, {ndt::type::make<float>(), ndt::type::make<int64_t>()}),
            ndt::type("{\n"
                      "    val: float32,\n"
                      "    id: int64\n"
                      "}\n"));
  EXPECT_EQ(ndt::struct_type::make({"val", "id"}, {ndt::type::make<float>(), ndt::type::make<int64_t>()}),
            ndt::type("{\n"
                      "    val: float32,\n"
                      "    id: int64,\n"
                      "}\n"));
}

TEST(DataShapeParser, Callable)
{
  ndt::type tp;

  EXPECT_EQ(ndt::type::make<int(int, int)>(), ndt::type("(int, int) -> int"));
  EXPECT_EQ(ndt::type::make<int(int, int)>("x"), ndt::type("(int, x: int) -> int"));
  EXPECT_EQ(ndt::type::make<int(int, int)>("x", "y"), ndt::type("(x: int, y: int) -> int"));

  tp = ndt::type("(N * S, func: (M * S) -> T) -> N * T");
  EXPECT_JSON_EQ_ARR("[\"N * S\"]", tp.p("pos_types"));
  EXPECT_JSON_EQ_ARR("[\"(M * S) -> T\"]", tp.p("kwd_types"));
  EXPECT_JSON_EQ_ARR("[\"func\"]", tp.p("kwd_names"));
  EXPECT_JSON_EQ_ARR("\"N * T\"", tp.p("return_type"));
}

TEST(DataShapeParser, ErrorBasic)
{
  try
  {
    ndt::type("float65");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 1") != std::string::npos);
    EXPECT_TRUE(msg.find("unrecognized data type") != std::string::npos);
  }
  try
  {
    ndt::type("float64+");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 8") != std::string::npos);
    EXPECT_TRUE(msg.find("unexpected token") != std::string::npos);
  }
  try
  {
    ndt::type("3 * int32 * float64");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 5") != std::string::npos);
    EXPECT_TRUE(msg.find("unrecognized dimension type") != std::string::npos);
  }
}

TEST(DataShapeParser, ErrorString)
{
  try
  {
    ndt::type("fixed_string[");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 14") != std::string::npos);
    EXPECT_TRUE(msg.find("expected a size integer") != std::string::npos);
  }
  try
  {
    ndt::type("fixed_string[0]");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 14") != std::string::npos);
    EXPECT_TRUE(msg.find("string size cannot be zero") != std::string::npos);
  }
  try
  {
    ndt::type("fixed_string[3,'U8',10]");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 20") != std::string::npos);
    EXPECT_TRUE(msg.find("expected closing ']'") != std::string::npos);
  }
  try
  {
    ndt::type("fixed_string[3,3]");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 1, column 16") != std::string::npos);
    EXPECT_TRUE(msg.find("expected a string encoding") != std::string::npos);
  }
}

TEST(DataShapeParser, ErrorRecord)
{
  try
  {
    ndt::type("{\n"
              "   id: int64\n"
              "   name: string\n"
              "   amount: invalidtype\n"
              "}\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 2, column 13") != std::string::npos);
    EXPECT_TRUE(msg.find("expected ',' or '}'") != std::string::npos);
  }
  try
  {
    ndt::type("{\n"
              "   id: int64,\n"
              "   name: string,\n"
              "   amount: invalidtype;\n"
              "}\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 4, column 12") != std::string::npos);
    EXPECT_TRUE(msg.find("unrecognized data type") != std::string::npos);
  }
  try
  {
    ndt::type("{\n"
              "   id: int64,\n"
              "   name: string,\n"
              "   amount: %,\n"
              "}\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 4, column 11") != std::string::npos);
    EXPECT_TRUE(msg.find("expected a data type") != std::string::npos);
  }
  try
  {
    ndt::type("{\n"
              "   id: int64,\n"
              "   name: string,\n"
              "   amount+ float32,\n"
              "}\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 4, column 10") != std::string::npos);
    EXPECT_TRUE(msg.find("expected ':'") != std::string::npos);
  }
  try
  {
    ndt::type("{\n"
              "   id: int64,\n"
              "   name: (3 * string,\n"
              "   amount: float32,\n"
              "}\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    // The name field works as a funcproto until it hits the '}' token
    EXPECT_TRUE(msg.find("line 4, column 20") != std::string::npos);
    EXPECT_TRUE(msg.find("expected a kwd arg in callable prototype") != std::string::npos);
  }
}

TEST(DataShapeParser, ErrorTypeAlias)
{
  try
  {
    ndt::type("\n"
              "type 33 = int32\n"
              "2, int32\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 2, column 5") != std::string::npos);
    EXPECT_TRUE(msg.find("expected an identifier") != std::string::npos);
  }
  try
  {
    ndt::type("\n"
              "type MyInt - int32\n"
              "2, MyInt\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 2, column 11") != std::string::npos);
    EXPECT_TRUE(msg.find("expected an '='") != std::string::npos);
  }
  try
  {
    ndt::type("\n"
              "type MyInt = &\n"
              "2 * MyInt\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 2, column 13") != std::string::npos);
    EXPECT_TRUE(msg.find("expected a data type") != std::string::npos);
  }
  try
  {
    ndt::type("\n"
              "type int32 = int64\n"
              "2 * int32\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 2, column 6") != std::string::npos);
    EXPECT_TRUE(msg.find("cannot redefine") != std::string::npos);
  }
  try
  {
    ndt::type("\n"
              "type MyInt = int64\n"
              "type MyInt = int32\n"
              "2 * MyInt\n");
    EXPECT_TRUE(false);
  }
  catch (const runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("line 3, column 6") != std::string::npos);
    EXPECT_TRUE(msg.find("type name already defined") != std::string::npos);
  }
}

TEST(DataShapeParser, KivaLoanDataShape)
{
  const char *klds = "type KivaLoan = {\n"
                     "    id: int64,\n"
                     "    name: string,\n"
                     "    description: {\n"
                     "        languages: var * fixed_string[2],\n"
                     "    #    texts: map(fixed_string[2], string),\n"
                     "    },\n"
                     "    status: string, # LoanStatusType,\n"
                     "    funded_amount: float64,\n"
                     "    #basket_amount: Option(float64),\n"
                     "    paid_amount: float64,\n"
                     "    image: {\n"
                     "        id: int64,\n"
                     "        template_id: int64,\n"
                     "    },\n"
                     "    #video: Option({\n"
                     "    #    id: int64,\n"
                     "    #    youtube_id: string,\n"
                     "    #}),\n"
                     "    activity: string,\n"
                     "    sector: string,\n"
                     "    use: string,\n"
                     "    # For 'delinquent', saw values \"null\" and \"true\" in brief "
                     "search, map null -> false on import?\n"
                     "    delinquent: bool,\n"
                     "    location: {\n"
                     "        country_code: fixed_string[2],\n"
                     "        country: string,\n"
                     "        town: string,\n"
                     "        geo: {\n"
                     "            level: string, # GeoLevelType\n"
                     "            pairs: string, # latlong\n"
                     "            type: string, # GeoTypeType\n"
                     "        }\n"
                     "    },\n"
                     "    partner_id: int64,\n"
                     "    #posted_date: datetime<seconds>,\n"
                     "    #planned_expiration_date: Option(datetime<seconds>),\n"
                     "    loan_amount: float64,\n"
                     "    #currency_exchange_loss_amount: Option(float64),\n"
                     "    borrowers: var * {\n"
                     "        first_name: string,\n"
                     "        last_name: string,\n"
                     "        gender: fixed_string[2], # GenderType\n"
                     "        pictured: bool,\n"
                     "    },\n"
                     "    terms: {\n"
                     "    #    disbursal_date: datetime<seconds>,\n"
                     "    #    disbursal_currency: Option(string),\n"
                     "        disbursal_amount: float64,\n"
                     "        loan_amount: float64,\n"
                     "        local_payments: var * {\n"
                     "    #        due_date: datetime<seconds>,\n"
                     "            amount: float64,\n"
                     "        },\n"
                     "        scheduled_payments: var * {\n"
                     "    #        due_date: datetime<seconds>,\n"
                     "            amount: float64,\n"
                     "        },\n"
                     "        loss_liability: {\n"
                     "    #        nonpayment: Categorical(string, [\"lender\", "
                     "\"partner\"]),\n"
                     "            currency_exchange: string,\n"
                     "    #        currency_exchange_coverage_rate: Option(float64),\n"
                     "        }\n"
                     "    },\n"
                     "    payments: var * {\n"
                     "        amount: float64,\n"
                     "        local_amount: float64,\n"
                     "    #    processed_date: datetime<seconds>,\n"
                     "    #    settlement_date: datetime<seconds>,\n"
                     "        rounded_local_amount: float64,\n"
                     "        currency_exchange_loss_amount: float64,\n"
                     "        payment_id: int64,\n"
                     "        comment: string,\n"
                     "    },\n"
                     "    #funded_date: datetime<seconds>,\n"
                     "    #paid_date: datetime<seconds>,\n"
                     "    journal_totals: {\n"
                     "        entries: int64,\n"
                     "        bulkEntries: int64,\n"
                     "    }\n"
                     "}\n";
  ndt::type d = ndt::type(klds);
  EXPECT_EQ(struct_type_id, d.get_type_id());
}

TEST(DataShapeParser, SpecialCharacterFields)
{
  /*
    This suite is designed to test the handling of special characters in struct
    field names
      when the datashape parser is turning a string into types.
    Any field name that has special characters should be enclosed in quotes in
    order to be handled properly.
    Special characters may include:
    - space
    - slashes, comma, :
    - unicode, \n, \r, \t, etc.
    - forwardslash?
    - parens?
    - date and time?
    - numbers
   */

  // p_dt == primary_datatype, s_dt == secondary_datatype
  ndt::type p_dt, s_dt;
  // ss == secondary_stream. This variable will be used to turn an existing
  // datatype into a string. That string will later be re-parsed and
  // compared against the primary datatype for equality.
  stringstream ss;

  // Test basic occurrences of spaces in field names.
  // Potential special cases are:
  // beginning of name
  // end of name
  // middle of name
  // multiple spaces at beginning, middle, or end
  p_dt = ndt::type("{ 'sepal length' : float64 }");
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"sepal length"}, {ndt::type::make<double>()}));
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ ' sepal length' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({" sepal length"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ 'sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"sepal length "}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ ' sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({" sepal length "}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '   sepal         length   ' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"   sepal         length   "}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // parens test
  // Potential special cases are:
  // beginning of name
  // end of name
  // middle of name
  // unmatched parens
  // multiple parens
  p_dt = ndt::type("{ 'unique key(foo)' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"unique key(foo)"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '(foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"(foo unique key"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '(foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"(foo unique key)"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ 'foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"foo unique key)"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '( foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"( foo unique key"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '((((foouniquekey))())((' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"((((foouniquekey))())(("}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // Quote tests. These can get a little tricky since the quotes must be
  // escaped, and the backslashes may have to be escaped as well
  p_dt = ndt::type("{ 'sepal \"\"\" length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"sepal \"\"\" length", "sepal width"},
                                         {ndt::type::make<double>(), ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"sepal ''' length", "sepal width"},
                                         {ndt::type::make<double>(), ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ ' sepal \\'\\'\\' length ' : float64, 'sepal "
                   "\" \\\" \\\\\\' width ' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({" sepal ''' length ", "sepal \" \" \\' width "},
                                         {ndt::type::make<double>(), ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"sepal ''' length", "sepal width"},
                                         {ndt::type::make<double>(), ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // A space is a valid field name. Perhaps it should not be.
  p_dt = ndt::type("{ ' ' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({" "}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // NOTE: According to the grammar, bare backslashes are not allowed as field
  // names.
  //      Also not allowed are bare \n, \r, ", or ' characters
  //      So, that means that the following tests should fail.
  // This one hurts my brain. When the string is parsed, each \\ gets turned
  // into a single \.
  // The final \\ is interpreted as a single \ in the name.
  // But, when converting the datatype back to a string, the single \ is escaped
  // as a \\ by print_escaped_utf8_string() via
  //  struct_type::print_type()
  /*To see that kind of code in action, try this:
    char char_backslash=char(92);
    const char *backslash=&c_bs;
    print_escaped_utf8_string(std::cout, backslash, backslash+1, true);
   */
  p_dt = ndt::type("{ '\\\\' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\\"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '\n' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\n"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '\r' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\r"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '\\\'' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\'"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '\"' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\""}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // Testing forward slashes
  p_dt = ndt::type("{ '/usr/bin' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"/usr/bin"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{ '\\\\abcdef' : float64 }");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"\\abcdef"}, {ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt = ndt::type("{'a field with spaces and \" \\\'': int}");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"a field with spaces and \" \'"}, {ndt::type::make<int>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // Testing field names as numbers
  p_dt = ndt::type("{ '1234' : float64, '-1234':float64,'500=1234':float64 }");
  EXPECT_EQ(p_dt,
            ndt::struct_type::make({"1234", "-1234", "500=1234"},
                                   {ndt::type::make<double>(), ndt::type::make<double>(), ndt::type::make<double>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  // Other characters or strings that the parser may have trouble with include
  // {,},[,],:,comma,...,>,=,*,<,>,-
  p_dt = ndt::type("{'field name[100]': int, 'field name [x]': int, "
                   "'field : {name-=foo} [0*<>...,]': int}");
  EXPECT_EQ(p_dt, ndt::struct_type::make({"field name[100]", "field name [x]", "field : {name-=foo} [0*<>...,]"},
                                         {ndt::type::make<int>(), ndt::type::make<int>(), ndt::type::make<int>()}));
  ss << p_dt;
  s_dt = ndt::type(ss.str());
  ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  /*Failing tests:
    If special names aren't enclosed in quotes, the parser should throw an
    exception.
   */
  try
  {
    ndt::type d = ndt::type("{ bad name : int }");
    EXPECT_TRUE(false);
  }
  catch (std::runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 6") != std::string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != std::string::npos);
  }

  try
  {
    ndt::type d = ndt::type("{ name(foo) : int }");
    EXPECT_TRUE(false);
  }
  catch (std::runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 7") != std::string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != std::string::npos);
  }

  try
  {
    ndt::type d = ndt::type("{ name\' : int }");
    EXPECT_TRUE(false);
  }
  catch (std::runtime_error &e)
  {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 7") != std::string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != std::string::npos);
  }
}

static std::string to_str(const nd::array &a)
{
  stringstream ss;
  ss << a;
  return ss.str();
}

TEST(DataShapeParser, TypeConstructorArgs)
{
  nd::array a, b;
  EXPECT_JSON_EQ_ARR("[[], {}]", parse_type_constr_args("[]"));
  EXPECT_JSON_EQ_ARR("[[3, -10, \"test\"], {}]", parse_type_constr_args("[3, -10, \"test\"]"));
  // The following currently raises a broadcast error in equality testing
  b = parse_type_constr_args("[3, -10, \"test\", first:int32, second:[1,2,3]]");
  a = parse_json(b.get_type(), "[[3, -10, \"test\"], [\"int32\", [1,2,3]]]");
  EXPECT_EQ(to_str(a), to_str(b));
  EXPECT_JSON_EQ_ARR("[[3, -10, \"test\"], [\"int32\", \"blah\"]]",
                     parse_type_constr_args("[3, -10, \"test\", first:int32, second:\"blah\"]"));
  // The following currently raises a broadcast error in equality testing
  b = parse_type_constr_args("[[1, 2, 3], [2 * int32, float32, 3 * int8], [\"x\", \"yz\"]]");
  a = parse_json(b.get_type(), "[[[1, 2, 3], [\"2 * int32\", \"float32\", \"3 * int8\"], [\"x\", \"yz\"]], []]");
  EXPECT_EQ(to_str(a), to_str(b));
  // The following currently raises a broadcast error in equality testing
  b = parse_type_constr_args("[[1, 2, 3], x: [2 * int32, float32, 3 * int8], y: [\"x\", \"yz\"]]");
  a = parse_json(b.get_type(), "[[[1, 2, 3]], [[\"2 * int32\", \"float32\", \"3 * int8\"], [\"x\", \"yz\"]]]");
  EXPECT_EQ(to_str(a), to_str(b));
}
