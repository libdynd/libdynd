//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/fixed_dimsym_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cfixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/func/callable.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeParser, Basic) {
    EXPECT_EQ(ndt::make_type<void>(), type_from_datashape("void"));
    EXPECT_EQ(ndt::make_type<dynd_bool>(), type_from_datashape("bool"));
    EXPECT_EQ(ndt::make_type<int8_t>(), type_from_datashape("int8"));
    EXPECT_EQ(ndt::make_type<int16_t>(), type_from_datashape("int16"));
    EXPECT_EQ(ndt::make_type<int32_t>(), type_from_datashape("int32"));
    EXPECT_EQ(ndt::make_type<int64_t>(), type_from_datashape("int64"));
    EXPECT_EQ(ndt::make_type<dynd_int128>(), type_from_datashape("int128"));
    EXPECT_EQ(ndt::make_type<intptr_t>(), type_from_datashape("intptr"));
    EXPECT_EQ(ndt::make_type<uint8_t>(), type_from_datashape("uint8"));
    EXPECT_EQ(ndt::make_type<uint16_t>(), type_from_datashape("uint16"));
    EXPECT_EQ(ndt::make_type<uint32_t>(), type_from_datashape("uint32"));
    EXPECT_EQ(ndt::make_type<uint64_t>(), type_from_datashape("uint64"));
    EXPECT_EQ(ndt::make_type<dynd_uint128>(), type_from_datashape("uint128"));
    EXPECT_EQ(ndt::make_type<uintptr_t>(), type_from_datashape("uintptr"));
    EXPECT_EQ(ndt::make_type<dynd_float16>(), type_from_datashape("float16"));
    EXPECT_EQ(ndt::make_type<float>(), type_from_datashape("float32"));
    EXPECT_EQ(ndt::make_type<double>(), type_from_datashape("float64"));
    EXPECT_EQ(ndt::make_type<dynd_float128>(), type_from_datashape("float128"));
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(),
              type_from_datashape("complex64"));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(),
              type_from_datashape("complex128"));
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(),
              type_from_datashape("complex[float32]"));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(),
              type_from_datashape("complex[float64]"));
    EXPECT_EQ(ndt::make_json(), type_from_datashape("json"));
    EXPECT_EQ(ndt::make_date(), type_from_datashape("date"));
    // Aliases for some of the above types
    EXPECT_EQ(ndt::make_type<int32_t>(), type_from_datashape("int"));
    EXPECT_EQ(ndt::make_type<double>(), type_from_datashape("real"));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), type_from_datashape("complex"));
}

TEST(DataShapeParser, BasicThrow) {
    EXPECT_THROW(type_from_datashape("boot"), runtime_error);
    EXPECT_THROW(type_from_datashape("int33"), runtime_error);
}

TEST(DataShapeParser, StringAtoms) {
    // Default string
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8),
                    type_from_datashape("string"));
    // String with encoding
    EXPECT_EQ(ndt::make_string(string_encoding_ascii),
                    type_from_datashape("string['A']"));
    EXPECT_EQ(ndt::make_string(string_encoding_ascii),
                    type_from_datashape("string['ascii']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8),
                    type_from_datashape("string['U8']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8),
                    type_from_datashape("string['utf8']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_8),
                    type_from_datashape("string['utf-8']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_16),
                    type_from_datashape("string['U16']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_16),
                    type_from_datashape("string['utf16']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_16),
                    type_from_datashape("string['utf-16']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_32),
                    type_from_datashape("string['U32']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_32),
                    type_from_datashape("string['utf32']"));
    EXPECT_EQ(ndt::make_string(string_encoding_utf_32),
                    type_from_datashape("string['utf-32']"));
    EXPECT_EQ(ndt::make_string(string_encoding_ucs_2),
                    type_from_datashape("string['ucs2']"));
    EXPECT_EQ(ndt::make_string(string_encoding_ucs_2),
                    type_from_datashape("string['ucs-2']"));
    // String with size
    EXPECT_EQ(ndt::make_fixedstring(1, string_encoding_utf_8),
                    type_from_datashape("string[1]"));
    EXPECT_EQ(ndt::make_fixedstring(100, string_encoding_utf_8),
                    type_from_datashape("string[100]"));
    // String with size and encoding
    EXPECT_EQ(ndt::make_fixedstring(1, string_encoding_ascii),
                    type_from_datashape("string[1, 'A']"));
    EXPECT_EQ(ndt::make_fixedstring(10, string_encoding_utf_8),
                    type_from_datashape("string[10, 'U8']"));
    EXPECT_EQ(ndt::make_fixedstring(1000, string_encoding_utf_16),
                    type_from_datashape("string[1000,'U16']"));
}

TEST(DataShapeParser, Unaligned) {
    EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_type<dynd_bool>()),
                    type_from_datashape("fixed * unaligned[bool]"));
    EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_unaligned(ndt::make_type<float>()), 2),
                    type_from_datashape("fixed * fixed * unaligned[float32]"));
    EXPECT_EQ(ndt::make_struct(ndt::make_unaligned(ndt::make_type<int32_t>()), "x",
                                ndt::make_unaligned(ndt::make_type<int64_t>()), "y"),
                    type_from_datashape("{x : unaligned[int32], y : unaligned[int64]}"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_unaligned(ndt::make_type<int32_t>()), "x",
                                ndt::make_unaligned(ndt::make_type<int64_t>()), "y"),
                    type_from_datashape("c{x : unaligned[int32], y : unaligned[int64]}"));
}

TEST(DataShapeParser, Option)
{
  EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_option<dynd_bool>()),
            type_from_datashape("fixed * option[bool]"));
  EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_option<dynd_bool>()),
            type_from_datashape("fixed * ?bool"));
  EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_option(
                ndt::make_fixed_dimsym(ndt::make_type<float>()))),
            type_from_datashape("fixed * option[fixed * float32]"));
  EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_option(
                ndt::make_fixed_dimsym(ndt::make_type<float>()))),
            type_from_datashape("fixed * ?fixed * float32"));
  EXPECT_EQ(ndt::make_struct(ndt::make_option(ndt::make_type<int32_t>()), "x",
                             ndt::make_option(ndt::make_type<int64_t>()), "y"),
            type_from_datashape("{x : option[int32], y : option[int64]}"));
  EXPECT_EQ(ndt::make_cstruct(ndt::make_option(ndt::make_type<int32_t>()), "x",
                              ndt::make_option(ndt::make_type<int64_t>()), "y"),
            type_from_datashape("c{x : ?int32, y : ?int64}"));
}

TEST(DataShapeParser, StridedDim) {
    EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_type<dynd_bool>()),
                    type_from_datashape("fixed * bool"));
    EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_type<float>(), 2),
                    type_from_datashape("fixed * fixed * float32"));
    EXPECT_EQ(type_from_datashape("fixed * fixed * float32"),
                    type_from_datashape("fixed**2 * float32"));
}

TEST(DataShapeParser, FixedDim) {
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_type<dynd_bool>()),
              type_from_datashape("3 * bool"));
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_type<dynd_bool>()),
              type_from_datashape("fixed[3] * bool"));
    EXPECT_EQ(ndt::make_fixed_dim(
                  4, ndt::make_fixed_dim(3, ndt::make_type<float>())),
              type_from_datashape("4 * 3 * float32"));
    EXPECT_EQ(type_from_datashape("7 * 7 * 7 * float32"),
              type_from_datashape("7**3 * float32"));
}

TEST(DataShapeParser, CFixedDim) {
    EXPECT_EQ(ndt::make_cfixed_dim(3, ndt::make_type<dynd_bool>()),
              type_from_datashape("cfixed[3] * bool"));
    EXPECT_EQ(ndt::make_cfixed_dim(
                  4, ndt::make_cfixed_dim(3, ndt::make_type<float>())),
              type_from_datashape("cfixed[4] * cfixed[3] * float32"));
}

TEST(DataShapeParser, VarDim) {
    EXPECT_EQ(ndt::make_var_dim(ndt::make_type<dynd_bool>()),
              type_from_datashape("var * bool"));
    EXPECT_EQ(ndt::make_var_dim(ndt::make_var_dim(ndt::make_type<float>())),
              type_from_datashape("var * var * float32"));
    EXPECT_EQ(type_from_datashape("var * var * var * var * var * float32"),
              type_from_datashape("var**5 * float32"));
}

TEST(DataShapeParser, StridedFixedDim) {
    EXPECT_EQ(
        ndt::make_fixed_dimsym(ndt::make_fixed_dim(3, ndt::make_type<float>())),
        type_from_datashape("fixed * 3 * float32"));
    EXPECT_EQ(
        ndt::make_fixed_dim(3, ndt::make_fixed_dimsym(ndt::make_type<float>())),
        type_from_datashape("3 * fixed * float32"));
}

TEST(DataShapeParser, StridedVarFixedDim) {
    EXPECT_EQ(ndt::make_fixed_dimsym(ndt::make_var_dim(
                  ndt::make_fixed_dim(3, ndt::make_type<float>()))),
              type_from_datashape("fixed * var * 3 * float32"));
}

TEST(DataShapeParser, RecordOneField) {
    EXPECT_EQ(ndt::make_struct(ndt::make_type<float>(), "val"),
                    type_from_datashape("{ val : float32 }"));
    EXPECT_EQ(ndt::make_struct(ndt::make_type<float>(), "val"),
                    type_from_datashape("{ val : float32 , }"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val"),
                    type_from_datashape("c{ val : float32 }"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val"),
                    type_from_datashape("c{ val : float32 , }"));
}

TEST(DataShapeParser, RecordTwoFields) {
    EXPECT_EQ(ndt::make_struct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("{\n"
                        "    val: float32,\n"
                        "    id: int64\n"
                        "}\n"));
    EXPECT_EQ(ndt::make_struct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("{\n"
                        "    val: float32,\n"
                        "    id: int64,\n"
                        "}\n"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("c{\n"
                        "    val: float32,\n"
                        "    id: int64\n"
                        "}\n"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("c{\n"
                        "    val: float32,\n"
                        "    id: int64,\n"
                        "}\n"));
}

TEST(DataShapeParser, FuncProto) {
    std::cout << ndt::type("(int, int) -> int") << std::endl;
    std::cout << ndt::type("(int, x: int) -> int") << std::endl;

    std::cout << ndt::make_funcproto<int (int, int)>("x", "y") << std::endl;

//    std::exit(-1);
}

TEST(DataShapeParser, ErrorBasic) {
    try {
        type_from_datashape("float65");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 1") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized data type") != string::npos);
    }
    try {
        type_from_datashape("float64+");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("unexpected token") != string::npos);
    }
    try {
        type_from_datashape("3 * int32 * float64");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 5") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized dimension type") != string::npos);
    }
}

TEST(DataShapeParser, ErrorString) {
    try {
        type_from_datashape("string[");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("expected a size integer or string encoding") != string::npos);
    }
    try {
        type_from_datashape("string[0]");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("string size cannot be zero") != string::npos);
    }
    try {
        type_from_datashape("string['badencoding']");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized string encoding") != string::npos);
    }
    try {
        type_from_datashape("string['U8',]");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 12") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ']'") != string::npos);
    }
    try {
        type_from_datashape("string[3,'U8',10]");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 14") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ']'") != string::npos);
    }
    try {
        type_from_datashape("string[3,3]");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 10") != string::npos);
        EXPECT_TRUE(msg.find("expected a string encoding") != string::npos);
    }
}

TEST(DataShapeParser, ErrorRecord) {
    try {
        type_from_datashape("{\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 2") != string::npos);
        EXPECT_TRUE(msg.find("expected a record item") != string::npos);
    }
    try {
        type_from_datashape("{\n"
            "   id: int64\n"
            "   name: string\n"
            "   amount: invalidtype\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 13") != string::npos);
        EXPECT_TRUE(msg.find("expected ',' or '}'") != string::npos);
    }
    try {
        type_from_datashape("{\n"
            "   id: int64,\n"
            "   name: string,\n"
            "   amount: invalidtype;\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 12") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized data type") != string::npos);
    }
    try {
        type_from_datashape("{\n"
            "   id: int64,\n"
            "   name: string,\n"
            "   amount: %,\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 11") != string::npos);
        EXPECT_TRUE(msg.find("expected a data type") != string::npos);
    }
    try {
        type_from_datashape("{\n"
            "   id: int64,\n"
            "   name: string,\n"
            "   amount+ float32,\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 10") != string::npos);
        EXPECT_TRUE(msg.find("expected ':'") != string::npos);
    }
    try {
        type_from_datashape("{\n"
            "   id: int64,\n"
            "   name: (3 * string,\n"
            "   amount: float32,\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 3, column 21") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ')'") != string::npos);
    }
}

TEST(DataShapeParser, ErrorTypeAlias) {
    try {
        type_from_datashape("\n"
            "type 33 = int32\n"
            "2, int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 5") != string::npos);
        EXPECT_TRUE(msg.find("expected an identifier") != string::npos);
    }
    try {
        type_from_datashape("\n"
            "type MyInt - int32\n"
            "2, MyInt\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 11") != string::npos);
        EXPECT_TRUE(msg.find("expected an '='") != string::npos);
    }
    try {
        type_from_datashape("\n"
            "type MyInt = &\n"
            "2 * MyInt\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 13") != string::npos);
        EXPECT_TRUE(msg.find("expected a data type") != string::npos);
    }
    try {
        type_from_datashape("\n"
            "type int32 = int64\n"
            "2 * int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 6") != string::npos);
        EXPECT_TRUE(msg.find("cannot redefine") != string::npos);
    }
    try {
        type_from_datashape("\n"
            "type MyInt = int64\n"
            "type MyInt = int32\n"
            "2 * MyInt\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 3, column 6") != string::npos);
        EXPECT_TRUE(msg.find("type name already defined") != string::npos);
    }
}

TEST(DataShapeParser, KivaLoanDataShape) {
    const char *klds =
        "type KivaLoan = {\n"
        "    id: int64,\n"
        "    name: string,\n"
        "    description: {\n"
        "        languages: var * string[2],\n"
        "    #    texts: map(string[2], string),\n"
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
        "    # For 'delinquent', saw values \"null\" and \"true\" in brief search, map null -> false on import?\n"
        "    delinquent: bool,\n"
        "    location: {\n"
        "        country_code: string[2],\n"
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
        "        gender: string[2], # GenderType\n"
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
        "    #        nonpayment: Categorical(string, [\"lender\", \"partner\"]),\n"
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
    ndt::type d = type_from_datashape(klds);
    EXPECT_EQ(struct_type_id, d.get_type_id());
}

TEST(DataShapeParser, SpecialCharacterFields){
  /*
    This suite is designed to test the handling of special characters in struct and cstruct field names 
      when the datashape parser is turning a string into types.
    Any field name that has special characters should be enclosed in quotes in order to be handled properly.
    Special characters may include:
    - space
    - slashes, comma, :
    - unicode, \n, \r, \t, etc.
    - forwardslash?
    - parens?
    - date and time?
    - numbers
   */

  //p_dt == primary_datatype, s_dt == secondary_datatype
  ndt::type p_dt, s_dt;
  //ss == secondary_stream. This variable will be used to turn an existing datatype into a string. That string will later be re-parsed and 
  // compared against the primary datatype for equality.
  stringstream ss;

  //Test basic occurrences of spaces in field names.
  //Potential special cases are:
  // beginning of name
  // end of name
  // middle of name
  // multiple spaces at beginning, middle, or end
  p_dt=type_from_datashape("{ 'sepal length' : float64 }");
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "sepal length"));
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'sepal length' : float64 }");
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "sepal length"));
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("{ ' sepal length' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), " sepal length"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ ' sepal length' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), " sepal length"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ 'sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "sepal length "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "sepal length "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ ' sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), " sepal length "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ ' sepal length ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), " sepal length "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '   sepal         length   ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "   sepal         length   "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '   sepal         length   ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "   sepal         length   "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  //parens test
  //Potential special cases are:
  // beginning of name
  // end of name
  // middle of name
  // unmatched parens
  // multiple parens
  p_dt=type_from_datashape("{ 'unique key(foo)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "unique key(foo)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'unique key(foo)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "unique key(foo)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '(foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "(foo unique key"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '(foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "(foo unique key"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '(foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "(foo unique key)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '(foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "(foo unique key)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ 'foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "foo unique key)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'foo unique key)' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "foo unique key)"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '( foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "( foo unique key"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '( foo unique key' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "( foo unique key"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '((((foouniquekey))())((' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "((((foouniquekey))())(("));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '((((foouniquekey))())((' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "((((foouniquekey))())(("));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);




  //Quote tests. These can get a little tricky since the quotes must be escaped, and the backslashes may have to be escaped as well
  p_dt=type_from_datashape("{ 'sepal \"\"\" length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "sepal \"\"\" length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'sepal \"\"\" length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "sepal \"\"\" length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "sepal ''' length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "sepal ''' length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);



  p_dt=type_from_datashape("{ ' sepal \\'\\'\\' length ' : float64, 'sepal \" \\\" \\\\\\' width ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), " sepal ''' length ", ndt::make_type<double>(), "sepal \" \" \\' width "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ ' sepal \\'\\'\\' length ' : float64, 'sepal \" \\\" \\\\\\' width ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), " sepal ''' length ", ndt::make_type<double>(), "sepal \" \" \\' width "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);



  p_dt=type_from_datashape("{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "sepal ''' length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ 'sepal \\'\\'\\' length' : float64, 'sepal width' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "sepal ''' length", ndt::make_type<double>(), "sepal width"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  //A space is a valid field name. Perhaps it should not be.
  p_dt=type_from_datashape("{ ' ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), " "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ ' ' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), " "));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  //NOTE: According to the grammar, bare backslashes are not allowed as field names.
  //      Also not allowed are bare \n, \r, ", or ' characters
  //      So, that means that the following tests should fail.
  //This one hurts my brain. When the string is parsed, each \\ gets turned into a single \.
  //The final \\ is interpreted as a single \ in the name.
  //But, when converting the datatype back to a string, the single \ is escaped as a \\ by print_escaped_utf8_string() via 
  //  struct_type::print_type() and cstruct_type::print_type()
  /*To see that kind of code in action, try this:
    char char_backslash=char(92);
    const char *backslash=&c_bs;
    print_escaped_utf8_string(std::cout, backslash, backslash+1, true);
   */
  p_dt=type_from_datashape("{ '\\\\' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\\"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\\\\' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\\"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("{ '\n' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\n"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\n' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\n"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '\r' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\r"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\r' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\r"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("{ '\\\'' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\'"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\\\'' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\'"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("{ '\"' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\""));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\"' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\""));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  //Testing forward slashes
  p_dt=type_from_datashape("{ '/usr/bin' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "/usr/bin"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '/usr/bin' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "/usr/bin"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{ '\\\\abcdef' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "\\abcdef"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '\\\\abcdef' : float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "\\abcdef"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  p_dt=type_from_datashape("{'a field with spaces and \" \\\'': int}");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<int>(), "a field with spaces and \" \'"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{'a field with spaces and \" \\\'': int}");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<int>(), "a field with spaces and \" \'"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  //Testing field names as numbers
  p_dt=type_from_datashape("{ '1234' : float64, '-1234':float64,'500=1234':float64 }");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<double>(), "1234", ndt::make_type<double>(), "-1234",ndt::make_type<double>(), "500=1234"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{ '1234' : float64, '-1234':float64,'500=1234':float64 }");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<double>(), "1234", ndt::make_type<double>(), "-1234",ndt::make_type<double>(), "500=1234"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);




  //Other characters or strings that the parser may have trouble with include
  // {,},[,],:,comma,...,>,=,*,<,>,-
  p_dt=type_from_datashape("{'field name[100]': int, 'field name [x]': int, 'field : {name-=foo} [0*<>...,]': int}");
  EXPECT_EQ(p_dt, ndt::make_struct(ndt::make_type<int>(), "field name[100]", ndt::make_type<int>(), "field name [x]", ndt::make_type<int>(), "field : {name-=foo} [0*<>...,]"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);

  p_dt=type_from_datashape("c{'field name[100]': int, 'field name [x]': int, 'field : {name-=foo} [0*<>...,]': int}");
  EXPECT_EQ(p_dt, ndt::make_cstruct(ndt::make_type<int>(), "field name[100]", ndt::make_type<int>(), "field name [x]", ndt::make_type<int>(), "field : {name-=foo} [0*<>...,]"));
  ss<<p_dt; s_dt=type_from_datashape(ss.str()); ss.str("");
  EXPECT_EQ(p_dt, s_dt);


  /*Failing tests:
    If special names aren't enclosed in quotes, the parser should throw an exception.
   */
  try{
    ndt::type d=type_from_datashape("{ bad name : int }");
    EXPECT_TRUE(false);
  } catch (std::runtime_error &e) {
    string msg=e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 6") != string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != string::npos);
  }

  try{
    ndt::type d=type_from_datashape("{ name(foo) : int }");
    EXPECT_TRUE(false);
  } catch (std::runtime_error &e) {
    string msg=e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 7") != string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != string::npos);
  }

  try{
    ndt::type d=type_from_datashape("{ name\' : int }");
    EXPECT_TRUE(false);
  } catch (std::runtime_error &e) {
    string msg=e.what();
    EXPECT_TRUE(msg.find("Error parsing datashape at line 1, column 7") != string::npos);
    EXPECT_TRUE(msg.find("Message: expected ':' after record item name") != string::npos);
  }

}
