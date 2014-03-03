//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/datashape_parser.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/json_type.hpp>
#include <dynd/types/type_alignment.hpp>

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
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(), type_from_datashape("complex64"));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), type_from_datashape("complex128"));
    EXPECT_EQ(ndt::make_type<dynd_complex<float> >(), type_from_datashape("complex[float32]"));
    EXPECT_EQ(ndt::make_type<dynd_complex<double> >(), type_from_datashape("complex[float64]"));
    EXPECT_EQ(ndt::make_json(), type_from_datashape("json"));
    EXPECT_EQ(ndt::make_date(), type_from_datashape("date"));
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
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<dynd_bool>()),
                    type_from_datashape("M * unaligned[bool]"));
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_unaligned(ndt::make_type<float>()), 2),
                    type_from_datashape("M * N * unaligned[float32]"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_unaligned(ndt::make_type<int32_t>()), "x",
                                ndt::make_unaligned(ndt::make_type<int64_t>()), "y"),
                    type_from_datashape("{x : unaligned[int32], y : unaligned[int64]}"));
}

TEST(DataShapeParser, StridedDim) {
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<dynd_bool>()),
                    type_from_datashape("M * bool"));
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<float>(), 2),
                    type_from_datashape("M * N * float32"));
}

TEST(DataShapeParser, FixedDim) {
    EXPECT_EQ(ndt::make_fixed_dim(3, ndt::make_type<dynd_bool>()), type_from_datashape("3 * bool"));
    EXPECT_EQ(ndt::make_fixed_dim(4, ndt::make_fixed_dim(3, ndt::make_type<float>())),
                    type_from_datashape("4 * 3 * float32"));
}

TEST(DataShapeParser, VarDim) {
    EXPECT_EQ(ndt::make_var_dim(ndt::make_type<dynd_bool>()), type_from_datashape("var * bool"));
    EXPECT_EQ(ndt::make_var_dim(ndt::make_var_dim(ndt::make_type<float>())),
                    type_from_datashape("var * var * float32"));
}

TEST(DataShapeParser, StridedFixedDim) {
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_fixed_dim(3, ndt::make_type<float>())),
                    type_from_datashape("M * 3 * float32"));
}

TEST(DataShapeParser, StridedVarFixedDim) {
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_var_dim(ndt::make_fixed_dim(3, ndt::make_type<float>()))),
                    type_from_datashape("M * var * 3 * float32"));
}

TEST(DataShapeParser, RecordOneField) {
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val"),
                    type_from_datashape("{ val : float32 }"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val"),
                    type_from_datashape("{ val : float32 , }"));
}

TEST(DataShapeParser, RecordTwoFields) {
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("{\n"
                        "    val: float32,\n"
                        "    id: int64\n"
                        "}\n"));
    EXPECT_EQ(ndt::make_cstruct(ndt::make_type<float>(), "val", ndt::make_type<int64_t>(), "id"),
                    type_from_datashape("{\n"
                        "    val: float32,\n"
                        "    id: int64,\n"
                        "}\n"));
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
        EXPECT_TRUE(msg.find("only free variables") != string::npos);
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
            "type MyInt = int32\n"
            "MyInt * int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 3, column 1") != string::npos);
        EXPECT_TRUE(msg.find("only free variables") != string::npos);
    }
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
    EXPECT_EQ(cstruct_type_id, d.get_type_id());
}
