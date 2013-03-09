//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/datashape_parser.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/json_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeParser, Basic) {
    EXPECT_EQ(make_dtype<void>(), dtype_from_datashape("void"));
    EXPECT_EQ(make_dtype<dynd_bool>(), dtype_from_datashape("bool"));
    EXPECT_EQ(make_dtype<int8_t>(), dtype_from_datashape("int8"));
    EXPECT_EQ(make_dtype<int16_t>(), dtype_from_datashape("int16"));
    EXPECT_EQ(make_dtype<int32_t>(), dtype_from_datashape("int32"));
    EXPECT_EQ(make_dtype<int64_t>(), dtype_from_datashape("int64"));
    EXPECT_EQ(make_dtype<uint8_t>(), dtype_from_datashape("uint8"));
    EXPECT_EQ(make_dtype<uint16_t>(), dtype_from_datashape("uint16"));
    EXPECT_EQ(make_dtype<uint32_t>(), dtype_from_datashape("uint32"));
    EXPECT_EQ(make_dtype<uint64_t>(), dtype_from_datashape("uint64"));
    EXPECT_EQ(make_dtype<float>(), dtype_from_datashape("float32"));
    EXPECT_EQ(make_dtype<double>(), dtype_from_datashape("float64"));
    EXPECT_EQ(make_dtype<complex<float> >(), dtype_from_datashape("complex64"));
    EXPECT_EQ(make_dtype<complex<double> >(), dtype_from_datashape("complex128"));
    EXPECT_EQ(make_json_dtype(), dtype_from_datashape("json"));
    EXPECT_EQ(make_date_dtype(), dtype_from_datashape("date"));
}

TEST(DataShapeParser, BasicThrow) {
    EXPECT_THROW(dtype_from_datashape("boot"), runtime_error);
    EXPECT_THROW(dtype_from_datashape("int33"), runtime_error);
}

TEST(DataShapeParser, StringAtoms) {
    // Default string
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    dtype_from_datashape("string"));
    // String with encoding
    EXPECT_EQ(make_string_dtype(string_encoding_ascii),
                    dtype_from_datashape("string('A')"));
    EXPECT_EQ(make_string_dtype(string_encoding_ascii),
                    dtype_from_datashape("string('ascii')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    dtype_from_datashape("string('U8')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    dtype_from_datashape("string('utf8')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    dtype_from_datashape("string('utf-8')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16),
                    dtype_from_datashape("string('U16')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16),
                    dtype_from_datashape("string('utf16')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16),
                    dtype_from_datashape("string('utf-16')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_32),
                    dtype_from_datashape("string('U32')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_32),
                    dtype_from_datashape("string('utf32')"));
    EXPECT_EQ(make_string_dtype(string_encoding_utf_32),
                    dtype_from_datashape("string('utf-32')"));
    EXPECT_EQ(make_string_dtype(string_encoding_ucs_2),
                    dtype_from_datashape("string('ucs2')"));
    EXPECT_EQ(make_string_dtype(string_encoding_ucs_2),
                    dtype_from_datashape("string('ucs-2')"));
    // String with size
    EXPECT_EQ(make_fixedstring_dtype(1, string_encoding_utf_8),
                    dtype_from_datashape("string(1)"));
    EXPECT_EQ(make_fixedstring_dtype(100, string_encoding_utf_8),
                    dtype_from_datashape("string(100)"));
    // String with size and encoding
    EXPECT_EQ(make_fixedstring_dtype(1, string_encoding_ascii),
                    dtype_from_datashape("string(1, 'A')"));
    EXPECT_EQ(make_fixedstring_dtype(10, string_encoding_utf_8),
                    dtype_from_datashape("string(10, 'U8')"));
    EXPECT_EQ(make_fixedstring_dtype(1000, string_encoding_utf_16),
                    dtype_from_datashape("string(1000,'U16')"));
}

TEST(DataShapeParser, StridedDim) {
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<dynd_bool>()),
                    dtype_from_datashape("M, bool"));
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<float>(), 2),
                    dtype_from_datashape("M, N, float32"));
}

TEST(DataShapeParser, FixedDim) {
    EXPECT_EQ(make_fixed_dim_dtype(3, make_dtype<dynd_bool>()), dtype_from_datashape("3, bool"));
    EXPECT_EQ(make_fixed_dim_dtype(4, make_fixed_dim_dtype(3, make_dtype<float>())),
                    dtype_from_datashape("4, 3, float32"));
}

TEST(DataShapeParser, VarDim) {
    EXPECT_EQ(make_var_dim_dtype(make_dtype<dynd_bool>()), dtype_from_datashape("VarDim, bool"));
    EXPECT_EQ(make_var_dim_dtype(make_var_dim_dtype(make_dtype<float>())),
                    dtype_from_datashape("VarDim, VarDim, float32"));
}

TEST(DataShapeParser, StridedFixedDim) {
    EXPECT_EQ(make_strided_dim_dtype(make_fixed_dim_dtype(3, make_dtype<float>())),
                    dtype_from_datashape("M, 3, float32"));
}

TEST(DataShapeParser, StridedVarFixedDim) {
    EXPECT_EQ(make_strided_dim_dtype(make_var_dim_dtype(make_fixed_dim_dtype(3, make_dtype<float>()))),
                    dtype_from_datashape("M, VarDim, 3, float32"));
}

TEST(DataShapeParser, RecordOneField) {
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<float>(), "val"),
                    dtype_from_datashape("{ val : float32 }"));
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<float>(), "val"),
                    dtype_from_datashape("{ val : float32 ; }"));
}

TEST(DataShapeParser, RecordTwoFields) {
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<float>(), "val", make_dtype<int64_t>(), "id"),
                    dtype_from_datashape("{\n"
                        "    val: float32;\n"
                        "    id: int64\n"
                        "}\n"));
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<float>(), "val", make_dtype<int64_t>(), "id"),
                    dtype_from_datashape("{\n"
                        "    val: float32;\n"
                        "    id: int64;\n"
                        "}\n"));
}

TEST(DataShapeParser, ErrorBasic) {
    try {
        dtype_from_datashape("float65");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 1") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized data type") != string::npos);
    }
    try {
        dtype_from_datashape("float64+");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("unexpected token") != string::npos);
    }
    try {
        dtype_from_datashape("3, int32, float64");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 4") != string::npos);
        EXPECT_TRUE(msg.find("only free variables") != string::npos);
    }
}

TEST(DataShapeParser, ErrorString) {
    try {
        dtype_from_datashape("string(");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("expected a size integer or string encoding") != string::npos);
    }
    try {
        dtype_from_datashape("string(0)");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("string size cannot be zero") != string::npos);
    }
    try {
        dtype_from_datashape("string('badencoding')");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 8") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized string encoding") != string::npos);
    }
    try {
        dtype_from_datashape("string('U8',)");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 12") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ')'") != string::npos);
    }
    try {
        dtype_from_datashape("string(3,'U8',10)");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 14") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ')'") != string::npos);
    }
    try {
        dtype_from_datashape("string(3,3)");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 10") != string::npos);
        EXPECT_TRUE(msg.find("expected a string encoding") != string::npos);
    }
}

TEST(DataShapeParser, ErrorRecord) {
    try {
        dtype_from_datashape("{\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 1, column 2") != string::npos);
        EXPECT_TRUE(msg.find("expected a record item") != string::npos);
    }
    try {
        dtype_from_datashape("{\n"
            "   id: int64\n"
            "   name: string\n"
            "   amount: invalidtype\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 13") != string::npos);
        EXPECT_TRUE(msg.find("expected ';' or '}'") != string::npos);
    }
    try {
        dtype_from_datashape("{\n"
            "   id: int64;\n"
            "   name: string;\n"
            "   amount: invalidtype;\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 12") != string::npos);
        EXPECT_TRUE(msg.find("unrecognized data type") != string::npos);
    }
    try {
        dtype_from_datashape("{\n"
            "   id: int64;\n"
            "   name: string;\n"
            "   amount: %;\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 11") != string::npos);
        EXPECT_TRUE(msg.find("expected a data type") != string::npos);
    }
    try {
        dtype_from_datashape("{\n"
            "   id: int64;\n"
            "   name: string;\n"
            "   amount+ float32;\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 4, column 10") != string::npos);
        EXPECT_TRUE(msg.find("expected ':'") != string::npos);
    }
    try {
        dtype_from_datashape("{\n"
            "   id: int64;\n"
            "   name: (3, string;\n"
            "   amount: float32;\n"
            "}\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 3, column 20") != string::npos);
        EXPECT_TRUE(msg.find("expected closing ')'") != string::npos);
    }
}

TEST(DataShapeParser, ErrorTypeAlias) {
    try {
        dtype_from_datashape("\n"
            "type MyInt = int32\n"
            "MyInt, int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 3, column 1") != string::npos);
        EXPECT_TRUE(msg.find("only free variables") != string::npos);
    }
    try {
        dtype_from_datashape("\n"
            "type 33 = int32\n"
            "2, int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 5") != string::npos);
        EXPECT_TRUE(msg.find("expected an identifier") != string::npos);
    }
    try {
        dtype_from_datashape("\n"
            "type MyInt - int32\n"
            "2, MyInt\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 11") != string::npos);
        EXPECT_TRUE(msg.find("expected an '='") != string::npos);
    }
    try {
        dtype_from_datashape("\n"
            "type MyInt = &\n"
            "2, MyInt\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 13") != string::npos);
        EXPECT_TRUE(msg.find("expected a data type") != string::npos);
    }
    try {
        dtype_from_datashape("\n"
            "type int32 = int64\n"
            "2, int32\n");
        EXPECT_TRUE(false);
    } catch (const runtime_error& e) {
        string msg = e.what();
        EXPECT_TRUE(msg.find("line 2, column 6") != string::npos);
        EXPECT_TRUE(msg.find("cannot redefine") != string::npos);
    }
    try {
        dtype_from_datashape("\n"
            "type MyInt = int64\n"
            "type MyInt = int32\n"
            "2, MyInt\n");
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
        "    id: int64;\n"
        "    name: string;\n"
        "    description: {\n"
        "        languages: VarDim, string(2);\n"
        "    #    texts: map(string(2), string);\n"
        "    };\n"
        "    status: string; # LoanStatusType;\n"
        "    funded_amount: float64;\n"
        "    #basket_amount: Option(float64);\n"
        "    paid_amount: float64;\n"
        "    image: {\n"
        "        id: int64;\n"
        "        template_id: int64;\n"
        "    };\n"
        "    #video: Option({\n"
        "    #    id: int64;\n"
        "    #    youtube_id: string;\n"
        "    #});\n"
        "    activity: string;\n"
        "    sector: string;\n"
        "    use: string;\n"
        "    # For 'delinquent', saw values \"null\" and \"true\" in brief search, map null -> false on import?\n"
        "    delinquent: bool;\n"
        "    location: {\n"
        "        country_code: string(2);\n"
        "        country: string;\n"
        "        town: string;\n"
        "        geo: {\n"
        "            level: string; # GeoLevelType\n"
        "            pairs: string; # latlong\n"
        "            type: string; # GeoTypeType\n"
        "        }\n"
        "    };\n"
        "    partner_id: int64;\n"
        "    #posted_date: datetime<seconds>;\n"
        "    #planned_expiration_date: Option(datetime<seconds>);\n"
        "    loan_amount: float64;\n"
        "    #currency_exchange_loss_amount: Option(float64);\n"
        "    borrowers: VarDim, {\n"
        "        first_name: string;\n"
        "        last_name: string;\n"
        "        gender: string(2); # GenderType\n"
        "        pictured: bool;\n"
        "    };\n"
        "    terms: {\n"
        "    #    disbursal_date: datetime<seconds>;\n"
        "    #    disbursal_currency: Option(string);\n"
        "        disbursal_amount: float64;\n"
        "        loan_amount: float64;\n"
        "        local_payments: VarDim, {\n"
        "    #        due_date: datetime<seconds>;\n"
        "            amount: float64;\n"
        "        };\n"
        "        scheduled_payments: VarDim, {\n"
        "    #        due_date: datetime<seconds>;\n"
        "            amount: float64;\n"
        "        };\n"
        "        loss_liability: {\n"
        "    #        nonpayment: Categorical(string, [\"lender\", \"partner\"]);\n"
        "            currency_exchange: string;\n"
        "    #        currency_exchange_coverage_rate: Option(float64);\n"
        "        }\n"
        "    };\n"
        "    payments: VarDim, {\n"
        "        amount: float64;\n"
        "        local_amount: float64;\n"
        "    #    processed_date: datetime<seconds>;\n"
        "    #    settlement_date: datetime<seconds>;\n"
        "        rounded_local_amount: float64;\n"
        "        currency_exchange_loss_amount: float64;\n"
        "        payment_id: int64;\n"
        "        comment: string;\n"
        "    };\n"
        "    #funded_date: datetime<seconds>;\n"
        "    #paid_date: datetime<seconds>;\n"
        "    journal_totals: {\n"
        "        entries: int64;\n"
        "        bulkEntries: int64;\n"
        "    }\n"
        "}\n";
    dtype d = dtype_from_datashape(klds);
    EXPECT_EQ(fixedstruct_type_id, d.get_type_id());
}
