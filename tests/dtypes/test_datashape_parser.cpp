//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtypes/datashape_parser.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/array_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DataShapeParser, Basic) {
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
}

TEST(DataShapeParser, StridedDim) {
    EXPECT_EQ(make_strided_array_dtype(make_dtype<dynd_bool>()), dtype_from_datashape("M, bool"));
    EXPECT_EQ(make_strided_array_dtype(make_dtype<float>(), 2), dtype_from_datashape("M, N, float32"));
}

TEST(DataShapeParser, FixedDim) {
    EXPECT_EQ(make_fixedarray_dtype(make_dtype<dynd_bool>(), 3), dtype_from_datashape("3, bool"));
    EXPECT_EQ(make_fixedarray_dtype(make_fixedarray_dtype(make_dtype<float>(), 3), 4),
                    dtype_from_datashape("4, 3, float32"));
}

TEST(DataShapeParser, VarDim) {
    EXPECT_EQ(make_array_dtype(make_dtype<dynd_bool>()), dtype_from_datashape("VarDim, bool"));
    EXPECT_EQ(make_array_dtype(make_array_dtype(make_dtype<float>())),
                    dtype_from_datashape("VarDim, VarDim, float32"));
}

TEST(DataShapeParser, StridedFixedDim) {
    EXPECT_EQ(make_strided_array_dtype(make_fixedarray_dtype(make_dtype<float>(), 3)),
                    dtype_from_datashape("M, 3, float32"));
}

TEST(DataShapeParser, StridedVarFixedDim) {
    EXPECT_EQ(make_strided_array_dtype(make_array_dtype(make_fixedarray_dtype(make_dtype<float>(), 3))),
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
