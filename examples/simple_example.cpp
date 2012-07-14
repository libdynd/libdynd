//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/string_dtype.hpp>
#include <dnd/dtypes/byteswap_dtype.hpp>
#include <dnd/ndarray_arange.hpp>
#include <dnd/codegen/codegen_cache.hpp>
#include <dnd/codegen/unary_kernel_adapter_codegen.hpp>

using namespace std;
using namespace dnd;

typedef complex<double> A0;
typedef float R;

static R spin(A0 x) {
    return 2 * x.real() + 2;
}

int main()
{
    try {
        A0 a[2];
        R b[2];

        codegen_cache cgcache;
        unary_operation_t *optable = cgcache.codegen_unary_function_adapter(make_dtype<R>(), make_dtype<A0>(), cdecl_callconv);
        //codegen_unary_function_adapter(emb, make_dtype<int>(), make_dtype<float>(), cdecl_callconv);
        a[0] = 1;
        a[1] = 2;
        optable[0]((char *)b, sizeof(b[0]), (char *)a, sizeof(a[0]), 2, (AuxDataBase *)((uintptr_t)&spin|1));
        cout << b[0] << endl;
        cout << b[1] << endl;

        return 0;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}

#define EXPECT_EQ(a, b) \
    cout << "first   : " << (a) << endl \
         << "second  : " << (b) << endl

int main1()
{
    try {
    ndarray a, b;

    // std::string goes in as a utf8 fixed string
    a = std::string("abcdefg");
    EXPECT_EQ(make_fixedstring_dtype(string_encoding_utf_8, 7), a.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), a.as<std::string>());

    // Convert to a blockref string dtype with the same utf8 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_8));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_8), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the utf16 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_16));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_16), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_16),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the utf32 codec
    b = a.as_dtype(make_string_dtype(string_encoding_utf_32));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_utf_32), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_utf_32),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    // Convert to a blockref string dtype with the ascii codec
    b = a.as_dtype(make_string_dtype(string_encoding_ascii));
    EXPECT_EQ(make_convert_dtype(make_string_dtype(string_encoding_ascii), make_fixedstring_dtype(string_encoding_utf_8, 7)),
                b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_string_dtype(string_encoding_ascii),
                    b.get_dtype());
    EXPECT_EQ(std::string("abcdefg"), b.as<std::string>());

    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
