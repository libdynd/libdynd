//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/ndarray_arange.hpp>

using namespace std;
using namespace dnd;

int main1()
{
    try {
        ndarray a;

        //a = {1, 5, 7};
        int avals[] = {1, 5, 7};
        a = avals;

        cout << a << endl;

        ndarray a2 = a.as_dtype<float>();

        cout << a2 << endl;

        ndarray a3 = a.as_dtype<double>();

        cout << a3 << endl;

        ndarray a4 = a2.as_dtype<double>();

        cout << a4 << endl;
        return 0;

        float avals2[2][3] = {{1,2,3}, {3,2,9}};
        ndarray b = avals2;

        ndarray c = a + b;

        c.debug_dump(cout);
        cout << c << endl;

        cout << c(0,1) << endl;
        a(1).val_assign(1.5f);
        cout << c(0,1) << endl;

        return 0;
    } catch(std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
}

#define EXPECT_EQ(a, b) \
    cout << "first   : " << (a) << endl \
         << "second  : " << (b) << endl

int main()
{
    try {
        ndarray a(2, make_fixedstring_dtype(string_encoding_utf_8, 7));
        single_compare_kernel_instance k;

        a.debug_dump(cout);
        a(0).vals() = std::string("abc");
        return 0;
        a.debug_dump(cout);
        a(1).vals() = std::string("abd");
        a.debug_dump(cout);

        // test ascii kernel
        a = a.vals();
        a.debug_dump(cout);
        a.get_dtype().get_single_compare_kernel(k);
        EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() < a(1).as<std::string>());
        EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() <= a(1).as<std::string>());
        EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() == a(1).as<std::string>());
        EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() != a(1).as<std::string>());
        EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() >= a(1).as<std::string>());
        EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), a(0).as<std::string>() > a(1).as<std::string>());

        // TODO: means for not hardcoding expected results in utf string comparison tests

        // test utf8 kernel
        a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_8, 7));
        a = a.vals();
        a.get_dtype().get_single_compare_kernel(k);
        EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);

        // test utf16 kernel
        a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_16, 7));
        a = a.vals();
        a.get_dtype().get_single_compare_kernel(k);
        EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);

        // test utf32 kernel
        a = a.as_dtype(make_fixedstring_dtype(string_encoding_utf_32, 7));
        a = a.vals();
        a.get_dtype().get_single_compare_kernel(k);
        EXPECT_EQ(k.comparisons[less_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[less_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[not_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), true);
        EXPECT_EQ(k.comparisons[greater_equal_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
        EXPECT_EQ(k.comparisons[greater_id]((char *)a(0).get_readonly_originptr(), (char *)a(1).get_readonly_originptr(), k.auxdata), false);
    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
