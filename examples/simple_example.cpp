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
        ndarray a;
        char values[8] = {1,2,3,4,5,6,7,8};

        // Constructor assignment
        a = values;
        a.debug_dump(cout);
        a(0).debug_dump(cout);
        a(1).debug_dump(cout);
        EXPECT_EQ(1, a.get_ndim());
        EXPECT_EQ(8, a.get_shape(0));
        EXPECT_EQ(make_dtype<char>(), a.get_dtype());
        EXPECT_EQ(1, a(0).as<char>());
        EXPECT_EQ(2, a(1).as<char>());
        EXPECT_EQ(3, a(2).as<char>());
        EXPECT_EQ(4, a(3).as<char>());
        EXPECT_EQ(5, a(4).as<char>());
        EXPECT_EQ(6, a(5).as<char>());
        EXPECT_EQ(7, a(6).as<char>());
        EXPECT_EQ(8, a(7).as<char>());

        // Value assignment
        a.vals() = 0;
        EXPECT_EQ(0, a(0).as<char>());
        a.vals() = values;
        EXPECT_EQ(1, a(0).as<char>());
        EXPECT_EQ(2, a(1).as<char>());
        EXPECT_EQ(3, a(2).as<char>());
        EXPECT_EQ(4, a(3).as<char>());
        EXPECT_EQ(5, a(4).as<char>());
        EXPECT_EQ(6, a(5).as<char>());
        EXPECT_EQ(7, a(6).as<char>());
        EXPECT_EQ(8, a(7).as<char>());
    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
