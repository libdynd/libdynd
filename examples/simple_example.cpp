//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
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

        a = arange(1, 10);
        cout << a << "\n";
        a = arange(1., 10., 0.5);
        cout << a << "\n";
        cout << arange(25.f) << "\n";
        cout << arange(0.,1.,0.1) << "\n";
        cout << arange(0.f,1.f,0.1f) << "\n";
        cout << arange(0.f,1.f,0.01f) << "\n";

        cout << ndarray(2) << "\n";
        cout << ndarray(2) * arange(20) << "\n";
        cout << 2 * arange(20) << "\n";
        cout << arange(3 <= irange() <= 20) << "\n";

        cout << linspace(10, 20) << "\n";
        cout << linspace(0, 5.0, 10) << "\n";


    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
