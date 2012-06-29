//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dnd/ndarray.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/dtypes/fixedstring_dtype.hpp>
#include <dnd/dtypes/byteswap_dtype.hpp>
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

        int16_t value16 = 0x1362;
        a = ndarray(make_byteswap_dtype<int16_t>(), (char *)&value16);
        EXPECT_EQ(0x6213, a.as<int16_t>());

        int32_t value32 = 0x12345678;
        a = ndarray(make_byteswap_dtype<int32_t>(), (char *)&value32);
        EXPECT_EQ(0x78563412, a.as<int32_t>());

        int64_t value64 = 0x12345678abcdef01LL;
        a = ndarray(make_byteswap_dtype<int64_t>(), (char *)&value64);
        EXPECT_EQ(0x01efcdab78563412LL, a.as<int64_t>());

        value32 = 0xDA0F4940;
        a = ndarray(make_byteswap_dtype<float>(), (char *)&value32);
        EXPECT_EQ(3.1415926f, a.as<float>());

        value64 = 0x112D4454FB210940LL;
        a = ndarray(make_byteswap_dtype<double>(), (char *)&value64);
        EXPECT_EQ(3.14159265358979, a.as<double>());
        a = a.vals();
        EXPECT_EQ(3.14159265358979, a.as<double>());

        uint32_t value32_pair[2] = {0xDA0F4940, 0xC1B88FD3};
        a = ndarray(make_byteswap_dtype<complex<float> >(), (char *)&value32_pair);
        EXPECT_EQ(complex<float>(3.1415926f, -1.23456e12f), a.as<complex<float> >());

        int64_t value64_pair[2] = {0x112D4454FB210940LL, 0x002892B01FF771C2LL};
        a = ndarray(make_byteswap_dtype<complex<double> >(), (char *)&value64_pair);
        EXPECT_EQ(complex<double>(3.14159265358979, -1.2345678912345e12), a.as<complex<double> >());
    } catch(int){//std::exception& e) { cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
