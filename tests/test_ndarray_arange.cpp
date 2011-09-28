#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"
#include "dnd/ndarray_arange.hpp"

using namespace std;
using namespace dnd;

TEST(NDArrayArange, Basic) {
    ndarray a;

    a = arange(1, 10);
    cout << a << "\n";
    a = arange(1., 10., 0.5);
    cout << a << "\n";
    cout << arange(25.f) << "\n";
    cout << arange(0.,1.,0.1) << "\n";
    cout << arange(0.f,1.f,0.1f) << "\n";
    cout << arange(0.f,1.f,0.01f) << "\n";

    cout << 2 * arange(20) << "\n";
    cout << arange(3 <= irange() <= 20) << "\n";

    cout << linspace(10, 20) << "\n";
    cout << linspace(0, 5.0, 10) << "\n";
}
