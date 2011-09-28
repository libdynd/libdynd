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
}
