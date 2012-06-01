#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/array_dtype.hpp"

using namespace std;
using namespace dnd;

TEST(ArrayDType, BasicConstructor) {
    intptr_t shape[] = {2,3,5};
    dtype adt(make_shared<array_dtype>(3, shape, make_dtype<int>()));
    cout << adt;
}