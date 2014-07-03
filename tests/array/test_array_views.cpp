//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/random.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayViews, OneDimensionalRawMemory) {
    nd::array a, b;
    signed char c_values[8];
    uint64_t u8_value;

    // Make an 8 byte aligned array of 80 chars
    a = nd::empty<uint64_t[10]>();
    a = a.view_scalars(ndt::make_type<char>());

    // Initialize the char values from a uint64_t,
    // to avoid having to know the endianness
    u8_value = 0x102030405060708ULL;
    memcpy(c_values, &u8_value, 8);
    a(irange() < 8).vals() = c_values;
    b = a.view_scalars<uint64_t>();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<uint64_t>()), b.get_type());
    EXPECT_EQ(1u, b.get_shape().size());
    EXPECT_EQ(10, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b(0).as<uint64_t>());
    b(0).vals() = 0x0505050505050505ULL;
    EXPECT_EQ(5, a(0).as<char>());

    // The system should automatically apply unaligned<>
    // where necessary
    a(1 <= irange() < 9).vals() = c_values;
    b = a(1 <= irange() < 73).view_scalars<uint64_t>();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_view(ndt::make_type<uint64_t>(), ndt::make_fixedbytes(8, 1))),
                    b.get_type());
    EXPECT_EQ(1u, b.get_shape().size());
    EXPECT_EQ(9, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr() + 1, b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b(0).as<uint64_t>());
}

TEST(ArrayViews, MultiDimensionalRawMemory) {
    nd::array a, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffffffff, 0x80000000, 0}};

    a = values;

    // Should throw if the view type is the wrong size
    EXPECT_THROW(b = a.view_scalars<int16_t>(), dynd::type_error);

    b = a.view_scalars<int32_t>();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int32_t>(), 2), b.get_type());
    EXPECT_EQ(2u, b.get_shape().size());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b(0, 0).as<int32_t>());
    EXPECT_EQ(2, b(0, 1).as<int32_t>());
    EXPECT_EQ(3, b(0, 2).as<int32_t>());
    EXPECT_EQ(-1, b(1, 0).as<int32_t>());
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), b(1, 1).as<int32_t>());
    EXPECT_EQ(0, b(1, 2).as<int32_t>());
}

TEST(ArrayViews, ExpressionDType) {
    nd::array a, a_u2, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffff, 0x8000, 0}};

    // Create a conversion from uint32_t -> uint16_t, followed by a
    // view uint16_t -> int16_t
    a = values;
    a_u2 = a.ucast<uint16_t>();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_convert<uint16_t, uint32_t>(), 2), a_u2.get_type());

    // Wrong size, so should throw
    EXPECT_THROW(b = a_u2.view_scalars<int32_t>(), dynd::type_error);

    b = a_u2.view_scalars<int16_t>();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_view(ndt::make_type<int16_t>(), ndt::make_convert<uint16_t, uint32_t>()), 2),
                    b.get_type());
    EXPECT_EQ(2u, b.get_shape().size());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b(0, 0).as<int16_t>());
    EXPECT_EQ(2, b(0, 1).as<int16_t>());
    EXPECT_EQ(3, b(0, 2).as<int16_t>());
    EXPECT_EQ(-1, b(1, 0).as<int16_t>());
    EXPECT_EQ(std::numeric_limits<int16_t>::min(), b(1, 1).as<int16_t>());
    EXPECT_EQ(0, b(1, 2).as<int16_t>());
}

TEST(ArrayViews, OneDimPermute) {
    int vals0[3] = {0, 1, 2};

    nd::array a = nd::empty<int[3]>();
    a.vals() = vals0;

    intptr_t ndim = 1;
    intptr_t axes[1] = {0};
    nd::array b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(a(i).as<int>(), b(i).as<int>());
    }

    a = nd::empty(ndt::type("3 * int"));
    a.vals() = vals0;

    b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(a(i).as<int>(), b(i).as<int>());
    }
}

TEST(ArrayViews, TwoDimPermute) {
    int vals0[3][3] = {{0, 1, 2}, {3, 4, -4}, {-3, -2, -1}};

    nd::array a = nd::empty<int[3][3]>();
    a.vals() = vals0;

    intptr_t ndim = 2;
    intptr_t axes[2] = {1, 0};
    nd::array b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(3, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(ndt::type("strided * strided * int"), b.get_type());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j).as<int>(), b(j, i).as<int>());
        }
    }

    a = nd::empty(3, ndt::type("3 * int"));
    a.vals() = vals0;

    b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(ndt::type("strided * strided * int"), b.get_type());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j).as<int>(), b(j, i).as<int>());
        }
    }

    a = nd::empty(ndt::type("3 * 3 * int"));
    a.vals() = vals0;

    b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(ndt::type("strided * strided * int"), b.get_type());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j).as<int>(), b(j, i).as<int>());
        }
    }

    int vals1[4][3] = {{0, 1, 2}, {3, 4, -4}, {-3, -2, -1}, {-5, 0, 2}};

    a = nd::empty<int[4][3]>();
    a.vals() = vals1;

    b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(ndt::type("strided * strided * int"), b.get_type());
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j).as<int>(), b(j, i).as<int>());
        }
    }

    a = nd::empty(ndt::type("4 * 3 * int"));
    a.vals() = vals1;

    b = a.permute(ndim, axes);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(ndt::type("strided * strided * int"), b.get_type());
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(a(i, j).as<int>(), b(j, i).as<int>());
        }
    }
}

TEST(ArrayViews, NDimPermute) {
    const intptr_t ndim0 = 4;
    intptr_t shape0[ndim0] = {7, 10, 15, 23};
    nd::array a = nd::typed_rand(ndim0, shape0, ndt::type("strided * strided * strided * strided * float64"));

    intptr_t axes0[ndim0] = {2, 3, 1, 0};
    nd::array b = a.permute(ndim0, axes0);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[3]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[1]);
    EXPECT_EQ(ndt::type("strided * strided * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape0[0]; ++i) {
        for (int j = 0; j < shape0[1]; ++j) {
            for (int k = 0; k < shape0[2]; ++k) {
                for (int l = 0; l < shape0[3]; ++l) {
                    EXPECT_EQ(a(i, j, k, l).as<double>(), b(k, l, j, i).as<double>());
                }
            }
        }
    }

    a = nd::typed_rand(ndim0, shape0, ndt::type("strided * strided * strided * 23 * float64"));

    b = a.permute(ndim0, axes0);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[3]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[1]);
    EXPECT_EQ(ndt::type("strided * strided * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape0[0]; ++i) {
        for (int j = 0; j < shape0[1]; ++j) {
            for (int k = 0; k < shape0[2]; ++k) {
                for (int l = 0; l < shape0[3]; ++l) {
                    EXPECT_EQ(a(i, j, k, l).as<double>(), b(k, l, j, i).as<double>());
                }
            }
        }
    }

    a = nd::typed_rand(ndim0, shape0, ndt::type("strided * strided * 15 * strided * float64"));

    b = a.permute(ndim0, axes0);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[3]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[1]);
    EXPECT_EQ(ndt::type("strided * strided * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape0[0]; ++i) {
        for (int j = 0; j < shape0[1]; ++j) {
            for (int k = 0; k < shape0[2]; ++k) {
                for (int l = 0; l < shape0[3]; ++l) {
                    EXPECT_EQ(a(i, j, k, l).as<double>(), b(k, l, j, i).as<double>());
                }
            }
        }
    }

    const intptr_t ndim1 = 5;
    intptr_t shape1[ndim1] = {5, 8, 3, 4, 2};
    a = nd::typed_rand(ndim1, shape1, ndt::type("strided * strided * strided * 4 * 2 * float64"));

    intptr_t axes1[ndim1] = {1, 0, 2, 3, 4};
    b = a.permute(2, axes1);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[3]);
    EXPECT_EQ(a.get_shape()[4], b.get_shape()[4]);
    EXPECT_EQ(ndt::type("strided * strided * strided * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape1[0]; ++i) {
        for (int j = 0; j < shape1[1]; ++j) {
            for (int k = 0; k < shape1[2]; ++k) {
                for (int l = 0; l < shape1[3]; ++l) {
                    for (int m = 0; m < shape1[4]; ++m) {
                        EXPECT_EQ(a(i, j, k, l, m).as<double>(), b(j, i, k, l, m).as<double>());
                    }
                }
            }
        }
    }

    a = nd::typed_rand(ndim1, shape1, ndt::type("strided * strided * var * strided * strided * float64"));

    b = a.permute(2, axes1);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[3]);
    EXPECT_EQ(a.get_shape()[4], b.get_shape()[4]);
    EXPECT_EQ(ndt::type("strided * strided * var * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape1[0]; ++i) {
        for (int j = 0; j < shape1[1]; ++j) {
            for (int k = 0; k < shape1[2]; ++k) {
                for (int l = 0; l < shape1[3]; ++l) {
                    for (int m = 0; m < shape1[4]; ++m) {
                        EXPECT_EQ(a(i, j, k, l, m).as<double>(), b(j, i, k, l, m).as<double>());
                    }
                }
            }
        }
    }

    intptr_t axes2[ndim1] = {1, 0, 2, 4, 3};
    b = a.permute(ndim1, axes2);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[4]);
    EXPECT_EQ(a.get_shape()[4], b.get_shape()[3]);
    EXPECT_EQ(ndt::type("strided * strided * var * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape1[0]; ++i) {
        for (int j = 0; j < shape1[1]; ++j) {
            for (int k = 0; k < shape1[2]; ++k) {
                for (int l = 0; l < shape1[3]; ++l) {
                    for (int m = 0; m < shape1[4]; ++m) {
                        EXPECT_EQ(a(i, j, k, l, m).as<double>(), b(j, i, k, m, l).as<double>());
                    }
                }
            }
        }
    }

    a = nd::typed_rand(ndim1, shape1, ndt::type("var * var * var * strided * strided * float64"));

    intptr_t axes3[ndim1] = {0, 1, 2, 4, 3};
    b = a.permute(ndim1, axes3);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(a.get_ndim(), b.get_ndim());
    EXPECT_EQ(a.get_shape()[0], b.get_shape()[0]);
    EXPECT_EQ(a.get_shape()[1], b.get_shape()[1]);
    EXPECT_EQ(a.get_shape()[2], b.get_shape()[2]);
    EXPECT_EQ(a.get_shape()[3], b.get_shape()[4]);
    EXPECT_EQ(a.get_shape()[4], b.get_shape()[3]);
    EXPECT_EQ(ndt::type("var * var * var * strided * strided * float64"), b.get_type());
    for (int i = 0; i < shape1[0]; ++i) {
        for (int j = 0; j < shape1[1]; ++j) {
            for (int k = 0; k < shape1[2]; ++k) {
                for (int l = 0; l < shape1[3]; ++l) {
                    for (int m = 0; m < shape1[4]; ++m) {
                        EXPECT_EQ(a(i, j, k, l, m).as<double>(), b(i, j, k, m, l).as<double>());
                    }
                }
            }
        }
    }
}

TEST(ArrayViews, NDimPermute_BadPerms) {
  nd::array a;
  const intptr_t ndim1 = 5;
  intptr_t shape1[ndim1] = {5, 8, 3, 4, 2};
  a = nd::typed_rand(
      ndim1, shape1,
      ndt::type("strided * strided * var * strided * strided * float64"));

  // A dimension may not be permuted across a var dimension
  intptr_t axes1[ndim1] = {1, 3, 2, 0, 4};
  EXPECT_THROW(a.permute(ndim1, axes1), invalid_argument);

  a = nd::typed_rand(
      ndim1, shape1,
      ndt::type("strided * strided * var * var * strided * float64"));

  // A var dimension dimension may not change position
  intptr_t axes2[ndim1] = {0, 1, 3, 2, 4};
  EXPECT_THROW(a.permute(ndim1, axes2), invalid_argument);

  a = nd::typed_rand(
      ndim1, shape1,
      ndt::type("strided * strided * strided * strided * strided * float64"));

  // The permutation must be valid and not be larger than ndim
  intptr_t axes3[ndim1] = {0, 1, 2, 3, 5};
  EXPECT_THROW(a.permute(ndim1, axes3), invalid_argument);
  intptr_t axes4[ndim1] = {0, 1, 2, 3, 0};
  EXPECT_THROW(a.permute(ndim1, axes4), invalid_argument);
  intptr_t axes5[ndim1 + 1] = {0, 1, 2, 3, 4, 5};
  EXPECT_THROW(a.permute(ndim1 + 1, axes5), invalid_argument);
}
