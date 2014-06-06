//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/array_range.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/elwise.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>

using namespace std;
using namespace dynd;

int main()
{
    try {
        nd::array a = nd::empty<double[4][3]>();
        a(0).vals() = nd::range(3);
        a(1).vals() = nd::range(3) + 5;
        a(2).vals() = nd::range(3) + 10;
        a(3).vals() = nd::range(3) + 12;

        bool reduction_dims[2] = {true, true};
        nd::arrfunc sum = lift_reduction_arrfunc(
            kernels::make_builtin_sum_reduction_arrfunc(float64_type_id),
            ndt::type("strided * strided * float64"), nd::arrfunc(), false, 2,
            reduction_dims, true, true, false, nd::array());

        cout << a << endl;
        nd::array b = nd::elwise([&](double x) -> double {
            return sum(a + x).as<double>();
        }, a);
        cout << b << endl;

    } catch(const std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
