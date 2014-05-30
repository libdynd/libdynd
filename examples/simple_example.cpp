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

using namespace std;
using namespace dynd;

int main()
{
    try {
        nd::array a, b;

        a = parse_json("?int8", "123");
        cout << a.get_type() << endl;
        cout << a.as<int8_t>() << endl;
        a = parse_json("?int8", "null");
        cout << a.get_type() << endl;
        cout << a << endl;

        a = parse_json("9 * ?int", "[null, 3, null, -1000, 1, 3, null, null, null]");
        cout << a.get_type() << endl;
        b = nd::empty("9 * int");
        //b.vals() = a;

        b = nd::empty("9 * ?int64");
        b.vals() = a;
        cout << nd::view(b, ndt::type("9 * int64")) << endl;
        cout << b << endl;
    } catch(const std::exception& e) {
        cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
