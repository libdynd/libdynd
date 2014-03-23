//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace std;

inline string argsep(bool newline)
{
    if (newline) {
        return ", \\\n    ";
    } else {
        return ", ";
    }
}

inline string argsep(int i, int args_per_line = 8) {
    return argsep((i + 1) % args_per_line == 0);
}

int main(int argc, char **argv) {
    const int pp_int_max = atoi(argv[1]);
    if (pp_int_max <= 0) {
        throw runtime_error("the maximum integer must be positive");
    }

    string filename("gen.hpp");
    if (argc > 2) {
        filename.insert(0, "/");
        filename.insert(0, argv[2]);
    }

    ofstream fout(filename.c_str());

    fout << "//" << endl;
    fout << "// Copyright (C) 2011-14 Irwin Zaid, DyND Developers" << endl;
    fout << "// BSD 2-Clause License, see LICENSE.txt" << endl;
    fout << "//" << endl;

    fout << endl;

    fout << "#ifndef _DYND__GEN_HPP_" << endl;
    fout << "#define _DYND__GEN_HPP_" << endl;

    fout << endl;

    fout << "#define DYND_PP_INT_MIN " << 0 << endl;
    fout << "#define DYND_PP_INT_MAX " << pp_int_max << endl;

    fout << endl;

    fout << "#define DYND_PP_INTS ";
    for (int i = 0; i < pp_int_max; i++) {
        fout << i << argsep(i);
    }
    fout << pp_int_max << endl;

    fout << endl;

    fout << "#define DYND_PP_AT_INT_MAX(";
    for (int i = 0; i <= pp_int_max; i++) {
        fout << "A" << i << argsep(i);
    }
    fout << "...) A" << pp_int_max << endl;

    fout << endl;

    fout << "#define DYND_PP_LEN_NONZERO(...) DYND_PP__LEN_NONZERO((__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp_int_max; i++) {
        fout << pp_int_max - i << argsep(i);
    }
    fout << 0 << "))" << endl;
    fout << "#define DYND_PP__LEN_NONZERO(PARENTHESIZED) DYND_PP_AT_INT_MAX PARENTHESIZED" << endl;

    fout << endl;

    fout << "#define DYND_PP_HAS_COMMA(...) DYND_PP__HAS_COMMA((__VA_ARGS__" << argsep(true);
    for (int i = 0; i <= pp_int_max - 2; i++) {
        fout << 1 << argsep(i);
    }
    fout << 0 << argsep(false) << 0 << "))" << endl;
    fout << "#define DYND_PP__HAS_COMMA(PARENTHESIZED) DYND_PP_AT_INT_MAX PARENTHESIZED" << endl;

    fout << endl;

    for (int i = 8; i <= pp_int_max; i++) {
        fout << "#define DYND_PP_CAT_" << i << "(A0" << argsep(false) << "...) DYND_PP_CAT_2(A0"
            << argsep(false) << "DYND_PP_CAT_" << i - 1 << "(__VA_ARGS__))" << endl;
    }

    fout << endl;

    for (int i = 0; i <= pp_int_max; i++) {
        fout << "#define DYND_PP_EQ_" << i << "_" << i << " ," << endl;
    }

    fout << endl;

    fout << "#endif" << endl;

    fout.close();

    return 0;
}
