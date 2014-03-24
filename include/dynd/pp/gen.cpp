//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>

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

string args(const string& prefix, int start, int stop) {
    typedef pair<string, pair<int, int> > key_type;
    typedef string value_type;

    static map<key_type, value_type> lookup;

    key_type key(prefix, make_pair(start, stop));
    map<key_type, value_type>::iterator it = lookup.find(key);
    if (it != lookup.end()) {
        return it->second;
    } else {
        ostringstream oss;
        for (int i = start; i < stop - 1; i++) {
            oss << prefix << i << argsep(i);
        }
        oss << prefix << stop - 1;
        return lookup[key] = oss.str();
    }
}

string args(const string& prefix, int stop) {
    return args(prefix, 0, stop);
}

int main(int argc, char **argv) {
    const int pp_int_max = atoi(argv[1]);
    if (pp_int_max < 8) {
        throw runtime_error("the maximum integer cannot be less than 8");
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

    fout << "#define DYND_PP_AT_INT_MAX(" << args("A", pp_int_max + 1) << ", ...) A" << pp_int_max << endl;

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
        fout << "#define DYND_PP_CAT_" << i << "(A0" << argsep(false) << args("A", 1, i) << ") DYND_PP_CAT_2(A0"
            << argsep(false) << "DYND_PP_CAT_" << i - 1 << "(" << args("A", 1, i) << "))" << endl;
    }

    fout << endl;

    for (int i = 0; i <= pp_int_max; i++) {
        fout << "#define DYND_PP_EQ_" << i << "_" << i << " ," << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_INC(A) DYND_PP_CAT_2(DYND_PP_INC_, A)" << endl;
    for (int i = 0; i < pp_int_max; i++) {
        fout << "#define DYND_PP_INC_" << i << " " << i + 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_DEC(A) DYND_PP_CAT_2(DYND_PP_DEC_, A)" << endl;
    for (int i = 1; i <= pp_int_max; i++) {
        fout << "#define DYND_PP_DEC_" << i << " " << i - 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_MAP(MAC, SEP, ...) DYND_PP_CAT_2(DYND_PP_MAP_, DYND_PP_LEN(__VA_ARGS__))(MAC, SEP, __VA_ARGS__)" << endl;
    fout << "#define DYND_PP_MAP_0(MAC, SEP)" << endl;
    fout << "#define DYND_PP_MAP_1(MAC, SEP, A0) MAC(A0)" << endl;
    for (int i = 2; i < pp_int_max; i++) {
        fout << "#define DYND_PP_MAP_" << i << "(MAC, SEP, " << args("A", i) << ") MAC(A0) DYND_PP_ID SEP DYND_PP_MAP_";
        fout << i - 1 << "(MAC, SEP, " << args("A", 1, i) << ")" << endl;
    }

    fout << endl;

    fout << "#endif" << endl;

    fout.close();

    return 0;
}
