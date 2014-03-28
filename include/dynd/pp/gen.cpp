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
    const int pp_len_max = atoi(argv[1]);
    if (pp_len_max < 8) {
        throw runtime_error("the maximum list length cannot be less than 8");
    }

    const int pp_int_max = pp_len_max - 1;

    const int pp_depth_max = atoi(argv[2]);
    if (pp_depth_max < 3) {
        throw runtime_error("the maximum macro depth cannot be less than 3");
    }

    string filename("gen.hpp");

    if (argc > 2) {
        filename.insert(0, "/");
        filename.insert(0, argv[3]);
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

    fout << "#define DYND_PP_INT_MAX " << pp_int_max << endl;
    fout << "#define DYND_PP_LEN_MAX " << pp_len_max << endl;

    fout << endl;

    fout << "#define DYND_PP_INTS (";
    for (int i = 0; i < pp_int_max; i++) {
        fout << i << argsep(i);
    }
    fout << pp_int_max << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_ZEROS_" << pp_len_max << " (";
    for (int i = 0; i < pp_int_max; i++) {
        fout << 0 << argsep(i);
    }
    fout << 0 << ")" << endl;

    fout << "#define DYND_PP_ONES_" << pp_len_max << " (";
    for (int i = 0; i < pp_int_max; i++) {
        fout << 1 << argsep(i);
    }
    fout << 1 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_GET_ARG_" << pp_len_max << "(...) DYND_PP_ID(DYND_PP__GET_ARG_" << pp_len_max << "(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP__GET_ARG_" << pp_len_max << "(" << args("A", pp_len_max + 1) << ", ...) A" << pp_len_max << endl;

    fout << endl;

    fout << "#define DYND_PP_HAS_COMMA(...) DYND_PP_ID(DYND_PP__HAS_COMMA(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP__HAS_COMMA(...) DYND_PP_GET_ARG_" << pp_len_max << "(__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp_int_max; i++) {
        fout << 1 << argsep(i);
    }
    fout << 0 << argsep(false) << 0 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_LEN(A) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(A))(DYND_PP_LEN_IF_EMPTY)(DYND_PP_LEN_IF_NOT_EMPTY)(A)" << endl;
    fout << "#define DYND_PP_LEN_IF_EMPTY(A) 0" << endl;
    fout << "#define DYND_PP_LEN_IF_NOT_EMPTY(A) DYND_PP_GET_ARG_" << pp_len_max << "(DYND_PP_ID A" << argsep(true);
    for (int i = 0; i < pp_int_max; i++) {
        fout << pp_len_max - i << argsep(i);
    }
    fout << 1 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_INC(A) DYND_PP_PASTE(DYND_PP_INC_, A)" << endl;
    for (int i = 0; i < pp_len_max; i++) {
        fout << "#define DYND_PP_INC_" << i << " " << i + 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_DEC(A) DYND_PP_PASTE(DYND_PP_DEC_, A)" << endl;
    for (int i = 1; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_DEC_" << i << " " << i - 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_CAT(A) DYND_PP_PASTE(DYND_PP_CAT_, DYND_PP_LEN(A))(A)" << endl;
    fout << "#define DYND_PP_CAT_0(A)" << endl;
    fout << "#define DYND_PP_CAT_1 DYND_PP_FIRST" << endl;
    fout << "#define DYND_PP_CAT_2(A) DYND_PP_PASTE(DYND_PP_FIRST(A), DYND_PP_FIRST(DYND_PP_POP_FIRST(A)))" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_CAT_" << i << "(A) DYND_PP_PASTE(DYND_PP_FIRST(A)";
        fout << argsep(false) << "DYND_PP_CAT_" << i - 1 << "(DYND_PP_POP_FIRST(A)))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_FROM(START, A) DYND_PP_PASTE(DYND_PP_SLICE_FROM_, START)(A)" << endl;
    fout << "#define DYND_PP_SLICE_FROM_0(A) A" << endl;
    fout << "#define DYND_PP_SLICE_FROM_1(A) DYND_PP_POP_FIRST(A)" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_FROM_" << i << "(A) DYND_PP_SLICE_FROM_" << i - 1 << "(DYND_PP_POP_FIRST(A))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_TO(STOP" << argsep(false) << "A) DYND_PP_PASTE(DYND_PP_SLICE_TO_" << argsep(false) << "STOP)(A)" << endl;
    fout << "#define DYND_PP_SLICE_TO_0(A) ()" << endl;
    fout << "#define DYND_PP_SLICE_TO_1(A) (DYND_PP_FIRST(A))" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_TO_" << i << "(A) DYND_PP_PREPEND(DYND_PP_FIRST(A)";
        fout << argsep(false) << "DYND_PP_SLICE_TO_" << i - 1 << "(DYND_PP_POP_FIRST(A)))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_WITH(STEP, A) DYND_PP_PASTE(DYND_PP_SLICE_WITH_, DYND_PP_PASTE(STEP, DYND_PP_PASTE(_, DYND_PP_LEN(A))))(A)" << endl;
    fout << "#define DYND_PP_SLICE_WITH_" << 1 << "_" << 0 << "(A) ()" << endl;
    for (int j = 1; j <= pp_len_max; j++) {
        fout << "#define DYND_PP_SLICE_WITH_" << 1 << "_" << j << "(A) A" << endl;
    }
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_WITH_" << i << "_" << 0 << "(A) ()" << endl;
        for (int j = 1; j <= pp_len_max; j++) {
            fout << "#define DYND_PP_SLICE_WITH_" << i << "_" << j << "(A) ";
            if (i >= j) {
                fout << "(DYND_PP_FIRST(A))";
            } else {
                fout << "DYND_PP_PREPEND(DYND_PP_FIRST(A)";
                fout << argsep(false) << "DYND_PP_SLICE_WITH_" << i << "_" << j - i << "(DYND_PP_SLICE_FROM(" << i << ", A)))";
            }
            fout << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_REPEAT(TOK, COUNT) DYND_PP_PASTE(DYND_PP_REPEAT_, COUNT)(TOK)" << endl;
    fout << "#define DYND_PP_REPEAT_0(TOK)" << endl;
    fout << "#define DYND_PP_REPEAT_1(TOK) TOK" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_REPEAT_" << i << "(TOK) TOK, DYND_PP_REPEAT_" << i - 1 << "(TOK)" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_MAP DYND_PP_NESTED_MAP(0)" << endl;
    fout << "#define DYND_PP_NESTED_MAP(DEPTH) DYND_PP_PASTE(DYND_PP_NESTED_MAP_, DEPTH)" << endl;
    for (int depth = 0; depth <= pp_depth_max; depth++) {
        fout << "#define DYND_PP_NESTED_MAP_" << depth << "(MAC, SEP, A) ";
        fout << "DYND_PP_PASTE(DYND_PP_NESTED_MAP_" << depth << "_ , DYND_PP_LEN(A))(MAC, SEP, A)" << endl;
        fout << "#define DYND_PP_NESTED_MAP_" << depth << "_" << "0(MAC, SEP, A)" << endl;
        fout << "#define DYND_PP_NESTED_MAP_" << depth << "_" << "1(MAC, SEP, A) MAC(DYND_PP_FIRST(A))" << endl;
        for (int len = 2; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_NESTED_MAP_" << depth << "_" << len << "(MAC, SEP, A) ";
            fout << "MAC(DYND_PP_FIRST(A)) DYND_PP_ID SEP DYND_PP_NESTED_MAP_";
            fout << depth << "_" << len - 1 << "(MAC, SEP, DYND_PP_POP_FIRST(A))" << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_REDUCE DYND_PP_NESTED_REDUCE(0)" << endl;
    fout << "#define DYND_PP_NESTED_REDUCE(DEPTH) DYND_PP_PASTE(DYND_PP_NESTED_REDUCE_, DEPTH)" << endl;
    for (int depth = 0; depth <= pp_depth_max; depth++) {
        fout << "#define DYND_PP_NESTED_REDUCE_" << depth << "(MAC, A) DYND_PP_PASTE(DYND_PP_NESTED_REDUCE_" << depth << "_, DYND_PP_LEN(A))(MAC, A)" << endl;
        fout << "#define DYND_PP_NESTED_REDUCE_" << depth << "_2(MAC, A) MAC(DYND_PP_FIRST(A), DYND_PP_FIRST(DYND_PP_POP_FIRST(A)))" << endl;
        for (int len = 3; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_NESTED_REDUCE_" << depth << "_" << len << "(MAC, A) ";
            fout << "DYND_PP_NESTED_REDUCE_" << depth << "_" << len - 1 << "(MAC, (MAC(DYND_PP_FIRST(A), ";
            fout << "DYND_PP_FIRST(DYND_PP_POP_FIRST(A))), DYND_PP_ID DYND_PP_POP_FIRST(DYND_PP_POP_FIRST(A))))" << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_ELEMENTWISE(MAC, A, B) DYND_PP_PASTE(DYND_PP_ELEMENTWISE_, DYND_PP_LEN(A))(MAC, A, B)" << endl;
    fout << "#define DYND_PP_ELEMENTWISE_1(MAC, A, B) (MAC(DYND_PP_FIRST(A), DYND_PP_FIRST(B)))" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ELEMENTWISE_" << i << "(MAC, A, B) DYND_PP_PREPEND(MAC(DYND_PP_FIRST(A), DYND_PP_FIRST(B)), DYND_PP_ELEMENTWISE_" << i - 1;
        fout << "(MAC, DYND_PP_POP_FIRST(A), DYND_PP_POP_FIRST(B)))" << endl;
    }

    fout << endl;

    for (char c = 'A'; c <= 'Z'; c++) {
        fout << "#define DYND_PP_" << c << "_EQ_" << c << endl;
    }

    fout << endl;

    for (char c = 'a'; c <= 'z'; c++) {
        fout << "#define DYND_PP_" << c << "_EQ_" << c << endl;
    }

    fout << endl;

    for (char c = '0'; c <= '9'; c++) {
        fout << "#define DYND_PP_" << c << "_EQ_" << c << endl;
    }
    for (int i = 10; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_" << i << "_EQ_" << i << endl;
    }

    fout << endl;

    fout << "#endif" << endl;

    fout.close();

    return 0;
}
