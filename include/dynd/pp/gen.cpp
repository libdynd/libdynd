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

// if a macro mixes named and variable arguments, proxy it through another macro with eval
// need eval still in a few places

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
    if (pp_int_max < 7) {
        throw runtime_error("the maximum integer cannot be less than 7");
    }

    const int pp_len_max = pp_int_max + 1;

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

    fout << "#define DYND_PP_MSC_EVAL(A) A" << endl;

    fout << endl;

    fout << "#define DYND_PP_NULL(...)" << endl;
    fout << "#define DYND_PP_ID(...) __VA_ARGS__" << endl;

    fout << endl;

    fout << "#define DYND_PP_TO_ZERO(...) 0" << endl;
    fout << "#define DYND_PP_TO_ONE(...) 1" << endl;
    fout << "#define DYND_PP_TO_COMMA(...) ," << endl;

    fout << "#define DYND_PP_TO_NULL(...)" << endl;
    fout << "#define DYND_PP_TO_EMPTY(...) ()" << endl;

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
    for (int i = 0; i < pp_len_max; i++) {
        fout << 0 << argsep(i);
    }
    fout << 0 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_ONES_" << pp_len_max << " (";
    for (int i = 0; i < pp_len_max; i++) {
        fout << 1 << argsep(i);
    }
    fout << 1 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_HEAD(A) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(A))(DYND_PP_TO_NULL)(DYND_PP__HEAD)(DYND_PP_ID A)" << endl;
    fout << "#define DYND_PP__HEAD(...) DYND_PP_EVAL(DYND_PP___HEAD(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP___HEAD(A0, ...) A0" << endl;

    fout << endl;

    fout << "#define DYND_PP_TAIL(A) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(A))(DYND_PP_TO_EMPTY)(DYND_PP__TAIL)(DYND_PP_ID A)" << endl;
    fout << "#define DYND_PP__TAIL(...) DYND_PP_EVAL(DYND_PP___TAIL(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP___TAIL(A0, ...) (__VA_ARGS__)" << endl;

    fout << "#define DYND_PP_REST DYND_PP_TAIL" << endl;

    fout << endl;

//    fout << "#define DYND_PP_GET(INDEX, A) DYND_PP_CAT_2(DYND_PP_GET_, INDEX)(A)" << endl;
    for (int i = 0; i < 8; i++) {
        fout << "#define DYND_PP_GET_" << i << "(A) DYND_PP__GET_" << i << "(DYND_PP_ID A)" << endl;
        fout << "#define DYND_PP__GET_" << i << "(" << args("A", i + 1) << argsep(false) << "...) A" << i << endl;
    }
    fout << "#define DYND_PP_GET_" << pp_len_max << "(...) DYND_PP_MSC_EVAL(DYND_PP__GET_" << pp_len_max << "(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP__GET_" << pp_len_max << "(" << args("A", pp_len_max + 1) << ", ...) A" << pp_len_max << endl;

    fout << endl;

    fout << "#define DYND_PP_LEN_IF_NOT_EMPTY(A) DYND_PP_GET_" << pp_len_max << "(DYND_PP_ID A" << argsep(true);
    for (int i = 0; i < pp_len_max; i++) {
        fout << pp_len_max - i << argsep(i);
    }
    fout << 0 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_LEN(A) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(A))(DYND_PP_TO_ZERO)(DYND_PP_LEN_IF_NOT_EMPTY)(A)" << endl;

    fout << endl;

    fout << "#define DYND_PP_HAS_COMMA(...) DYND_PP_GET_" << pp_len_max << "(__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp_len_max - 1; i++) {
        fout << 1 << argsep(i);
    }
    fout << 0 << argsep(false) << 0 << ")" << endl;

    fout << endl;

    fout << "#define DYND_PP_CAT(A) DYND_PP_CAT_2(DYND_PP_CAT_, DYND_PP_LEN(A))(A)" << endl;
    fout << "#define DYND_PP_CAT_0 DYND_PP__CAT_0" << endl;
    fout << "#define DYND_PP__CAT_0()" << endl;
    fout << "#define DYND_PP_CAT_1 DYND_PP__CAT_1" << endl;
    fout << "#define DYND_PP__CAT_1(A) A" << endl;
    fout << "#define DYND_PP_CAT_2(A, B) DYND_PP__CAT_2(A, B)" << endl;
    fout << "#define DYND_PP__CAT_2(A, B) A ## B" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_CAT_" << i << "(FIRST" << argsep(false) << "...) DYND_PP_CAT_2(FIRST";
        fout << argsep(false) << "DYND_PP_CAT_" << i - 1 << "(__VA_ARGS__))" << endl;
    }

    fout << endl;

    for (int i = 0; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_EQ_" << i << "_" << i << " ," << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_INC(A) DYND_PP_CAT_2(DYND_PP_INC_, A)" << endl;
    for (int i = 0; i < pp_len_max; i++) {
        fout << "#define DYND_PP_INC_" << i << " " << i + 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_DEC(A) DYND_PP_CAT_2(DYND_PP_DEC_, A)" << endl;
    for (int i = 1; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_DEC_" << i << " " << i - 1 << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_FROM(START, A) DYND_PP_CAT_2(DYND_PP_SLICE_FROM_, START)(A)" << endl;
    fout << "#define DYND_PP_SLICE_FROM_0 DYND_PP_ID" << endl;
    fout << "#define DYND_PP_SLICE_FROM_1 DYND_PP_TAIL" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_FROM_" << i << "(A) DYND_PP_SLICE_FROM_" << i - 1 << "(DYND_PP_TAIL(A))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_TO(STOP" << argsep(false) << "A) DYND_PP_CAT_2(DYND_PP_SLICE_TO_" << argsep(false) << "STOP)(A)" << endl;
    fout << "#define DYND_PP_SLICE_TO_0(A) ()" << endl;
    fout << "#define DYND_PP_SLICE_TO_1(A) (DYND_PP_HEAD(A))" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_TO_" << i << "(A) DYND_PP_MERGE((DYND_PP_HEAD(A))";
        fout << argsep(false) << "DYND_PP_SLICE_TO_" << i - 1 << "(DYND_PP_TAIL(A)))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_WITH(STEP, A) DYND_PP_CAT_4(DYND_PP_SLICE_WITH_, DYND_PP_LEN(A), _, STEP)(A)" << endl;
    for (int i = 1; i <= pp_len_max; i++) {
        for (int j = 1; j <= pp_len_max; j++) {
            fout << "#define DYND_PP_SLICE_WITH_" << i << "_" << j << "(A) ";
            if (i - j >= j) {
                fout << "DYND_PP_PREPEND(DYND_PP_FIRST(A)";
                fout << argsep(false) << "DYND_PP_SLICE_WITH_" << i - j << "_" << j << "(DYND_PP_SLICE_FROM(" << j - 1 << argsep(false) << "DYND_PP_TAIL(A))))";
            } else {
                fout << "(DYND_PP_FIRST(A))";
            }
            fout << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_REPEAT(TOK, COUNT) DYND_PP_CAT_2(DYND_PP_REPEAT_, COUNT)(TOK)" << endl;
    fout << "#define DYND_PP_REPEAT_0(TOK)" << endl;
    fout << "#define DYND_PP_REPEAT_1(TOK) TOK" << endl;
    fout << "#define DYND_PP_REPEAT_2(TOK) TOK, TOK" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_REPEAT_" << i << "(TOK) TOK, DYND_PP_REPEAT_" << i - 1 << "(TOK)" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_MAP(MAC, SEP, A) DYND_PP_CAT_2(DYND_PP_MAP_, DYND_PP_LEN(A))(MAC, SEP, A)" << endl;
    fout << "#define DYND_PP_MAP_0(MAC" << argsep(false) << "SEP, A)" << endl;
    fout << "#define DYND_PP_MAP_1(MAC, SEP, A) MAC(DYND_PP_FIRST(A))" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_MAP_" << i << "(MAC, SEP, A) MAC(DYND_PP_FIRST(A)) DYND_PP_ID SEP DYND_PP_MAP_";
        fout << i - 1 << "(MAC, SEP, DYND_PP_TAIL(A))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_REDUCE(MAC, A) DYND_PP_CAT_2(DYND_PP_REDUCE_, DYND_PP_LEN(A))(MAC, A)" << endl;
    fout << "#define DYND_PP_REDUCE_2(MAC, A) MAC(DYND_PP_HEAD(A), DYND_PP_HEAD(DYND_PP_TAIL(A)))" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_REDUCE_" << i << "(MAC, A) DYND_PP_REDUCE_" << i - 1 << "(MAC, (MAC(DYND_PP_HEAD(A), DYND_PP_HEAD(DYND_PP_TAIL(A))), DYND_PP_ID DYND_PP_TAIL(DYND_PP_TAIL(A))))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_ELEMENTWISE(MAC, A, B) DYND_PP_CAT_2(DYND_PP_ELEMENTWISE_, DYND_PP_LEN(A))(MAC, A, B)" << endl;
    fout << "#define DYND_PP_ELEMENTWISE_1(MAC, A, B) (MAC(DYND_PP_FIRST(A), DYND_PP_FIRST(B)))" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ELEMENTWISE_" << i << "(MAC, A, B) DYND_PP_PREPEND(MAC(DYND_PP_FIRST(A), DYND_PP_FIRST(B)), DYND_PP_ELEMENTWISE_" << i - 1;
        fout << "(MAC, DYND_PP_REST(A), DYND_PP_REST(B)))" << endl;
    }

    fout << endl;

    fout << "#endif" << endl;

    fout.close();

    return 0;
}
