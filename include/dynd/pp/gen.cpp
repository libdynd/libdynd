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

    fout << "#define DYND_PP_TO_COMMA(...) ," << endl;

    fout << endl;

    fout << "#define DYND_PP_INT_MAX " << pp_int_max << endl;
    fout << "#define DYND_PP_LEN_MAX " << pp_len_max << endl;

    fout << endl;

    fout << "#define DYND_PP_INTS ";
    for (int i = 0; i < pp_int_max; i++) {
        fout << i << argsep(i);
    }
    fout << pp_int_max << endl;

    fout << endl;

    fout << "#define DYND_PP_ZEROS_" << pp_len_max << " ";
    for (int i = 0; i < pp_len_max; i++) {
        fout << 0 << argsep(i);
    }
    fout << 0 << endl;

    fout << endl;

    fout << "#define DYND_PP_ONES_" << pp_len_max << " ";
    for (int i = 0; i < pp_len_max; i++) {
        fout << 1 << argsep(i);
    }
    fout << 1 << endl;

    fout << endl;

    fout << "#define DYND_PP_GET_" << pp_len_max << "(" << args("A", pp_len_max + 1) << ", ...) A" << pp_len_max << endl;

    fout << endl;

    fout << "#define DYND_PP_LEN_NONZERO(...) DYND_PP_MSC_EVAL(DYND_PP_GET_" << pp_len_max << "(__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp_len_max; i++) {
        fout << pp_len_max - i << argsep(i);
    }
    fout << 0 << "))" << endl;

    fout << endl;

    fout << "#define DYND_PP_HAS_COMMA(...) DYND_PP_MSC_EVAL(DYND_PP_GET_" << pp_len_max << "(__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp_len_max - 1; i++) {
        fout << 1 << argsep(i);
    }
    fout << 0 << argsep(false) << 0 << "))" << endl;

    fout << endl;

    fout << "#define DYND_PP_CAT(...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(DYND_PP_CAT_, DYND_PP_ID(DYND_PP_LEN(__VA_ARGS__)))(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP_CAT_0()" << endl;
    fout << "#define DYND_PP_CAT_1(A) A" << endl;
    fout << "#define DYND_PP_CAT_2(A, B) DYND_PP__CAT_2(A, B)" << endl;
    fout << "#define DYND_PP__CAT_2(A, B) A ## B" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_CAT_" << i << "(HEAD" << argsep(false) << "...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(HEAD";
        fout << argsep(false) << "DYND_PP_CAT_" << i - 1 << "(__VA_ARGS__)))" << endl;
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

    fout << "#define DYND_PP_SLICE_FROM(INDEX, ...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(DYND_PP_SLICE_FROM_, INDEX)(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP_SLICE_FROM_0(...) __VA_ARGS__" << endl;
    fout << "#define DYND_PP_SLICE_FROM_1(HEAD, ...) __VA_ARGS__" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_FROM_" << i << "(HEAD, ...) DYND_PP_MSC_EVAL(DYND_PP_SLICE_FROM_" << i - 1 << "(__VA_ARGS__))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_SLICE_TO(INDEX" << argsep(false) << "...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(DYND_PP_SLICE_TO_";
    fout << argsep(false) << "INDEX)(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP_SLICE_TO_0(...)" << endl;
    fout << "#define DYND_PP_SLICE_TO_1(HEAD" << argsep(false) << "...) HEAD" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_SLICE_TO_" << i << "(HEAD" << argsep(false) << "...) HEAD";
        fout << argsep(false) << "DYND_PP_MSC_EVAL(DYND_PP_SLICE_TO_" << i - 1 << "(__VA_ARGS__))" << endl;
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

    fout << "#define DYND_PP_MAP(MAC, SEP, ...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(DYND_PP_MAP_, DYND_PP_LEN(__VA_ARGS__))(MAC, SEP, __VA_ARGS__))" << endl;
    fout << "#define DYND_PP_MAP_0(MAC, SEP)" << endl;
    fout << "#define DYND_PP_MAP_1(MAC, SEP, HEAD) MAC(HEAD)" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_MAP_" << i << "(MAC, SEP, HEAD, ...) MAC(HEAD) DYND_PP_ID SEP DYND_PP_MSC_EVAL(DYND_PP_MAP_";
        fout << i - 1 << "(MAC, SEP, __VA_ARGS__))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_REDUCE(MAC, ...) DYND_PP_MSC_EVAL(DYND_PP_CAT_2(DYND_PP_REDUCE_, DYND_PP_MSC_EVAL(DYND_PP_LEN(__VA_ARGS__)))(MAC, __VA_ARGS__))" << endl;
    fout << "#define DYND_PP_REDUCE_2(MAC, FIRST, SECOND) MAC(FIRST, SECOND)" << endl;
    for (int i = 3; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_REDUCE_" << i << "(MAC, FIRST, SECOND, ...) DYND_PP_MSC_EVAL(DYND_PP_REDUCE_" << i - 1 << "(MAC, MAC(FIRST, SECOND), __VA_ARGS__))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_ZIP(A, B) DYND_PP_CAT_2(DYND_PP_ZIP_, DYND_PP_LEN A)(A, B)" << endl;
    fout << "#define DYND_PP_ZIP_1(A, B) (DYND_PP_FIRST A, DYND_PP_FIRST B)" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ZIP_" << i << "(A, B) (DYND_PP_FIRST A, DYND_PP_FIRST B), DYND_PP_ZIP_" << i - 1;
        fout << "((DYND_PP_SLICE_FROM(1, DYND_PP_ID A)), (DYND_PP_SLICE_FROM(1, DYND_PP_ID B)))" << endl;
    }

    fout << endl;

    fout << "#endif" << endl;

    fout.close();

    return 0;
}
