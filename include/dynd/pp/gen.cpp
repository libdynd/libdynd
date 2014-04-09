//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

#include <algorithm>

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


inline string id(const string name) {
    return name;
}

inline string one(const string) {
    return "1";
}

string pp_first(const string name) {
    ostringstream oss;
    oss << "DYND_PP_FIRST(" << name << ")";
    return oss.str();
}

string pp_pop_first(const string name) {
    ostringstream oss;
    oss << "DYND_PP_POP_FIRST(" << name << ")";
    return oss.str();
}

string pp_len(const string name) {
    ostringstream oss;
    oss << "DYND_PP_LEN(" << name << ")";
    return oss.str();
}

string pp_inc_len(const string name) {
    ostringstream oss;
    oss << "DYND_PP_INC(DYND_PP_LEN(" << name << "))";
    return oss.str();
}

string pp_dec_len(const string name) {
    ostringstream oss;
    oss << "DYND_PP_DEC(DYND_PP_LEN(" << name << "))";
    return oss.str();
}

string pp_shuffle(const string name) {
    ostringstream oss;
    oss << "DYND_PP_APPEND(DYND_PP_FIRST(" << name << "), DYND_PP_POP_FIRST(" << name << "))";
    return oss.str();
}

string cat(const string &name, int i) {
    ostringstream oss;
    oss << name << i;
    return oss.str();
}


string repeat(const string& prefix, string (*func)(const string), const string &sep, int start, int stop, int step) {
    ostringstream oss;
    for (int i = start; i < stop - 1; i += step) {
        oss << (*func)(prefix) << sep;
    }
    oss << (*func)(prefix);
    return oss.str();
}

string args(const string& prefix, string (*func)(const string), const string &sep, int start, int stop, int step) {
    ostringstream oss;
    if (start < stop) {
        oss << (*func)(cat(prefix, start));
    }
    for (int i = start + step; i < stop; i += step) {
        oss << sep << (*func)(cat(prefix, i));
    }
    return oss.str();
}

template <typename iter>
string args(const string& prefix, string (*func)(const string), const string &sep, iter first, iter last) {
    ostringstream oss;    
    if (first != last) {
        oss << (*func)(cat(prefix, *first));
    }
    first++;
    while (first != last) {
        oss << sep << (*func)(cat(prefix, *first));
        ++first;
    }
    return oss.str();
}

string args(const string& prefix, int stop) {
    return args(prefix, &id, ", ", 0, stop, 1);
}

string args(const string& prefix, const string sep, int stop) {
    return args(prefix, &id, sep, 0, stop, 1);
}

template <typename iter>
string args(const string& prefix, iter first, iter last) {
    return args(prefix, &id, ", ", first, last);
}

template <typename iter>
string args(const string& prefix, const string sep, iter first, iter last) {
    return args(prefix, &id, sep, first, last);
}

string repeat(const string& prefix, int stop) {
    return repeat(prefix, &id, ", ", 0, stop, 1);
}

string repeat(const string& prefix, const string sep, int stop) {
    return repeat(prefix, &id, sep, 0, stop, 1);
}

string args(const string& prefix, const string sep, int start, int stop, int step) {
    return args(prefix, &id, sep, start, stop, step);
}

string args(const string& prefix, string (*func)(const string), const string sep, int stop) {
    return args(prefix, func, sep, 0, stop, 1);
}

string args(const string& prefix, string (*func)(const string), const string sep, int start, int stop) {
    return args(prefix, func, sep, start, stop, 1);
}

string args(const string& prefix, string (*func)(const string), int stop) {
    return args(prefix, func, ", ", 0, stop, 1);
}

template <typename T>
string args(const string& prefix, string (*func)(const string), const string sep, T *flags, int start, int stop) {
    return args(prefix, func, sep, flags, start, stop, 1);
}

int next(int *curr, int len_max, int ndim) {
    const int val_max = len_max - 1;

    int idx = 0;
    reverse(curr, curr + ndim);
    while (curr[idx] == val_max) {
        curr[idx] = 0;
        ++idx;
    }
    ++curr[idx];
    reverse(curr, curr + ndim);

    return idx;
}



template <typename Iter>
std::string join(std::string const& sep, Iter begin, Iter end) {
    std::ostringstream oss;
    if (begin != end) {
        oss << *begin++;
    }
    while (begin != end) {
        oss << sep << *begin++;
    }
    return oss.str();
}

int main(int argc, char **argv) {
    const int pp_len_max = atoi(argv[1]);
    const int pp__len_max = 2 * pp_len_max;
    if (pp_len_max < 8) {
        throw runtime_error("the maximum list length cannot be less than 8");
    }

    const int pp_int_max = pp_len_max - 1;

    const int pp_dep_max = atoi(argv[2]);
    if (pp_dep_max < 3) {
        throw runtime_error("the maximum macro depth cannot be less than 3");
    }

    const int pp_ary_max = 4;

    int *range = new int[pp_ary_max];
    for (int i = 0; i < pp_ary_max; i++) {
        range[i] = i;
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

    fout << "#ifndef _DYND__PP_GEN_HPP_" << endl;
    fout << "#define _DYND__PP_GEN_HPP_" << endl;

    fout << endl;

    fout << "#define DYND_PP_INT_MAX " << pp_int_max << endl; // delete this

    fout << "#define DYND_PP_LEN_MAX " << pp_len_max;
    fout << " // The maximum list length that is fully supported" << endl;
    fout << "#define DYND_PP_DEP_MAX " << pp_dep_max;
    fout << " // The maximum macro depth that is fully supported" << endl;

    fout << endl;

    fout << "#define DYND_PP_INTS (";
    for (int i = 0; i < pp_int_max; i++) {
        fout << i << argsep(i);
    }
    fout << pp_int_max << ")" << endl;

    fout << endl;

    fout << "/**" << endl;
    fout << " * Expands to whatever is at 2 * DYND_PP_LEN_MAX in __VA_ARGS__." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_GET_ARG_" << pp__len_max << "(...) DYND_PP_ID(DYND_PP__GET_ARG_" << pp__len_max << "(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP__GET_ARG_" << pp__len_max << "(";
    for (int i = 0; i < pp__len_max + 1; i++) {
        fout << "A" << i << argsep(i);
    }
    fout << "...) A" << pp__len_max << endl;

    fout << endl;

    fout << "/**" << endl;
    fout << " * Expands to 1 if __VA_ARGS__ has a comma. Otherwise 0." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_HAS_COMMA(...) DYND_PP_ID(DYND_PP__HAS_COMMA(__VA_ARGS__))" << endl;
    fout << "#define DYND_PP__HAS_COMMA(...) DYND_PP_GET_ARG_" << pp__len_max << "(__VA_ARGS__" << argsep(true);
    for (int i = 0; i < pp__len_max - 1; i++) {
        fout << 1 << argsep(i);
    }
    fout << 0 << argsep(pp__len_max - 1) << 0 << ")" << endl;

    fout << endl;

    fout << "/**" << endl;
    fout << " * Expands to the length of A. A has to be a list with at most 2 * DYND_PP_LEN_MAX tokens." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_LEN(A) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(A))(DYND_PP_LEN_IF_EMPTY)(DYND_PP_LEN_IF_NOT_EMPTY)(A)" << endl;
    fout << "#define DYND_PP_LEN_IF_EMPTY(A) 0" << endl;
    fout << "#define DYND_PP_LEN_IF_NOT_EMPTY(A) DYND_PP_GET_ARG_" << pp__len_max << "(DYND_PP_ID A" << argsep(true);
    for (int i = 0; i < pp__len_max - 1; i++) {
        fout << pp__len_max - i << argsep(i);
    }
    fout << 1 << ")" << endl;

    fout << endl;

    fout << "/**" << endl;
    fout << " * Increments A. A has to be an integer between 0 and DYND_PP_LEN_MAX - 1 inclusively." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_INC(A) DYND_PP_PASTE(DYND_PP_INC_, A)" << endl;
    for (int i = 0; i < pp_len_max; i++) {
        fout << "#define DYND_PP_INC_" << i << " " << i + 1 << endl;
    }

    fout << endl;

    fout << "/**" << endl;
    fout << " * Decrements A. A has to be an integer between 1 and DYND_PP_LEN_MAX inclusively." << endl;
    fout << " */" << endl;
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

    fout << "/**" << endl;
    fout << " * Expands to a list filled with 0. N has to be an integer between 0 and DYND_PP_LEN_MAX inclusively." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_ZEROS(N) DYND_PP_PASTE(DYND_PP_ZEROS_, N)" << endl;
    fout << "#define DYND_PP_ZEROS_0 ()" << endl;
    for (int i = 1; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ZEROS_" << i << " (";
        for (int j = 0; j < i - 1; j++) {
            fout << 0 << argsep(j);
        }
        fout << 0 << ")" << endl;
    }

    fout << endl;

    fout << "/**" << endl;
    fout << " * Expands to a list filled with 1. N has to be an integer between 0 and DYND_PP_LEN_MAX inclusively." << endl;
    fout << " */" << endl;
    fout << "#define DYND_PP_ONES(N) DYND_PP_PASTE(DYND_PP_ONES_, N)" << endl;
    fout << "#define DYND_PP_ONES_0 ()" << endl;
    for (int i = 1; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ONES_" << i << " (";
        for (int j = 0; j < i - 1; j++) {
            fout << 1 << argsep(j);
        }
        fout << 1 << ")" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_REPEAT DYND_PP_REPEAT_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_REPEAT_" << dep << "(TOK, LEN)";
        fout << " ";
        fout << "(DYND_PP_JOIN_REPEAT_" << dep << "(TOK, (,), LEN))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_JOIN_REPEAT DYND_PP_JOIN_REPEAT_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_JOIN_REPEAT_" << dep << "(TOK, SEP, LEN)";
        fout << " ";
        fout << "DYND_PP_PASTE(DYND_PP_JOIN_REPEAT_" << dep << "_, LEN)(TOK, SEP)" << endl;
        fout << "#define DYND_PP_JOIN_REPEAT_" << dep << "_0(TOK, SEP)" << endl;
        fout << "#define DYND_PP_JOIN_REPEAT_" << dep << "_1(TOK, SEP) TOK" << endl;
        for (int len = 2; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_JOIN_REPEAT_" << dep << "_" << len << "(TOK, SEP)";
            fout << " TOK DYND_PP_ID SEP ";
            fout << "DYND_PP_JOIN_REPEAT_" << dep << "_" << len - 1 << "(TOK, SEP)" << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_MAP DYND_PP_MAP_0" << endl;
    for (int dep = 0; dep <= pp_dep_max; dep++) {
        fout << "#define DYND_PP_MAP_" << dep << "(MAC, A) (DYND_PP_JOIN_MAP_" << dep << "(MAC, (,), A))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_JOIN_MAP DYND_PP_JOIN_MAP_0" << endl;
    for (int depth = 0; depth <= pp_dep_max; depth++) {
        fout << "#define DYND_PP_JOIN_MAP_" << depth << "(MAC, SEP, A) ";
        fout << "DYND_PP_PASTE(DYND_PP_JOIN_MAP_" << depth << "_ , DYND_PP_LEN(A))(MAC, SEP, A)" << endl;
        fout << "#define DYND_PP_JOIN_MAP_" << depth << "_" << "0(MAC, SEP, A)" << endl;
        fout << "#define DYND_PP_JOIN_MAP_" << depth << "_" << "1(MAC, SEP, A) MAC(DYND_PP_FIRST(A))" << endl;
        for (int len = 2; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_JOIN_MAP_" << depth << "_" << len << "(MAC, SEP, A) ";
            fout << "MAC(DYND_PP_FIRST(A)) DYND_PP_ID SEP DYND_PP_JOIN_MAP_";
            fout << depth << "_" << len - 1 << "(MAC, SEP, DYND_PP_POP_FIRST(A))" << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_REDUCE DYND_PP_REDUCE_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_REDUCE_" << dep << "(MAC, A0)";
        fout << " ";
        fout << "DYND_PP_PASTE(DYND_PP_REDUCE_" << dep << "_, DYND_PP_LEN(A0))(MAC, A0)" << endl;
        fout << "#define DYND_PP_REDUCE_" << dep << "_1(MAC, A0)";
        fout << " ";
        fout << "DYND_PP_FIRST(A0)" << endl;
        fout << "#define DYND_PP_REDUCE_" << dep << "_2(MAC, A0)";
        fout << " ";
        fout << "MAC(DYND_PP_FIRST(A0), DYND_PP_FIRST(DYND_PP_POP_FIRST(A0)))" << endl;
        for (int len = 3; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_REDUCE_" << dep << "_" << len << "(MAC, A0)";
            fout << " ";
            fout << "DYND_PP_REDUCE_" << dep << "_" << len - 1 << "(MAC, DYND_PP_PREPEND(MAC(DYND_PP_FIRST(A0), ";
            fout << "DYND_PP_FIRST(DYND_PP_POP_FIRST(A0))), DYND_PP_POP_FIRST(DYND_PP_POP_FIRST(A0))))" << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_ELWISE DYND_PP_ELWISE_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_ELWISE_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP__ELWISE_" << dep << "(__VA_ARGS__))" << endl;
        fout << "#define DYND_PP__ELWISE_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP_PASTE(DYND_PP__ELWISE_" << dep << "_, DYND_PP_DEC(DYND_PP_LEN((__VA_ARGS__))))(__VA_ARGS__))" << endl;
        for (int ary = 2; ary <= pp_ary_max; ary++) {
            fout << "#define DYND_PP__ELWISE_" << dep << "_" << ary << "(MAC, " << args("A", ary) << ")";
            fout << " ";
            fout << "(DYND_PP__JOIN_ELWISE_" << dep << "_" << ary << "(MAC, (,), " << args("A", ary) << "))";
            fout << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_JOIN_ELWISE DYND_PP_JOIN_ELWISE_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_JOIN_ELWISE_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP__JOIN_ELWISE_" << dep << "(__VA_ARGS__))" << endl;
        fout << "#define DYND_PP__JOIN_ELWISE_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP_PASTE(DYND_PP__JOIN_ELWISE_" << dep << "_, DYND_PP_DEC(DYND_PP_DEC(DYND_PP_LEN((__VA_ARGS__)))))(__VA_ARGS__))" << endl;
        for (int ary = 2; ary <= pp_ary_max; ary++) {
            fout << "#define DYND_PP__JOIN_ELWISE_" << dep << "_" << ary;
            fout << "(MAC, SEP, " << args("A", ary) << ")";
            fout << " DYND_PP_PASTE(DYND_PP__JOIN_ELWISE_" << dep << "_" << ary << "_, DYND_PP_LEN(A0))";
            fout << "(MAC, SEP, " << args("A", ary) << ")" << endl;
            fout << "#define DYND_PP__JOIN_ELWISE_" << dep << "_" << ary << "_" << 1;
            fout << "(MAC, SEP, " << args("A", ary) << ") MAC(" << args("A", &pp_first, ary) << ")" << endl;
            for (int len = 2; len <= pp_len_max; len++) {
                fout << "#define DYND_PP__JOIN_ELWISE_" << dep << "_" << ary << "_" << len;
                fout << "(MAC, SEP, " << args("A", ary) << ") MAC(" << args("A", &pp_first, ary) << ")";
                fout << " DYND_PP_ID SEP ";
                fout << "DYND_PP__JOIN_ELWISE_" << dep << "_" << ary << "_" << len - 1;
                fout << "(MAC, SEP, " << args("A", &pp_pop_first, ary) << ")" << endl;
            }
        }
    }

    fout << endl;

    fout << "#define DYND_PP_OUTER DYND_PP_OUTER_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_OUTER_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP__OUTER_" << dep << "(__VA_ARGS__))" << endl;
        fout << "#define DYND_PP__OUTER_" << dep << "(...) ";
        fout << "DYND_PP_ID(DYND_PP_PASTE(DYND_PP__OUTER_" << dep << "_, DYND_PP_DEC(DYND_PP_LEN((__VA_ARGS__))))(__VA_ARGS__))" << endl;
        for (int ary = 2; ary <= pp_ary_max; ary++) {
            fout << "#define DYND_PP__OUTER_" << dep << "_" << ary << "(MAC, " << args("A", ary) << ")";
            fout << " ";
            fout << "(DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "(MAC, (,), " << args("A", ary) << "))";
            fout << endl;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_JOIN_OUTER DYND_PP_JOIN_OUTER_0" << endl;
    for (int dep = 0; dep < pp_dep_max; dep++) {
        fout << "#define DYND_PP_JOIN_OUTER_" << dep << "(...)";
        fout << " ";
        fout << "DYND_PP_ID(DYND_PP__JOIN_OUTER_" << dep << "(__VA_ARGS__))" << endl;
        fout << "#define DYND_PP__JOIN_OUTER_" << dep << "(...)";
        fout << " ";
        fout << "DYND_PP_ID(DYND_PP_PASTE(DYND_PP__JOIN_OUTER_" << dep << "_, ";
        fout << "DYND_PP_DEC(DYND_PP_DEC(DYND_PP_LEN((__VA_ARGS__)))))(__VA_ARGS__))" << endl;
        for (int ary = 2; ary <= pp_ary_max; ary++) {
            fout << "#define DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "(MAC, SEP, " << args("A", ary) << ")";
            fout << " DYND_PP_CAT((DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "_, " << args("A", pp_dec_len, ", _, ", ary) << "))(MAC, SEP, " << args("A", ary) << ")" << endl;
            fout << "#define DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "_" << repeat("0", "_", ary);
            fout << "(MAC, SEP, " << args("A", ary) << ") MAC(" << args("A", pp_first, ary) << ")" << endl;
            int *curr = new int[ary];
            fill(curr, curr + ary, 0);
            for (int cnt = (int) pow((double) pp_len_max, ary) - 1; cnt > 0; cnt--) {
                int *prev = new int[ary];
                for (int i = 0; i < ary; i++) {
                    prev[i] = curr[i];
                }
//                copy(curr, curr + ary, prev);
                int idx = next(curr, pp_len_max, ary);
                fout << "#define DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "_" << args("", "_", curr, curr + ary);
                fout << "(MAC, SEP, " << args("A", ary) << ") MAC(" << args("A", &pp_first, ary) << ")";
                fout << " DYND_PP_ID SEP ";
                if (idx > 0) {
                    fout << "DYND_PP_CAT((";
                }
                fout << "DYND_PP__JOIN_OUTER_" << dep << "_" << ary << "_" << join("_", prev, prev + ary - idx);
                if (idx > 0) {
                    fout << ", _, ";
                }
                fout << args("A", &pp_dec_len, ", _, ", ary - idx, ary);
                if (idx > 0) {
                    fout << "))";
                }
                fout << "(MAC, SEP" << args(", A", "", ary - idx - 1);
                fout << ", " << args("A", &pp_shuffle, ", ", ary - idx - 1, ary) << ")" << endl;
                delete prev;
            }
            delete curr;
        }
    }

    fout << endl;

    fout << "#define DYND_PP_ARRAY_DIMS(A) DYND_PP_PASTE(DYND_PP_ARRAY_DIMS_, DYND_PP_LEN(A))(DYND_PP_ID(A))" << endl;
    fout << "#define DYND_PP_ARRAY_DIMS_1(A) [DYND_PP_FIRST(A)]" << endl;
    for (int i = 2; i <= pp_len_max; i++) {
        fout << "#define DYND_PP_ARRAY_DIMS_" << i << "(A) [DYND_PP_FIRST(A)]";
        fout << "DYND_PP_ARRAY_DIMS_" << i - 1 << "(DYND_PP_POP_FIRST(A))" << endl;
    }

    fout << endl;

    fout << "#define DYND_PP_JOIN DYND_PP_JOIN_0" << endl;
    for (int depth = 0; depth <= pp_dep_max; depth++) {
        fout << "#define DYND_PP_JOIN_" << depth << "(SEP, A) ";
        fout << "DYND_PP_PASTE(DYND_PP_JOIN_" << depth << "_ , DYND_PP_LEN(A))(SEP, A)" << endl;
        fout << "#define DYND_PP_JOIN_" << depth << "_" << "0(SEP, A)" << endl;
        fout << "#define DYND_PP_JOIN_" << depth << "_" << "1(SEP, A) DYND_PP_FIRST(A)" << endl;
        for (int len = 2; len <= pp_len_max; len++) {
            fout << "#define DYND_PP_JOIN_" << depth << "_" << len << "(SEP, A) ";
            fout << "DYND_PP_FIRST(A) DYND_PP_ID SEP DYND_PP_JOIN_";
            fout << depth << "_" << len - 1 << "(SEP, DYND_PP_POP_FIRST(A))" << endl;
        }
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
