#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "unary_gfunc.hpp" namespace "pydnd":
    cdef cppclass unary_gfunc:
        string& get_name()
        void add_kernel(object) except +
        object call(object, object) except +
        string debug_dump() except +

    string unary_gfunc_repr(unary_gfunc&) except +
    string unary_gfunc_debug_dump(unary_gfunc&) except +

    cdef struct unary_gfunc_placement_wrapper:
        pass
    void unary_gfunc_placement_new(unary_gfunc_placement_wrapper&, char *)
    void unary_gfunc_placement_delete(unary_gfunc_placement_wrapper&)
    # ndarray placement cast
    unary_gfunc& GET(unary_gfunc_placement_wrapper&)
