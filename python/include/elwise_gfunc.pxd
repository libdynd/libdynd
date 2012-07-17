#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "elwise_gfunc.hpp" namespace "pydnd":
    cdef cppclass elwise_gfunc:
        string& get_name()
        void add_kernel(codegen_cache&, object) except +
        object call(object, object) except +
        string debug_dump() except +

    string elwise_gfunc_repr(elwise_gfunc&) except +
    string elwise_gfunc_debug_dump(elwise_gfunc&) except +

    cdef struct elwise_gfunc_placement_wrapper:
        pass
    void elwise_gfunc_placement_new(elwise_gfunc_placement_wrapper&, char *)
    void elwise_gfunc_placement_delete(elwise_gfunc_placement_wrapper&)
    # ndarray placement cast
    elwise_gfunc& GET(elwise_gfunc_placement_wrapper&)
