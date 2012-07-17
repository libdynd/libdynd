#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "elwise_reduce_gfunc.hpp" namespace "pydnd":
    cdef cppclass elwise_reduce_gfunc:
        string& get_name()
        void add_kernel(codegen_cache&, object, bint, bint) except +
        object call(object, object) except +
        string debug_dump() except +

    string elwise_reduce_gfunc_repr(elwise_reduce_gfunc&) except +
    string elwise_reduce_gfunc_debug_dump(elwise_reduce_gfunc&) except +

    cdef struct elwise_reduce_gfunc_placement_wrapper:
        pass
    void elwise_reduce_gfunc_placement_new(elwise_reduce_gfunc_placement_wrapper&, char *)
    void elwise_reduce_gfunc_placement_delete(elwise_reduce_gfunc_placement_wrapper&)
    # ndarray placement cast
    elwise_reduce_gfunc& GET(elwise_reduce_gfunc_placement_wrapper&)
