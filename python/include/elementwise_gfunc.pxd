#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "elementwise_gfunc.hpp" namespace "pydnd":
    cdef cppclass elementwise_gfunc:
        string& get_name()
        void add_kernel(codegen_cache&, object) except +
        object call(object, object) except +
        string debug_dump() except +

    string elementwise_gfunc_repr(elementwise_gfunc&) except +
    string elementwise_gfunc_debug_dump(elementwise_gfunc&) except +

    cdef struct elementwise_gfunc_placement_wrapper:
        pass
    void elementwise_gfunc_placement_new(elementwise_gfunc_placement_wrapper&, char *)
    void elementwise_gfunc_placement_delete(elementwise_gfunc_placement_wrapper&)
    # ndarray placement cast
    elementwise_gfunc& GET(elementwise_gfunc_placement_wrapper&)
