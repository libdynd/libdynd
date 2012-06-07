include "dnd.pxd"

from cython.operator import dereference

cdef class w_dtype:
    # To access the embedded dtype, use "dpc(self.d)",
    # which returns a reference to the dtype.
    cdef dtype_placement_wrapper d

    def __cinit__(self):
        dtype_placement_new(self.d)
    def __dealloc__(self):
        dtype_placement_delete(self.d)

    def printval(self):
        dtype_print(dpc(self.d))

cdef class w_ndarray:
    # To access the embedded dtype, use "npc(self.n)",
    # which returns a reference to the ndarray
    cdef ndarray_placement_wrapper n

    def __cinit__(self):
        ndarray_placement_new(self.n)
    def __dealloc__(self):
        ndarray_placement_delete(self.n)

    def debug_dump(self):
        npc(self.n).debug_dump(cout)