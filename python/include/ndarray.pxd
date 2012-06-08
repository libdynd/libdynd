
cdef extern from "dnd/ndarray.hpp" namespace "dnd":
    cdef cppclass ndarray:
        ndarray() except +
        ndarray(signed char value)
        ndarray(short value)
        ndarray(int value)
        ndarray(long value)
        ndarray(long long value)
        ndarray(unsigned char value)
        ndarray(unsigned short value)
        ndarray(unsigned int value)
        ndarray(unsigned long value)
        ndarray(unsigned long long value)
        ndarray(float value)
        ndarray(double value)
        ndarray(complex[float] value)
        ndarray(complex[double] value)
        ndarray(dtype&)
        ndarray(dtype, int, intptr_t *, int *)

        ndarray operator+(ndarray&)
        ndarray operator-(ndarray&)
        ndarray operator*(ndarray&)
        ndarray operator/(ndarray&)

        dtype& get_dtype()
        int get_ndim()
        intptr_t* get_shape()
        intptr_t* get_strides()
        intptr_t get_num_elements()

        char* get_originptr()

        void val_assign(ndarray&, assign_error_mode)
        void val_assign(dtype&, char*, assign_error_mode)

        ndarray as_dtype(dtype&, assign_error_mode)

        void debug_dump(ostream&)

cdef extern from "ndarray_functions.hpp" namespace "pydnd":
    string ndarray_str(ndarray&)
    string ndarray_repr(ndarray&)
    string ndarray_debug_dump(ndarray&)

    void ndarray_init(ndarray&, object obj) except +
    void ndarray_init(ndarray&, object obj, dtype&) except +
    ndarray ndarray_vals(ndarray& n) except +

    ndarray ndarray_add(ndarray&, ndarray&) except +

