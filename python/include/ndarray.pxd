
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

        # Cython bug: operator overloading doesn't obey "except +"
        # TODO: Report this bug
        # ndarray operator+(ndarray&) except +
        #ndarray operator-(ndarray&) except +
        #ndarray operator*(ndarray&) except +
        #ndarray operator/(ndarray&) except +

        dtype& get_dtype()
        int get_ndim()
        intptr_t* get_shape()
        intptr_t* get_strides()
        intptr_t get_num_elements()

        char* get_originptr()

        void val_assign(ndarray&, assign_error_mode) except +
        void val_assign(dtype&, char*, assign_error_mode) except +

        ndarray as_dtype(dtype&, assign_error_mode) except +

        void debug_dump(ostream&)

cdef extern from "ndarray_functions.hpp" namespace "pydnd":
    string ndarray_str(ndarray&) except +
    string ndarray_repr(ndarray&) except +
    string ndarray_debug_dump(ndarray&)

    void ndarray_init(ndarray&, object obj) except +
    ndarray ndarray_vals(ndarray& n) except +

    ndarray ndarray_add(ndarray&, ndarray&) except +
    ndarray ndarray_subtract(ndarray&, ndarray&) except +
    ndarray ndarray_multiply(ndarray&, ndarray&) except +
    ndarray ndarray_divide(ndarray&, ndarray&) except +


