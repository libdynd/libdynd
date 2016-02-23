The DyND ND::Array
==================

The DyND nd::array is a multidimensional data storage container, inspired
by the NumPy ndarray and based on the Blaze datashape system. Like NumPy,
it supports strided multidimensional arrays of data with a uniform
data type, but has the ability to store ragged arrays and data types
with variable-sized data.


Every DyND type is able to store an arbitrary, but fixed amount, of metadata in `nd::array` object.
This is a place to store information that doesn't really belong in the type, but is also necessary
to fully describe the data encapsulted by the `nd::array`.

Scalar Types
------------

DyND types that are scalars consist
