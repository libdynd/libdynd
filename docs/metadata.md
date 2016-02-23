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

| Type          | Metadata                                    |
| ------------- |:-------------------------------------------:|
| bool          | None.                                       |
| int8          |                                             |
| int16         |                                             |    
| int32         |                                             |
| int64         |                                             |
| int128        |                                             |
| uint8         |                                             |
| uint16        |                                             |    
| uint32        |                                             |
| uint64        |                                             |
| uint128       |                                             |
| float16       |                                             |    
| float32       |                                             |
| float64       |                                             |
| float128      |                                             |
| complex_float32       |                                             |
| complexfloat64       |                                             |


```
struct fixed_dim_type::metadata_type {
  size_t size;
  intptr_t stride;
};

struct var_dim_type::metadata_type {
  intrusive_ptr<memory_block_data> blockref; // A reference to the memory block which contains the array's data.
  intptr_t stride;
  intptr_t offset; // Each pointed-to destination is offset by this amount
};

struct pointer_type::metadata_type {
  intrusive_ptr<memory_block_data> blockref; // A reference to the memory block which contains the array's data.
  intptr_t offset; // Each pointed-to destination is offset by this amount
};
````
