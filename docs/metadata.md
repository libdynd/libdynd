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

| Type                  | Metadata
| --------------------- |:------------------------------------------------------:|
| bool                  | None
| int8                  | None
| int16                 | None
| int32                 | None
| int64                 | None
| int128                | None
| uint8                 | None
| uint16                | None
| uint32                | None
| uint64                | None
| uint128               | None
| float16               | None
| float32               | None
| float64               | None
| float128              | None
| complex_float32       | None
| complex_float64       | None
| void                  | None
| fixed_bytes           | None
| bytes*                | None
| char                  | None
| fixed_string          | None
| string*               | None
| tuple                 | Offsets (in bytes) to the data of each element, *i.e.* `uintptr_t[N]` for a tuple with `N` fields
| struct                | Same as for tuple
| fixed_dim             | The size and stride of the dimension, *i.e.* a `size_t` and an `intptr_t`.
| var_dim               | The stride of the dimension and a reference to which `nd::memory_block` contains the data, *i.e.* an `intptr_t` and a `intrusive_ptr<nd::memory_block>`
| categorical           | None
| option                | None
| pointer               | A reference to which `nd::memory_block` contains the data, *i.e.* a `intrusive_ptr<nd::memory_block>`
| memory                | None

* The bytes and string types *used* to have metadata that was exactly a reference to an `nd::memory_block`
that contained their data. This was used to allow views into bytes and strings that were allocated elsewhere.
While more generic, this model made string processing far too complicated, so we removed it.




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
