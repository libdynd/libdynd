The DyND Array Metadata
=======================

The DyND `nd::array` is a multidimensional data storage container, inspired
by the NumPy `ndarray` and based on the Blaze `Datashape` grammar. Like NumPy,
it supports strided multidimensional arrays of data with a uniform data type,
but has the ability to store ragged arrays and data types with variable-sized data.

Every DyND type is able to store an arbitrary, but fixed amount, of metadata in the `nd::array` object.
This is really just a block of bytes (very specifically, a `char *`) that is allocated alongside the `nd::array`, but interpreted and manipulated by the `ndt::type`. The array metadata is the place to store information that doesn't really belong in the type, but is also necessary to fully describe the data encapsulated by the `nd::array`. A classical example of such information is the stride for a dimension type.

In DyND, we typically put most of the ty

| Datashape (DyND Type) | Low-Level Descriptor (DyND Metadata)
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
| type                  | None
| callable              | None
| array                 | None

* The bytes and string types *used* to have metadata that was a reference to a `nd::memory_block` that contained their data. This was necessary to allow views into bytes and strings that were allocated elsewhere. While more generic, that model made string processing far too complicated, so we removed it.

Note that the `intrusive_ptr<T>` template mentioned above is a custom smart pointer in DyND based off of `boost::intrusive_ptr<T>`. In C++, it enables smart reference counting with very simple semantics, analogous to `std::shared_ptr<T>` in C++11 but with smaller storage overhead. In the context of a C ABI, an `intrusive_ptr<T>` is equivalent to a `T *`.
