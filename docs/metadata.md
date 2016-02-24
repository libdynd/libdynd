Low-Level Datashape and DyND Array Metadata
===========================================

The DyND `nd::array` is a multidimensional data storage container, inspired
by the NumPy `ndarray` and based on the Blaze `Datashape` grammar. Like NumPy,
it supports strided multidimensional arrays of data with a uniform data type,
but has the ability to store ragged arrays and data types with variable-sized data.

Every DyND type is able to store an arbitrary, but fixed amount, of metadata in the `nd::array` object.
This is really just a block of bytes (very specifically, a `char *`) that is allocated alongside the `nd::array`, but interpreted and manipulated by the `ndt::type`. The array metadata is the place to store information that doesn't really belong in the type, but is necessary to fully describe the data encapsulated by the `nd::array`. A classic example of such information is the stride for a dimension type.

At construction, a DyND type (dynamic or otherwise) forwards its total metadata size to the `ndt::base_type` constructor. Its metadata typically consists of whatever particular metadata the type itself needs and the metadata of any child types. For example, the metadata for `10 * int32` has a size that is the size of the metadata for a `ndt::fixed_dim_type` (namely 16 bytes that represents a size and a stride) and the size of the
metadata for a `int32` (exactly 0 bytes). Analogously, the metadata for a type `20 * 10 * int32` is 32 bytes (a size, a stride, a size, and a stride in that order).

The API for the metadata consists of a set of virtual functions in `ndt::base_type` that are overridable
by any derived type. These are basically for construction, copy-construction, and destruction of the metadata.
A C API could easily be provided for these without any issues.

```
  /**
   * Returns The size of the metadata for this type.
   */
  size_t ndt::base_type::get_metadata_size() const;

  /**
   * Constructs the metadata for this type using default settings.
   *
   * \param metadata        The metadata to default construct.
   * \param blockref_alloc  If ``true``, blockref types should allocate
   *                        writable memory blocks, and if ``false``, they
   *                        should set their blockrefs to NULL. The latter
   *                        indicates the blockref memory is owned by
   *                        the parent nd::array, and is useful for viewing
   *                        external memory with compatible layout.
   */
  virtual void ndt::base_type::metadata_default_construct(char *metadata, bool blockref_alloc) const;

  /**
   * Constructs the metadata for this type, copying everything exactly from input metadata for the same type.
   *
   * \param dst_metadata        The new metadata memory which is constructed.
   * \param src_metadata        Existing arrmeta memory from which to copy.
   * \param embedded_reference  For references which are NULL, add this reference in the output.
   *                            A NULL means the data was embedded in the original nd::array, so
   *                            when putting it in a new nd::array, need to hold a reference to
   *                            that memory.
   */
  virtual void ndt::base_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata,
                                                       const nd::array *embedded_reference) const;

  /** Destructs any references or other state contained in the metadata */
  virtual void ndt::base_type::metadata_destruct(char *metadata) const;
```

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
