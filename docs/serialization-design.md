# Serialization

DyND stores typed multi-dimensional data in memory, which needs to be saved to disk and transported between machines for things like distributed and out of core computation. We would like the serialization to be simple and obvious where possible, and well-defined in a way suitable for use as a building block for creating Merkle DAGs.

What follows is a proposal for how this can be done, to get things started and provide a basis for design discussion.

## Document/Design Scope

Ultimately, we may want to add a streaming capability to DyND, so that a DyND array which is almost as big as memory could stream directly to a file without being placed in an intermediate bytes object. In initial design and implementation of serialization, we can simply use the `bytes` type without hampering such future extension.

We are not concerned with deduplication, as that would complicate the serialization format a fair bit. Deduplication can be handled by a future extension to a new type like `array[Any]`, but which operates as a Merkle link, together with serialization to/from a content addressable storage (CAS).

We are not concerned with compression, that is something that can be layered on top of serialization. If a stream interface is added to DyND, chaining compression to serialization via that interface would be natural to pursue.

As a serialization protocol, this document is concerned with converting to/from byte streams, and  not with random access or efficient seeking. In some cases, random access is possible, however.

## Serialize Callable

The `serialize` callable has signature `(Any) -> bytes`, and operates as a reduction kernel. This means it concatenates data to the output bytes object. Only the data gets serialized, not the type. This is not a problem, however, because to serialize an arbitrary array, one can serialize it as `array[Any]`, which must serialize both the type and the data.

In DyND's storage within memory, there is some metadata which describes memory layout, data references, and other details incidental to how things are stored. For serialization, the layout is defined completely in terms of the type, and there is no such metadata. For example, the `fixed` dimension type has a stride in its metadata. In the serialization of a strided array, the data is always tightly packed, and no stride is stored anywhere.

## Deserialize Callable

The `deserialize` callable has signature `(bytes, dst_tp: type) -> Any`, where the output array has the specified destination type. The generic case of of deserializing a general DyND array is treated as an element of type `array[Any]`, as noted in the serialize section above.

## Type to Bytes Mapping

A fundamental principle used to design this mapping is that the serialization of particular data for a particular type must be unique. This is not the case in DyND's storage within memory, where there is some metadata which additionally describes memory layout and some other details. The reason for this design choice is to allow this serialization to be a building block in CAS/Merkle DAG technologies.

One choice made to remove this ambiguity is that all primitives are serialized in little-endian order. A majority of platforms are little-endian, so this is also good for the typical case.

A consequence of this choice is that multi-dimensional arrays always serialize in "C order". This is great for deduplication of data, but does have some performance consequence when working with "F order" arrays. I (Mark) view the support of the CAS/Merkle use case to be strong enough to not change the unique serialization principle, and propose doing something within the type system layered on top of the simple core serialization definition for cases where the performance is more critical. For example, a type which represents a storage axis permutation could do this in general, like `storage[3 * 5 * 7 * float64, axis=[2, 1, 0]]` or `storage[3 * 5 * 7 * float64, order='F']` could represent F order.

* `bool`
  * Single byte `0x00` (false) `0x01` (true)
* `int##`, `uint##`, `float##`
  * The bytes are packed in little endian order.
* `complex[T]`
  * The real and imaginary components are serialized one after the other.
* `bytes[N]`
  * The bytes are stored as is.
* `bytes`
  * The size is stored as a [base 128 Varint](https://developers.google.com/protocol-buffers/docs/encoding?hl=en#varints), and the bytes follow as is.
* `string`
  * The UTF-8 encoding of the string is serialized identically to `bytes`.
* `char`
  * The code point is serialized as its UTF-8 bytes. This means arrays of `char` serialize into a UTF-8 representation.
* `type`
  * The type represented as a string, serialized as `string`. (TODO: Need to specify a well-defined type to string serialization that we will never change.)
* `void`
  * Adds no bytes to the serialization.
* `N * T`
  * Each element is serialized in sequence (a packed contiguous serialization). Because `N` is part of the type, it does not need to be stored.
* `var * T`
  * The size is stored as a [base 128 Varint](https://developers.google.com/protocol-buffers/docs/encoding?hl=en#varints), and then the elements are serialized the same as `N * T`.
* `(T0, T1, ...)`, `{field0: T0, field1: T1, ...}`
  * Each element of the tuple/struct is serialized in sequence (a packed contiguous serialization).
* `?T`
  * For types where the missing data sentinel is well-defined, the data is serialized just as `T` would be. For other types, missing is serialized as a single byte `0x00`, and available data is serialized as a single byte `0x01` followed by the serialization as `T`.
* `pointer[T]`
  * Serialized exactly as `T` would be.
* `array[T]`
  * Serialized as if it were a tuple `(type, T)`, with the type serialized, immediately followed by the data.

## Possible DyND Array Serialization As CBOR

Some of the serialization could be as [CBOR](http://cbor.io/), through the mechanism described in http://tools.ietf.org/html/rfc7049#section-7.2. This is not thought out properly yet. An idea for this is to serialize `array[T]` as CBOR, storing as straight CBOR in the simple cases, and falling back to a DyND type + DyND serialization as CBOR bytes in the general case. This requires carefully defining when CBOR native types are used, and a canonical form so as to keep the unique serialization property.
