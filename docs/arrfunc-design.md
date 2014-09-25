# High Level Design of DyND Array Functions (ArrFuncs)

DyND arrfuncs represent array computations in DyND.
Everything from data copying, basic arithmetic, to
indexing and reduction fit into the arrfunc model.

## Requirements

* Be able to run the function repeatedly on
  identically laid out memory in an efficient
  manner. This is the most core requirement,
  from DyND being an array programming system.
* C ABI-level access, no C++-specific
  requirements. This is to permit JIT-based systems
  to fully publish and consume arrfuncs without
  too much complexity.
* Both *array parameters* and *dynamic parameters*.
  Array parameters contain many instances of data,
  while dynamic parameters usually do not, and may affect
  the result type or be used to choose more efficient
  kernel code.
* Ability to specify read-write parameters, for
  example in functions with multiple accumulators, or
  something like in-place ``sort``.
* Some functions need to return views into an input
  array instead of creating fully new data. The indexing
  operation is an example of this.

## Design Outline

The arrfunc is an object that goes inside an nd::array
container, with type ``arrfunc``. In C++, it's typically
held in an nd::arrfunc object, which has some
convenience interface for making function calls and other
things.

Execution always occurs through a two step process:

1. (optional) Resolve the output type.
2. Instantiate a ckernel.
3. Run the ckernel on raw data, as many times as desired.

We'll run through these steps in more detail.

### Output Type Resolution

Many functions' output type can be represented by a
function prototype expressed as datashape with
type variables. In these cases, the output type can
be produced by pattern matching the function
prototype against all the input types, then
substituting in to the output.

For example, with the prototype
``(int32, int32) -> int64``, the output type
will always be ``int64``. With the prototype
``(T, T, bool) -> ?T``, the output type will be an
optional version of the first two arguments.

Not all output types resolutions may be represented
in this simple fashion, and in some cases it should
be dependent on the values of dynamic parameters.
For these cases, an arrfunc can define a type
resolution function:

```
int resolve_dst_type(const arrfunc_type_data *self,
                     intptr_t nsrc,
                     const ndt::type *src_tp,
                     const char *const *src_arrmeta,
                     const nd::array &dyn_params,
                     ndt::type *out_dst_tp);
```

In earlier versions of the arrfunc design, this
type resolution step didn't have access to the
input arrmeta, and the output shape was populated
by another function. In the current design, the
output type must be constructable without requiring
additional shape information.

The reason this is an optional step is for calling
arrfuncs with an "out=" argument. In that case, the
output type is the type of that output array, and
it is instantiate's job to determine whether it is
compatible or not.

### CKernel Instantiation

There are two modes in which one may call the ckernel
instantiation, depending on whether the output is
being generated in a way which might view the input
data. First is instantiation without allowing views.

```
intptr_t instantiate(const arrfunc_type_data *self,
                     dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                     const ndt::type &dst_tp, const char *dst_arrmeta,
                     const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq,
                     const nd::array &dyn_params,
                     const eval::eval_context *ectx);
```

Second is instantiation allowing for views, by having instantiate
populate the output arrmeta. In this version, the output type
should have been created by ``resolve_dst_type``, and an empty shell
nd::array created with blank arrmeta to be filled.

```
intptr_t instantiate(const arrfunc_type_data *self,
                     dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                     const ndt::type &dst_tp, const char *dst_arrmeta,
                     const ndt::type *src_tp, const char *const *src_arrmeta,
                     memory_block_data *const *src_data_refs,
                     kernel_request_t kernreq,
                     const nd::array &dyn_params,
                     const eval::eval_context *ectx,
                     int require_view,
                     memory_block_data **out_dst_data_ref,
                     char *out_dst_metadata,
                     char **out_dst_data_ptr);
```

