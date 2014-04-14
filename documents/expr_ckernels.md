# Expression CKernels

Expression ckernels accept multiple input
memory addresses as an input, and write to a
single memory address, with no state
shared between multiple calls of the ckernel.
The function pointer for such a ckernel
looks like this:

```
# include/dynd/kernels/expr_kernels.hpp

/** Typedef for an expression operation on a single element */
typedef void (*expr_single_operation_t)(char *dst, const char *const *src,
                                        ckernel_prefix *extra);
/** Typedef for an expression operation on a strided segment of elements */
typedef void (*expr_strided_operation_t)(char *dst, intptr_t dst_stride,
                                         const char *const *src,
                                         const intptr_t *src_stride,
                                         size_t count, ckernel_prefix *extra);
```

The number of source addresses is baked into the ckernel
prior to getting to this low level of representation,
but is available at the deferred ckernel level.
