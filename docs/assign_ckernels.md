# Assignment CKernels

Assignment ckernels assign one input
memory address to another, with no state
shared between multiple calls of the ckernel.
The function pointer for such a ckernel
looks like this:

```
# include/dynd/kernels/assignment_kernels.hpp

/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                                         ckernel_prefix *extra);
/** Typedef for a unary operation on a strided segment of elements */
typedef void (*unary_strided_operation_t)(char *dst, intptr_t dst_stride,
                                          const char *src, intptr_t src_stride,
                                          size_t count, ckernel_prefix *extra);
```