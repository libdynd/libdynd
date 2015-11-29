# Comparison CKernels

Comparison ckernels accept two memory addresses, and
return a boolean value. They provide a low level
interface with which to build algorithms that require
element comparisons, like sorting for example.

```
# include/dynd/kernels/comparison_kernels.hpp

/**
 * Typedef for a binary predicate (comparison) on a single element.
 * The predicate function type uses 'int' instead of 'bool' so
 * that it is a well-specified C function pointer.
 */
typedef int (*binary_single_predicate_t)(const char *src0, const char *src1,
                                         ckernel_prefix *extra);
```
