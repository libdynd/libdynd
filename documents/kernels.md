DyND Kernels
============

The initial versions of DyND began with an iterator approach,
similar to nditer in NumPy. With the addition of the variable-sized
dimension type 'var', it became clear that this does not generalize
cleanly to handle broadcasting and other operations on 'var' dimensions
nicely. The solution was to define a new kernel mechanism, which
is described here.

Headers and implementation for kernels are in the 'dynd/kernels'
subdirectories.

Current Limitations
-------------------

The biggest limitation of the kernel mechanism that it's not
threadsafe. In particular, with kernels needing intermediate buffers
for chaining the execution of other kernels. Supporting this properly
will likely require a significant change to the interface, for example
the kernels might receive a 'temporary space' buffer as another argument.

Kernel Basics
-------------

    include/dynd/kernels/hierarchical_kernels.hpp
    include/dynd/kernels/assignment_kernels.hpp

A DyND kernel is a block of memory which contains at its start
a kernel function pointer and a destructor. The class defining
these members is 'kernel_data_prefix', and the class which
manages the memory for creating such a kernel is
'hierarchical_kernel'.

The most basic kernel is the single assignment kernel, which
assigns one element of data from input data to output data.
Here's the function prototype:

```cpp
/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                kernel_data_prefix *extra);
```

The kernel is constructed using full knowledge of the input and
output memory layout, so to execute the assignment only requires
the source and destination memory, along with the kernel data itself.
Likely another parameter will be added for a temporary buffer, to
make things thread safe. Here's some example code which does one
assignment using this mechanism.

```cpp
void default_assign(const dtype& dst_dt, const char *dst_metadata, char *dst_data,
                const dtype& src_dt, const char *src_metadata, const char *src_data)
{
    assignment_kernel k;
    make_assignment_kernel(&k, 0,
                    dst_dt, dst_metadata,
                    src_dt, src_metadata,
                    kernel_request_single,
                    assign_error_fractional, &eval::default_eval_context);
    k(dst_data, src_data);
}
```

Note that the 'assignment_kernel' object is a subclass of
'hierarchical_kernel', which overloads the call operator to
provide a more convenient kernel function call syntax.

Assignment Kernel Example
-------------------------

Kernels are constructed hierarchically, typically matching
the hierarchical structure of the dtypes it is operating
on. To illustrate how this works, we'll work out the memory
structure and functions created dynamically for creating
the assignment of a strided integer array. We've included
some of the function prototypes from the dynd headers
to provide some more context.

```cpp
    /** Typedef for a unary operation on a single element */
    typedef void (*unary_single_operation_t)(char *dst, const char *src,
                    kernel_data_prefix *extra);
    /** Typedef for a unary operation on a strided segment of elements */
    typedef void (*unary_strided_operation_t)(
                    char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, kernel_data_prefix *extra);

    struct strided_int32_copy_kernel_data {
        // The first kernel_data_prefix
        unary_single_operation_t main_kernel_func;
        destructor_fn_t main_kernel_destructor;
        // Data for the first kernel
        intptr_t size;
        intptr_t dst_stride, src_stride;
        // The second kernel_data_prefix
        unary_strided_operation_t child_kernel_func;
        destructor_fn_t child_kernel_destructor;
    };

    void main_kernel_func_implementation(char *dst, const char *src,
                        kernel_data_prefix *extra)
    {
        // Cast the kernel data to the right type, then call the child strided
        // kernel with the correct values
        strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)extra;
        unary_strided_operation_t opchild = e->child_kernel_func;
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, &e->child_kernel_func);
    }

    void main_kernel_destructor_implementation(
                    kernel_data_prefix *extra)
    {
        // Call the destructor of the nested child kernel
        strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)extra;
        if (e->child_kernel_destructor) {
            e->child_kernel_destructor(&e->child_kernel_func);
        }
    }

    void child_kernel_func_implementation(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, kernel_data_prefix *DYND_UNUSED(extra))
    {
        for (size_t i = 0; i != count; ++i,
                        dst += dst_stride, src += src_stride) {
            *(int32_t *)dst = *(int32_t *)src;
        }
    }
```

Kernel Data Restrictions
------------------------

A kernel may store nearly anything in its data, subject
to one major constraint: it must be movable with
memcpy/memmove. This means, for example, that the kernel
data must not contain a pointer to another field within
the kernel data. This must be accomplished by using an
offset instead.

The data movability constraint allows the container
class for construction to use the C realloc() function
call when increasing the size of the kernel buffer.
Normal C++ semantics do not permit this.

The code constructing the kernel may assume that all
kernel memory is zero-initialized. This is done to allow
for safe destruction of partially-constructed kernels
when an exception is raised during kernel construction.

Kernel Construction Cautions
----------------------------

Some care must be taken by code which is constructing
kernels. First, the construction and destruction must
be crafted to account for exceptions. Second, any time
additional space is requested for the kernel buffer,
any pointers the kernel constructor has into the kernel
buffer must be retrieved again, as the buffer may have moved.

Because nearly any step of the construction may
raise an exception, the kernel must always be in a state
where calling its destructor will successfully free all
allocated resources without crashing on some uninitialized
memory.

This is done by relying on the fact that all
the kernel memory is zero-initialized. The destructor should
check any value it is deallocating for NULL, and the constructor
should populate the kernel in an order which would never
cause the destructor to point at uninitialized memory.

If a kernel uses a child kernel for part of its operation,
its destructor must call that child kernel's destructor if
it is not NULL. What this means is that, before the kernel's
destructor function pointer is set, memory for the 'kernel_data_prefix'
of the child kernel must already be allocated and initialized
to zero. This is handled automatically by the 'ensure_capacity'
function of the 'hierarchical_kernel'.

If a kernel is a leaf, i.e. it terminates any hierarchy in the
chain, it should use the 'ensure_capacity_leaf' function instead
of 'ensure_capacity', to avoid overallocation of space for a child
'kernel_data_prefix'.

Leaf Kernel Construction Pattern
--------------------------------

