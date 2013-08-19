Blaze/DyND CKernels
===================

The initial versions of DyND began with an iterator approach,
similar to nditer in NumPy. With the addition of the variable-sized
dimension type 'var', it became clear that this does not generalize
cleanly to handle broadcasting and other operations on 'var' dimensions
nicely. The solution was to define a new kernel mechanism, which
is described here.

See also the [Multi-dimensional kernel documentation](multidim_kernels.md].

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

    include/dynd/kernels/ckernel_builders.hpp
    include/dynd/kernels/assignment_kernels.hpp

A DyND kernel is a block of memory which contains at its start
a kernel function pointer and a destructor. The class defining
these members is `ckernel_prefix`, and the class which
manages the memory for creating such a kernel is
`ckernel_builder`. Here's the ckernel_prefix structure:

```cpp
struct ckernel_prefix {
    typedef void (*destructor_fn_t)(ckernel_prefix *);

    void *function;
    destructor_fn_t destructor;
};
```

The most basic kernel is the single assignment kernel, which
assigns one element of data from input data to output data.
Here's the function prototype:

```cpp
/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src,
                ckernel_prefix *extra);
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
    assignment_ckernel_builder k;
    make_assignment_kernel(&k, 0,
                    dst_dt, dst_metadata,
                    src_dt, src_metadata,
                    kernel_request_single,
                    assign_error_fractional, &eval::default_eval_context);
    k(dst_data, src_data);
}
```

Note that the 'assignment_kernel' object is a subclass of
'ckernel_builder', which overloads the call operator to
provide a more convenient kernel function call syntax.

### Assignment Kernel Example

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
                    ckernel_prefix *extra);
    /** Typedef for a unary operation on a strided segment of elements */
    typedef void (*unary_strided_operation_t)(
                    char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra);

    struct strided_int32_copy_kernel_data {
        // The first ckernel_prefix
        unary_single_operation_t main_kernel_func;
        destructor_fn_t main_kernel_destructor;
        // Data for the first kernel
        intptr_t size;
        intptr_t dst_stride, src_stride;
        // The second ckernel_prefix
        unary_strided_operation_t child_kernel_func;
        destructor_fn_t child_kernel_destructor;
    };

    void main_kernel_func_implementation(char *dst, const char *src,
                        ckernel_prefix *extra)
    {
        // Cast the kernel data to the right type, then call the child strided
        // kernel with the correct values
        strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)extra;
        unary_strided_operation_t opchild = e->child_kernel_func;
        opchild(dst, e->dst_stride, src, e->src_stride, e->size, &e->child_kernel_func);
    }

    void main_kernel_destructor_implementation(
                    ckernel_prefix *extra)
    {
        // Call the destructor of the nested child kernel
        strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)extra;
        if (e->child_kernel_destructor) {
            e->child_kernel_destructor(&e->child_kernel_func);
        }
    }

    void child_kernel_func_implementation(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *DYND_UNUSED(extra))
    {
        for (size_t i = 0; i != count; ++i,
                        dst += dst_stride, src += src_stride) {
            *(int32_t *)dst = *(int32_t *)src;
        }
    }
```

Constructing Kernels
--------------------

Kernels are constructed by ensuring the 'ckernel_builder'
output object has enough space, building any child kernels,
and populating the needed kernel data. For efficiency
and correctness reasons, there are a number of restrictions
on how data is stored in the kernel.

### Kernel Data Restrictions

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

### Kernel Construction Cautions

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
destructor function pointer is set, memory for the 'ckernel_prefix'
of the child kernel must already be allocated and initialized
to zero. This is handled automatically by the 'ensure_capacity'
function of the 'ckernel_builder'.

If a kernel is a leaf, i.e. it terminates any hierarchy in the
chain, it should use the 'ensure_capacity_leaf' function instead
of 'ensure_capacity', to avoid overallocation of space for a child
'ckernel_prefix'.

### Trivial Leaf Kernel Construction Pattern

This is the simplest case, where all information about the
data is included in the static function pointer. This is how
the assignment kernels for builtin types look. Here's a
hypothetical kernel factory for int32.

```cpp
static void single_assign(char *dst, const char *src,
                ckernel_prefix *DYND_UNUSED(extra))
{
    *(int32_t *)dst = *(const int32_t *)src;
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
{
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        *(int32_t *)dst = *(const int32_t *)src;
    }
}

size_t make_int32_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                kernel_request_t kernreq)
{
    // No additional space needs to be allocated for this
    // leaf kernel, because the minimal 'ckernel_prefix'
    // is always preallocated by the parent kernel.
    ckernel_prefix *result;
    result = out->get_at<ckernel_prefix>(offset_out);

    // Set the appropriate function based on the type of kernel requested
    switch (kernreq) {
        case kernel_request_single:
            result->set_function<unary_single_operation_t>(&single_assign);
            break;
        case kernel_request_strided:
            result->set_function<unary_strided_operation_t>(&strided_assign);
            break;
        default:
            throw runtime_error("...");
    }

    // Return the offset immediately after this kernel's data
    return offset_out + sizeof(ckernel_prefix);
}
```

### Leaf Kernel Construction

When a leaf kernel needs some additional data, it must also ensure
the kernel it's creating has enough buffer capacity. Here's a
hypothetical kernel factory for unaligned data assignment.

```cpp
// This is the data for the kernel. It starts with a
// ckernel_prefix, then has fields for other data.
// Remember, this data must be movable by using a memcpy,
// which is trivially true in this case.
struct unaligned_kernel_extra {
    ckernel_prefix base;
    size_t data_size;
};

static void single_assign(char *dst, const char *src,
                ckernel_prefix *extra)
{
    unaligned_kernel_extra *e = reinterpret_cast<unaligned_kernel_extra *>(extra);
    size_t data_size = e->data_size;
    memcpy(dst, src, data_size);
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *extra)
{
    unaligned_kernel_extra *e = reinterpret_cast<unaligned_kernel_extra *>(extra);
    size_t data_size = e->data_size;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        memcpy(dst, src, data_size);
    }
}

size_t make_unaligned_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                size_t data_size,
                kernel_request_t kernreq)
{
    // Allocate the necessary space in the output kernel buffer
    out->ensure_capacity_leaf(offset_out + sizeof(unaligned_kernel_extra));
    unaligned_kernel_extra *result;
    result = out->get_at<unaligned_kernel_extra>(offset_out);

    // Set the appropriate function based on the type of kernel requested
    switch (kernreq) {
        case kernel_request_single:
            result->base.set_function<unary_single_operation_t>(&single_assign);
            break;
        case kernel_request_strided:
            result->base.set_function<unary_strided_operation_t>(&strided_assign);
            break;
        default:
            throw runtime_error("...");
    }
    // Set the kernel data
    result->data_size = data_size;

    // Return the offset immediately after this kernel's data
    return offset_out + sizeof(unaligned_kernel_extra);
}
```

### Leaf Kernel Construction With Dynamic Resources

The final kind of leaf kernels are ones with dynamically-allocated
resources they are responsible for. The example we use is a kernel
created by a JIT engine. On destruction, the kernel must free
its reference to the resources holding the JITted function.

```cpp
// A hypothetical JIT API
void create_jit_assignment(const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq,
                void **out_function_ptr,
                void **out_jit_handle)
void jit_free(void *jit_handle);

// This is the data for the kernel. It starts with a
// ckernel_prefix, then has fields for other data.
// Remember, this data must be movable by using a memcpy,
// which is trivially true in this case.
struct jit_kernel_extra {
    ckernel_prefix base;
    void *jit_handle;
};

static void destruct(ckernel_prefix *extra)
{
    jit_kernel_extra *e = reinterpret_cast<jit_kernel_extra *>(extra);
    if (e->jit_handle != NULL) {
        jit_free(e->jit_handle);
    }
}

size_t make_jit_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                kernel_request_t kernreq)
{
    // Allocate the necessary space in the output kernel buffer
    out->ensure_capacity_leaf(offset_out + sizeof(jit_kernel_extra));
    jit_kernel_extra *result;
    result = out->get_at<jit_kernel_extra>(offset_out);

    // Always set the destructor first, so that if things go wrong
    // later, partially constructed resources are freed.
    result->base.destructor = &destruct;

    // Set the function and the jit handle at once.
    // Assume the jit_assignment function raises an exception on error
    create_jit_assignment(dst_dt, dst_metadata, src_dt, src_metadata, kernreq,
                    &result->base.function,
                    &result->jit_handle);

    // Return the offset immediately after this kernel's data
    return offset_out + sizeof(jit_kernel_extra);
}
```

### Simple Nested Kernel Construction

Many DyND types are composed of other types, and constructing
kernels for them typically involves composing kernels from
their subtypes. The simplest case of this is a single child
kernel. For this example, we show how an assignment from a
pointer<T1> to T2 can be constructed.

```cpp

static void destruct(ckernel_prefix *extra)
{
    ckernel_prefix *echild = extra + 1;
    if (echild->destructor) {
        echild->destructor(echild);
    }
}

static void single_assign(char *dst, const char *src,
                ckernel_prefix *extra)
{
    ckernel_prefix *echild = extra + 1;
    unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();
    opchild(dst, *(const char **)src, echild);
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *src, intptr_t src_stride,
                        size_t count, ckernel_prefix *extra)
{
    ckernel_prefix *echild = extra + 1;
    unary_single_operation_t opchild = echild->get_function<unary_single_operation_t>();
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src += src_stride) {
        opchild(dst, *(const char **)src, echild);
    }
}

size_t make_ptr_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_target_dt, const char *src_target_metadata,
                kernel_request_t kernreq)
{
    ckernel_prefix *result;
    result = out->get_at<ckernel_prefix>(offset_out);

    // Always set the destructor first, so that if things go wrong
    // later, partially constructed resources are freed.
    result->destructor = &destruct;

    // Set the appropriate function based on the type of kernel requested
    switch (kernreq) {
        case kernel_request_single:
            result->set_function<unary_single_operation_t>(&single_assign);
            break;
        case kernel_request_strided:
            result->set_function<unary_strided_operation_t>(&strided_assign);
            break;
        default:
            throw runtime_error("...");
    }

    // Construct the child assignment kernel, and
    // return the offset immediately after the child kernel's data
    return make_assignment_kernel(out, offset_out + sizeof(ckernel_prefix),
                    dst_dt, dst_metadata,
                    src_target_dt, src_target_metadata,
                    kernel_request_single,
                    assign_error_default, &eval::default_eval_context);
}
```

Memory Allocation During Kernel Creation
----------------------------------------

The hierarchical kernel design is created in such a way that if
used from C++, simple assignments will trigger no heap memory
allocations at all. There is still the expense of the virtual
function calls and if/switch statements figuring out the kernel
to create, but only using stack-allocated memory is possible.

This is handled by a trick inside the 'ckernel_builder' class,
which handles memory allocation and resizing of the kernel data
buffer. This class contains a small fixed-size array which is
used to start. If no kernel factory uses more than this amount
of memory, then no memory is allocated on the heap.
