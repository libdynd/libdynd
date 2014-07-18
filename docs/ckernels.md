# Blaze/DyND CKernels

The initial versions of DyND began with an iterator approach,
similar to the nditer object in NumPy. With the addition of
the variable-sized dimension type ``var``, it became
clear that this does not generalize cleanly to handle
broadcasting and other operations on 'var' dimensions
nicely. The solution was to define a hierarchical kernel
mechanism, which is described here.

* [Assignment CKernels](assign_ckernels.md)
* [Expression CKernels](expr_ckernels.md)
* [Comparison CKernels](compare_ckernels.md)
* [Accumulator CKernels](accum_ckernels.md)
* [Multi-dimensional kernel documentation](multidim_kernels.md)

Headers and implementation for kernels are in the
``dynd/kernels`` subdirectories.

Current Limitations
-------------------

The biggest limitation of the ckernel mechanism that some of
the implemented buffering operations aren't threadsafe.
These should be able to use the stack for smaller buffers,
or use a TLS variable when the buffer is large. The plan
is to add multi-threading to DyND's computation by default
using TBB after fleshing out the front end with
NumPy-like ufuncs.

CKernel Basics
--------------

    include/dynd/kernels/ckernel_builders.hpp
    include/dynd/kernels/assignment_kernels.hpp

A DyND ckernel is a block of memory that contains at its start
a function pointer and a destructor. The ``ckernel_prefix`` class
defines these members, and the class which manages the memory for
creating such a kernel is ``ckernel_builder``. Here's the
ckernel_prefix structure:

```cpp
struct ckernel_prefix {
  typedef void (*destructor_fn_t)(ckernel_prefix *);

  void *function;
  destructor_fn_t destructor;
};
```

The simplest ckernel to implement is an ``expr_single_t``
ckernel, which performs an operation on ``N`` source data
elements, writing the result to a destination. Its
function prototype is:

```cpp
typedef void (*expr_single_t)(char *dst, const char *const *src,
                              ckernel_prefix *self);
```

The ``N`` parameter is baked into the ckernel, so it is not
provided redundantly as a parameter. Similarly, it must
have full knowledge of the input and output memory layout.

Here's some example code which does one assignment using
this mechanism, by building an assignment ckernel and
calling it.

```cpp
void dynd::typed_data_assign(const ndt::type &dst_tp,
                             const char *dst_arrmeta,
                             char *dst_data, const ndt::type &src_tp,
                             const char *src_arrmeta, const char *src_data,
                             const eval::eval_context *ectx)
{
  unary_ckernel_builder k;
  make_assignment_kernel(&k, 0, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
                         kernel_request_single, ectx);
  k(dst_data, src_data);
}
```

The ``unary_ckernel_builder`` object is a subclass of
``ckernel_builder``, which overloads the call operator to
provide a more convenient kernel function call syntax.

### Assignment Kernel Example

CKernels are typically constructed hierarchically, often matching
the hierarchical structure of the types they are operating
on. To illustrate how this works, we'll work out the memory
structure and functions created dynamically for creating
the copying of a strided integer array. We've included
some of the function prototypes from the dynd headers
to provide some more context.

```cpp
typedef void (*expr_single_t)(char *dst, const char *const *src,
                              ckernel_prefix *self);
typedef void (*expr_strided_t)(char *dst, intptr_t dst_stride,
                               const char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *self);

struct strided_int32_copy_kernel_data {
  // The first ckernel_prefix
  expr_single_t main_kernel_func;
  destructor_fn_t main_kernel_destructor;
  // Data for the first kernel
  intptr_t size;
  intptr_t dst_stride, src_stride;

  // The second ckernel_prefix
  expr_strided_t child_kernel_func;
  destructor_fn_t child_kernel_destructor;
};

void main_kernel_func_implementation(char *dst, const char *const *src,
                                     ckernel_prefix *self)
{
  // Cast the kernel data to the right type, then call the child strided
  // kernel with the correct values
  strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)self;
  unary_strided_operation_t child_fn = e->child_kernel_func;
  opchild(dst, e->dst_stride,
          src, e->src_stride,
          e->size,
          (ckernel_prefix *)&e->child_kernel_func);
}

void main_kernel_destructor_implementation(ckernel_prefix *self)
{
  // Call the destructor of the child kernel
  strided_int32_copy_kernel_data *e = (strided_int32_copy_kernel_data *)self;
  if (e->child_kernel_destructor) {
    e->child_kernel_destructor((ckernel_prefix *)&e->child_kernel_func);
  }
}

void child_kernel_func_implementation(char *dst, intptr_t dst_stride,
                const char *const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *DYND_UNUSED(self))
{
  const char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *(int32_t *)dst = *(int32_t *)src0;
    dst += dst_stride;
    src0 += src0_stride;
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

A ckernel may store nearly anything in its data, subject
to one major constraint: it must be movable with
memcpy/memmove. This means, for example, that the ckernel
data must not contain a pointer to another field within
the ckernel data. This must be accomplished by using an
offset instead.

The data movability constraint allows the container
class for construction to use the C realloc() function
call when increasing the size of the ckernel buffer.
Normal C++ semantics do not permit this.

The code constructing the ckernel may assume that all
ckernel memory is zero-initialized. This is done to allow
for safe destruction of partially-constructed ckernels
when an exception is raised during ckernel construction.

### Kernel Construction Cautions

Some care must be taken by code which is constructing
ckernels. 

1. The construction and destruction must be crafted to account
   for exceptions.

2. Any time additional space is requested for the ckernel buffer,
   any pointers the ckernel constructor has into the ckernel
   buffer must be retrieved again, as the buffer may have moved.

Because nearly any step of the construction may
raise an exception, the ckernel must always be in a state
where calling its destructor will successfully free all
allocated resources without crashing on some uninitialized
memory.

This is done by relying on the fact that all
the ckernel memory is zero-initialized. The destructor should
check any value it is deallocating for NULL, and the constructor
should populate the ckernel in an order which would never
cause the destructor to point at uninitialized memory.

If a ckernel uses a child ckernel for part of its operation,
its destructor must call that child ckernel's destructor if
it is not NULL. What this means is that, before the ckernel's
destructor function pointer is set, memory for the 'ckernel_prefix'
of the child ckernel must already be allocated and initialized
to zero. This is handled automatically by the 'ensure_capacity'
function of the 'ckernel_builder'.

If a ckernel is a leaf, i.e. it terminates any hierarchy in the
chain, it should use the 'ensure_capacity_leaf' function instead
of 'ensure_capacity', to avoid overallocation of space for a child
'ckernel_prefix'.

### Trivial Leaf Kernel Construction Pattern

This is the simplest case, where all information about the
data is included in the static function pointer. This is how
the assignment ckernels for builtin types look. Here's a
hypothetical ckernel factory for int32.

```cpp
static void single_assign(char *dst, const char *const *src,
                ckernel_prefix *DYND_UNUSED(self))
{
  *(int32_t *)dst = *(const int32_t *)src[0];
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(self))
{
  const char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    *(int32_t *)dst = *(const int32_t *)src0;
    dst += dst_stride;
    src0 += src0_stride;
  }
}

size_t make_int32_assignment_kernel(ckernel_builder *ckb, size_t ckb_offset,
                                    kernel_request_t kernreq)
{
  // No additional space needs to be allocated for this
  // leaf kernel, because the minimal 'ckernel_prefix'
  // is always preallocated by the parent kernel.
  ckernel_prefix *result;
  result = ckb->get_at<ckernel_prefix>(ckb_offset);

  // Set the appropriate function based on the type of kernel requested
  result->set_expr_function(kernreq, &single_assign, &strided_assign);

  // Return the offset immediately after this kernel's data
  return ckb_offset + sizeof(ckernel_prefix);
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
struct unaligned_assign_ck {
    ckernel_prefix base;
    size_t data_size;
};

static void single_assign(char *dst, const char *const *src,
                          ckernel_prefix *self)
{
  unaligned_assign_ck *e = reinterpret_cast<unaligned_assign_ck *>(self);
  memcpy(dst, src, e->data_size);
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *self)
{
  unaligned_assign_ck *e = reinterpret_cast<unaligned_assign_ck *>(self);
  size_t data_size = e->data_size;
  const char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    memcpy(dst, src0, data_size);
    dst += dst_stride;
    src0 += src0_stride;
  }
}

size_t make_unaligned_assignment_kernel(
                ckernel_builder *ckb, size_t ckb_offset,
                size_t data_size,
                kernel_request_t kernreq)
{
  // Allocate the necessary space in the output kernel buffer
  ckb->ensure_capacity_leaf(ckb_offset + sizeof(unaligned_assign_ck));
  unaligned_assign_ck *result;
  result = ckb->get_at<unaligned_assign_ck>(ckb_offset);

  // Set the appropriate function based on the type of kernel requested
  result.base->set_expr_function(kernreq, &single_assign, &strided_assign);
  // Set the kernel data
  result->data_size = data_size;

  // Return the offset immediately after this ckernel's data
  return ckb_offset + sizeof(unaligned_assign_ck);
}
```

### Leaf Kernel Construction With Dynamic Resources

The final kind of leaf ckernels are ones with dynamically-allocated
resources they are responsible for. The example we use is a ckernel
created by a JIT engine. On destruction, the ckernel must free
its reference to the resources holding the JITted function.

```cpp
// A hypothetical JIT API
void create_jit_assignment(const dtype& dst_dt, const char *dst_arrmeta,
                const dtype& src_dt, const char *src_arrmeta,
                kernel_request_t kernreq,
                void **out_function_ptr,
                void **out_jit_handle)
void jit_free(void *jit_handle);

// This is the data for the ckernel. It starts with a
// ckernel_prefix, then has fields for other data.
// Remember, this data must be movable by using a memcpy,
// which is trivially true in this case.
struct jit_ck {
  ckernel_prefix base;
  void *jit_handle;
};

static void destruct(ckernel_prefix *self)
{
  jit_ck *e = reinterpret_cast<jit_ck *>(self);
  if (e->jit_handle != NULL) {
    jit_free(e->jit_handle);
  }
}

size_t make_jit_assignment_kernel(
                ckernel_builder *ckb, size_t ckb_offset,
                const dtype& dst_dt, const char *dst_arrmeta,
                const dtype& src_dt, const char *src_arrmeta,
                kernel_request_t kernreq)
{
  // Allocate the necessary space in the output kernel buffer
  ckb->ensure_capacity_leaf(ckb_offset + sizeof(jit_ck));
  jit_ck *result;
  result = ckb->get_at<jit_ck>(ckb_offset);

  // Always set the destructor first, so that if things go wrong
  // later, partially constructed resources are freed.
  result->base.destructor = &destruct;

  // Set the function and the jit handle at once.
  // Assume the jit_assignment function raises an exception on error
  create_jit_assignment(dst_dt, dst_arrmeta, src_dt, src_arrmeta, kernreq,
                  &result->base.function,
                  &result->jit_handle);

  // Return the offset immediately after this kernel's data
  return ckb_offset + sizeof(jit_ck);
}
```

### Simple Nested Kernel Construction

Many DyND types are composed of other types, and constructing
ckernels for them typically involves composing ckernels from
their subtypes. The simplest case of this is a single child
ckernel. For this example, we show how an assignment from a
pointer<T1> to T2 can be constructed.

```cpp

static void destruct(ckernel_prefix *self)
{
  ckernel_prefix *echild = self + 1;
  if (echild->destructor) {
    echild->destructor(echild);
  }
}

static void single_assign(char *dst, const char *const *src,
                          ckernel_prefix *self)
{
  ckernel_prefix *echild = self + 1;
  unary_single_operation_t opchild =
      echild->get_function<unary_single_operation_t>();
  opchild(dst, *(const char **)src[0], echild);
}

static void strided_assign(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *self)
{
  ckernel_prefix *echild = self + 1;
  unary_single_operation_t opchild =
       echild->get_function<unary_single_operation_t>();
  const char *src0 = src[0];
  intptr_t src0_stride = src_stride[0];
  for (size_t i = 0; i != count; ++i) {
    opchild(dst, *(const char **)src0, echild);
    dst += dst_stride;
    src0 += src0_stride;
  }
}

size_t make_ptr_assignment_kernel(
                ckernel_builder *ckb, size_t ckb_offset,
                const dtype& dst_dt, const char *dst_arrmeta,
                const dtype& src_target_dt, const char *src_target_arrmeta,
                kernel_request_t kernreq)
{
  ckernel_prefix *result;
  result = ckb->get_at<ckernel_prefix>(ckb_offset);

  // Always set the destructor first, so that if things go wrong
  // later, partially constructed resources are freed.
  result->destructor = &destruct;

  // Set the appropriate function based on the type of kernel requested
  result->set_expr_function(kernreq, &single_assign, &strided_assign);

  // Construct the child assignment kernel, and
  // return the offset immediately after the child kernel's data
  return make_assignment_kernel(ckb, ckb_offset + sizeof(ckernel_prefix),
                  dst_dt, dst_arrmeta,
                  src_target_dt, src_target_arrmeta,
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
