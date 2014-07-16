//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dim_iter.hpp>
#include <dynd/array.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

////////////////////////////////
// Implementation of the strided dim iter.

static void strided_dim_iter_destructor(dim_iter *self)
{
    // Free the reference of the element type
    base_type_xdecref(self->eltype);
    // Free the reference owning the data
    memory_block_data *memblock = reinterpret_cast<memory_block_data *>(self->custom[2]);
    if (memblock != NULL) {
        memory_block_decref(memblock);
    }
}

static int strided_dim_iter_next(dim_iter *self)
{
    if (self->data_ptr == NULL) {
        // The first time `next` is called, populate the data and size
        self->data_ptr = reinterpret_cast<const char *>(self->custom[0]);
        self->data_elcount = static_cast<intptr_t>(self->custom[1]);
        return 1;
    } else {
        // In the strided case, there is always just one chunk,
        // so advancing the iterator a second time
        self->data_elcount = 0;
        return 0;
    }
}

static void strided_dim_iter_seek(dim_iter *self, intptr_t i)
{
    // Seek positions the data chunk to be from the specified
    // element to the end of the range.
    const char *data_ptr = reinterpret_cast<const char *>(self->custom[0]);
    intptr_t dim_size = static_cast<intptr_t>(self->custom[1]);
    if (i >= 0 && i < dim_size) {
        self->data_ptr = data_ptr + i * self->data_stride;
        self->data_elcount = dim_size - i;
    } else {
        self->data_ptr = NULL;
        self->data_elcount = 0;
    }
}

static dim_iter_vtable strided_dim_iter_vt = {
    strided_dim_iter_destructor,
    strided_dim_iter_next,
    strided_dim_iter_seek
};

void dynd::make_strided_dim_iter(
    dim_iter *out_di,
    const ndt::type& tp, const char *arrmeta,
    const char *data_ptr, intptr_t size, intptr_t stride,
    const memory_block_ptr& ref)
{
    out_di->vtable = &strided_dim_iter_vt;
    out_di->data_ptr = NULL;
    out_di->data_elcount = 0;
    out_di->data_stride = stride;
    out_di->flags = dim_iter_restartable | dim_iter_seekable;
    if ((intptr_t)tp.get_data_size() == stride) {
        out_di->flags |= dim_iter_contiguous;
    }
    out_di->eltype = ndt::type(tp).release();
    out_di->el_arrmeta = arrmeta;
    // The custom fields are where we place the data needed for seeking
    // and the reference object.
    out_di->custom[0] = reinterpret_cast<uintptr_t>(data_ptr);
    out_di->custom[1] = static_cast<uintptr_t>(size);
    if (ref.get() != NULL) {
        memory_block_incref(ref.get());
        out_di->custom[2] = reinterpret_cast<uintptr_t>(ref.get());
    } else {
        out_di->custom[2] = 0;
    }
}

////////////////////////////////
// Implementation of the buffered strided dim iter.

static void buffered_strided_dim_iter_destructor(dim_iter *self)
{
    // Free the reference of the element type
    base_type_xdecref(self->eltype);
    // Free the reference owning the temporary buffer
    memory_block_data *memblock = reinterpret_cast<memory_block_data *>(self->custom[2]);
    if (memblock != NULL) {
        memory_block_decref(memblock);
    }
    // Free the reference owning the data
    memblock = reinterpret_cast<memory_block_data *>(self->custom[3]);
    if (memblock != NULL) {
        memory_block_decref(memblock);
    }
}

static int buffered_strided_dim_iter_next(dim_iter *self)
{
    intptr_t i = static_cast<intptr_t>(self->custom[0]);
    intptr_t size = static_cast<intptr_t>(self->custom[1]);
    if (i < size) {
        nd::array buf(memory_block_ptr(reinterpret_cast<memory_block_data *>(self->custom[5]), true));
        if (!buf.get_type().is_builtin()) {
            // For types with block references, need to reset the buffers each time
            // we fill buf with data.
            buf.get_type().extended()->arrmeta_reset_buffers(buf.get_arrmeta());
        }
        // Figure out how many elements we will buffer
        intptr_t bufsize = reinterpret_cast<const strided_dim_type_arrmeta *>(
                               buf.get_arrmeta())->dim_size;
        if (i + bufsize > size) {
            bufsize = size - i;
        }
        // Indicate in the dim_iter where the next index is
        self->custom[0] = i + bufsize;
        // Copy the data into the buffer
        const char *data_ptr = reinterpret_cast<const char *>(self->custom[2]);
        intptr_t stride = static_cast<intptr_t>(self->custom[3]);
        ckernel_builder *ckb = reinterpret_cast<ckernel_builder *>(self->custom[4]);
        ckernel_prefix *kdp = ckb->get();
        expr_strided_t fn = kdp->get_function<expr_strided_t>();
        const char *child_data_ptr = data_ptr + i * stride;
        fn(buf.get_readwrite_originptr(), self->data_stride, &child_data_ptr,
           &stride, bufsize, kdp);
        // Update the dim_iter's size
        self->data_elcount = bufsize;
        return 1;
    } else {
        self->data_elcount = 0;
        return 0;
    }
}

static void buffered_strided_dim_iter_seek(dim_iter *self, intptr_t i)
{
    // Set the index to where we want to seek, and use the `next`
    // function to do the rest.
    self->custom[0] = static_cast<uintptr_t>(i);
    buffered_strided_dim_iter_next(self);
}

static dim_iter_vtable buffered_strided_dim_iter_vt = {
    buffered_strided_dim_iter_destructor,
    buffered_strided_dim_iter_next,
    buffered_strided_dim_iter_seek
};

void dynd::make_buffered_strided_dim_iter(
    dim_iter *out_di,
    const ndt::type& val_tp,
    const ndt::type& mem_tp, const char *mem_arrmeta,
    const char *data_ptr, intptr_t size, intptr_t stride,
    const memory_block_ptr& ref, intptr_t buffer_max_mem,
    const eval::eval_context *ectx)
{
    if (val_tp == mem_tp) {
        // If no buffering is needed, ust the straight strided iter
        make_strided_dim_iter(out_di, mem_tp, mem_arrmeta,
                data_ptr, size, stride, ref);
        return;
    }
    // Allocate the temporary buffer
    intptr_t buffer_elcount = buffer_max_mem;
    intptr_t buffer_data_size = mem_tp.get_data_size();
    intptr_t buffer_ndim = mem_tp.get_ndim() + 1;
    dimvector buffer_shape(buffer_ndim);
    if (!mem_tp.is_builtin()) {
        // Get the shape from mem_tp/mem_meta
        mem_tp.extended()->get_shape(buffer_ndim - 1, 0, buffer_shape.get() + 1, mem_arrmeta, NULL);
        buffer_data_size = mem_tp.extended()->get_default_data_size(buffer_ndim - 1, buffer_shape.get() + 1);
    }
    buffer_elcount /= buffer_data_size;
    if (buffer_elcount >= size) {
        buffer_elcount = size;
    }
    buffer_shape[0] = buffer_elcount;
    nd::array buf = nd::array(nd::typed_empty(buffer_ndim, buffer_shape.get(),
                                              ndt::make_strided_dim(val_tp)));
    if (buffer_ndim > 2 && val_tp.get_type_id() == strided_dim_type_id) {
        // Reorder the strides to preserve F-order if it's a strided array
        val_tp.tcast<strided_dim_type>()->reorder_default_constructed_strides(
            buf.get_arrmeta() + sizeof(strided_dim_type_arrmeta), mem_tp,
            mem_arrmeta);
    }
    intptr_t buffer_stride = reinterpret_cast<const strided_dim_type_arrmeta *>(
                                 buf.get_arrmeta())->stride;
    // Make the ckernel that copies data to the buffer
    ckernel_builder k;
    make_assignment_kernel(&k, 0, val_tp,
                           buf.get_arrmeta() + sizeof(strided_dim_type_arrmeta),
                           mem_tp, mem_arrmeta, kernel_request_strided, ectx);

    if (buffer_elcount == size) {
        // If the buffer is big enough for all the data, just make a copy and
        // create a strided dim_iter around it.
        ckernel_prefix *kdp = k.get();
        expr_strided_t fn = kdp->get_function<expr_strided_t>();
        fn(buf.get_readwrite_originptr(),
            buffer_stride, &data_ptr, &stride, size, kdp);
        make_strided_dim_iter(out_di, val_tp,
            buf.get_arrmeta() + sizeof(strided_dim_type_arrmeta),
            buf.get_readonly_originptr(), size, buffer_stride, buf.get_memblock());
        return;
    }
    out_di->vtable = &buffered_strided_dim_iter_vt;
    out_di->data_ptr = buf.get_readonly_originptr();
    out_di->data_elcount = 0;
    out_di->data_stride = buffer_stride;
    out_di->flags = dim_iter_restartable | dim_iter_seekable;
    if ((intptr_t)buf.get_dtype().get_data_size() == buffer_stride) {
        out_di->flags |= dim_iter_contiguous;
    }
    out_di->eltype = ndt::type(val_tp).release();
    out_di->el_arrmeta = buf.get_arrmeta() + sizeof(strided_dim_type_arrmeta);
    // The custom fields are where we place the data needed for seeking
    // and the reference object.
    out_di->custom[0] = 0; // The next index to buffer
    out_di->custom[1] = static_cast<uintptr_t>(size);
    out_di->custom[2] = reinterpret_cast<uintptr_t>(data_ptr);
    out_di->custom[3] = static_cast<uintptr_t>(stride);
    ckernel_builder *ckb = new ckernel_builder;
    k.swap(*ckb);
    out_di->custom[4] = reinterpret_cast<uintptr_t>(ckb);
    memory_block_incref(buf.get_memblock().get());
    out_di->custom[5] = reinterpret_cast<uintptr_t>(buf.get_memblock().get());
    if (ref.get() != NULL) {
        memory_block_incref(ref.get());
        out_di->custom[6] = reinterpret_cast<uintptr_t>(ref.get());
    } else {
        out_di->custom[6] = 0;
    }
}
