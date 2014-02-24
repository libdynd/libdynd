//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_device_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

cuda_device_type::cuda_device_type(const ndt::type& target_tp)
    : base_memory_type(cuda_device_type_id, memory_kind, target_tp.get_data_size(),
        target_tp.get_data_alignment(), target_tp.get_flags(), target_tp.get_metadata_size(), target_tp.get_ndim(), target_tp)
{
}

cuda_device_type::~cuda_device_type()
{
}

void cuda_device_type::transform_child_types(type_transform_fn_t transform_fn, void *extra,
                ndt::type& out_transformed_tp, bool& out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_target_tp, extra, tmp_tp, was_transformed);
    if (was_transformed) {
        out_transformed_tp = ndt::type(new cuda_device_type(tmp_tp), false);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}

void cuda_device_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    m_target_tp.print_data(o, metadata, data);
}

void cuda_device_type::print_type(std::ostream& o) const
{
    o << "cuda_device(" << m_target_tp << ")";
}

void cuda_device_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape,
                const char *metadata, const char *data) const
{
    m_target_tp.extended()->get_shape(ndim, i, out_shape, metadata, data);
}

void cuda_device_type::get_strides(size_t i, intptr_t *out_strides, const char *metadata) const
{
    m_target_tp.extended()->get_strides(i, out_strides, metadata);
}

ndt::type cuda_device_type::apply_linear_index(intptr_t nindices, const irange *indices,
            size_t current_i, const ndt::type& root_tp, bool leading_dimension) const
{
    return make_cuda_device(m_target_tp.extended()->apply_linear_index(nindices, indices,
        current_i, root_tp, leading_dimension));
}

ndt::type cuda_device_type::at_single(intptr_t i0, const char **inout_metadata, const char **inout_data) const
{
    return make_cuda_device(m_target_tp.extended()->at_single(i0, inout_metadata, inout_data));
}

ndt::type cuda_device_type::get_type_at_dimension(char **inout_metadata, intptr_t i, intptr_t total_ndim) const
{
    if (i == 0) {
        return ndt::type(this, true);
    }
    else {
        return make_cuda_device(m_target_tp.extended()->get_type_at_dimension(inout_metadata, i, total_ndim));
    }
}

void cuda_device_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void cuda_device_type::metadata_copy_construct(char *dst_metadata, const char *src_metadata, memory_block_data *embedded_reference) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_copy_construct(dst_metadata,
                        src_metadata, embedded_reference);
    }
}

void cuda_device_type::metadata_destruct(char *metadata) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(metadata);
    }
}

void cuda_device_type::data_alloc(char **data, size_t size) const
{
    throw_if_not_cuda_success(cudaMalloc(data, size));
}

void cuda_device_type::data_zeroinit(char *data, size_t size) const
{
    throw_if_not_cuda_success(cudaMemset(data, 0, size));
}

inline cudaMemcpyKind get_cuda_memcpy_kind(const ndt::type& dst_tp, const ndt::type& src_tp) {
    if (dst_tp.get_type_id() == cuda_device_type_id) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return cudaMemcpyDeviceToDevice;
        }

        return cudaMemcpyHostToDevice;
    }

    if (src_tp.get_type_id() == cuda_device_type_id) {
        return cudaMemcpyDeviceToHost;
    }

    return cudaMemcpyHostToHost;
}

inline kernel_request_t get_single_cuda_kernreq(const ndt::type& dst_tp, const ndt::type& src_tp) {
    if (dst_tp.get_type_id() == cuda_device_type_id) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return kernel_request_single_cuda_device_to_device;
        }

        return kernel_request_single_cuda_host_to_device;
    }

    if (src_tp.get_type_id() == cuda_device_type_id) {
        return kernel_request_single_cuda_device_to_host;
    }

    return kernel_request_single;
}


inline kernel_request_t get_strided_cuda_kernreq(const ndt::type& dst_tp, const ndt::type& src_tp) {
    if (dst_tp.get_type_id() == cuda_device_type_id) {
        if (src_tp.get_type_id() == cuda_device_type_id) {
            return kernel_request_strided_cuda_device_to_device;
        }

        return kernel_request_strided_cuda_host_to_device;
    }

    if (src_tp.get_type_id() == cuda_device_type_id) {
        return kernel_request_strided_cuda_device_to_host;
    }

    return kernel_request_strided;
}

size_t cuda_device_type::make_assignment_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                const ndt::type& src_tp, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx) const
{
    const ndt::type& dst_target_tp = dst_tp.get_canonical_type();
    const ndt::type& src_target_tp = src_tp.get_canonical_type();

    if (false && dst_target_tp == src_target_tp && dst_target_tp.data_layout_compatible_with(src_target_tp)) {
        if (dst_target_tp.is_builtin()) {
            return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata,
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
        } else {
//            cout << "dst/src pod" << endl;
    //        cout << dst_tp << endl;
  //          cout << src_tp << endl;
            return make_pod_typed_data_assignment_kernel(out, offset_out, dst_target_tp.get_data_size(),
                dst_target_tp.get_data_alignment(), get_single_cuda_kernreq(dst_tp, src_tp));
        }
    } else {
        if (this == dst_tp.extended()) {
            if (dst_target_tp.is_builtin()) {
//                cout << "dst inexact builtin" << endl;
                if (kernreq == kernel_request_strided) {
                    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
                }
                return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata,
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
           // } else if (dst_tp.get_ndim() == 1 && src_tp.get_type_id() == cuda_device_type_id) { // if dst is strided and only has 1 dimension
  //              cout << "dst strided device->device" << endl;
    //            cout << dst_tp << endl;
      //          cout << src_tp << endl;
        //        return 0;
            } else {
//                cout << "shifting dst" << endl;
                const char *shifted_metadata = dst_metadata;
                ndt::type shifted_tp = dst_tp.with_shifted_memory_type();
  //              cout << "unshifted: " << dst_tp << endl;
    //            cout << "shifted: " << shifted_tp << endl;
                return ::make_assignment_kernel(out, offset_out,
                    shifted_tp, shifted_metadata, src_tp, src_metadata,
                    kernreq, errmode, ectx);
            }
        } else if (this == src_tp.extended()) {
            if (src_target_tp.is_builtin()) {
                if (kernreq == kernel_request_strided) {
                    offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, kernreq);
                }
                    return ::make_assignment_kernel(out, offset_out,
                                dst_target_tp, dst_metadata,
                                src_target_tp, src_metadata,
                                get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
            //} else if (src_tp.get_ndim() == 1 && dst_tp.get_type_id() == cuda_device_type_id) { // is src is strided and only has 1 dimension
              //  cout << "src strided device->device" << endl;
                //cout << dst_tp << endl;
               // cout << src_tp << endl;
                //return 0;
            } else {
//                cout << "shifting src" << endl;
                const char *shifted_metadata = src_metadata;
                ndt::type shifted_tp = src_tp.with_shifted_memory_type();
  //              cout << "unshifted: " << src_tp << endl;
    //            cout << "shifted: " << shifted_tp << endl;
                return ::make_assignment_kernel(out, offset_out,
                    dst_tp, dst_metadata, shifted_tp, shifted_metadata,
                    kernreq, errmode, ectx);
            }
        }
        else {
            cout << "ERROR" << endl;
            return 0;
        }
    }
}


//        if (dst_target_tp.is_builtin()) {
  //          return ::make_assignment_kernel(out, offset_out,
    //                    dst_target_tp, dst_metadata,
      //                  src_target_tp, src_metadata,
        //                get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
      //  } else {
        //    return make_pod_typed_data_assignment_kernel(out, offset_out, dst_target_tp.get_data_size(),
          //      dst_target_tp.get_data_alignment(), get_single_cuda_kernreq(dst_tp, src_tp));
        //}

    

//        cout << "compat" << endl;



//        out->ensure_capacity(offset_out);

/*
        single_cuda_device_assign_kernel_extra *e = out->get_at<single_cuda_device_assign_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(&single_cuda_device_assign_kernel_extra::single);
        e->base.destructor = single_cuda_device_assign_kernel_extra::destruct;
        e->size = dst_target_tp.get_data_size();
        e->kind = get_cuda_memcpy_kind(dst_tp, src_tp);

        return offset_out + sizeof(single_cuda_device_assign_kernel_extra);*/

//        return ::make_assignment_kernel(out, offset_out,
  //                      dst_target_tp, dst_metadata,
    //                    src_target_tp, src_metadata,
      //                  get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
  //  }

    // the assignment needs a cast first
    // only do this if one of the arrays is a scalar or both arrays on the device

/*
    if (this == dst_tp.extended()) {
        return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata,
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);
    }

        return ::make_assignment_kernel(out, offset_out,
                        dst_target_tp, dst_metadata,
                        src_target_tp, src_metadata,
                        get_single_cuda_kernreq(dst_tp, src_tp), errmode, ectx);*/

/*
        out->ensure_capacity(offset_out + sizeof(single_cuda_device_assign_kernel_extra));
        single_cuda_device_assign_kernel_extra *e = out->get_at<single_cuda_device_assign_kernel_extra>(offset_out);
        e->base.set_function<unary_single_operation_t>(&single_cuda_device_assign_kernel_extra::single);
        e->base.destructor = single_cuda_device_assign_kernel_extra::destruct;
        e->size = dst_target_tp.get_data_size();
        e->kind = get_cuda_memcpy_kind(dst_tp, src_tp);

        return offset_out + sizeof(single_cuda_device_assign_kernel_extra);*/

#endif // DYND_CUDA
