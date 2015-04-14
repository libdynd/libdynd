//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <stdexcept>

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/kernels/single_assigner_builtin.hpp>
#include <map>

namespace dynd {

/**
 * See the ckernel_builder class documentation
 * for details about how ckernels can be built and
 * used.
 *
 * This kernel type is for ckernels which assign a
 * strided sequence of data values from one
 * type/arrmeta source to a different type/arrmeta
 * destination, using the `unary_strided_operation_t`
 * function prototype.
 */
class assignment_strided_ckernel_builder
    : public ckernel_builder<kernel_request_host> {
public:
  assignment_strided_ckernel_builder() : ckernel_builder<kernel_request_host>()
  {
  }

  inline expr_strided_t get_function() const
  {
    return get()->get_function<expr_strided_t>();
  }

  /** Calls the function to do the assignment */
  inline void operator()(char *dst, intptr_t dst_stride, char *src,
                         intptr_t src_stride, size_t count) const
  {
    ckernel_prefix *kdp = get();
    expr_strided_t fn = kdp->get_function<expr_strided_t>();
    fn(dst, dst_stride, &src, &src_stride, count, kdp);
  }
};

namespace kernels {

  template <class dst_type, class src_type, assign_error_mode errmode>
  struct assign_ck : nd::base_kernel<assign_ck<dst_type, src_type, errmode>,
                                     kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      single_assigner_builtin<dst_type, src_type, errmode>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(*src));
    }
  };

  template <class dst_type, class src_type>
  struct assign_ck<dst_type, src_type, assign_error_nocheck>
      : nd::base_kernel<assign_ck<dst_type, src_type, assign_error_nocheck>,
                        kernel_request_cuda_host_device, 1> {
    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      single_assigner_builtin<dst_type, src_type, assign_error_nocheck>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(*src));
    }
  };

#ifdef DYND_CUDA

  struct cuda_host_to_device_assign_ck
      : nd::expr_ck<cuda_host_to_device_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *dst;

    cuda_host_to_device_assign_ck(size_t data_size)
        : data_size(data_size), dst(new char[data_size])
    {
    }

    ~cuda_host_to_device_assign_ck() { delete[] dst; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();

      single(this->dst, src, child);
      cuda_throw_if_not_success(
          cudaMemcpy(dst, this->dst, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct cuda_host_to_device_copy_ck
      : nd::expr_ck<cuda_host_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_host_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct cuda_device_to_host_assign_ck
      : nd::expr_ck<cuda_device_to_host_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *src;

    cuda_device_to_host_assign_ck(size_t data_size)
        : data_size(data_size), src(new char[data_size])
    {
    }

    ~cuda_device_to_host_assign_ck() { delete[] src; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();

      cuda_throw_if_not_success(
          cudaMemcpy(this->src, *src, data_size, cudaMemcpyDeviceToHost));
      single(dst, &this->src, child);
    }
  };

  struct cuda_device_to_host_copy_ck
      : nd::expr_ck<cuda_device_to_host_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_host_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToHost));
    }
  };

  struct cuda_device_to_device_copy_ck
      : nd::expr_ck<cuda_device_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToDevice));
    }
  };

#endif
} // namespace kernels

/**
 * Creates an assignment kernel for one data value from the
 * src type/arrmeta to the dst type/arrmeta. This adds the
 * kernel at the 'ckb_offset' position in 'ckb's data, as part
 * of a hierarchy matching the dynd type's hierarchy.
 *
 * This function should always be called with this == dst_tp first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 *
 * \param ckb  The ckernel_builder being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_tp  The destination dynd type.
 * \param dst_arrmeta  Arrmeta for the destination data.
 * \param src_tp  The source dynd type.
 * \param src_arrmeta  Arrmeta for the source data
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'ckb' immediately after the
 *           created kernel.
 */
intptr_t
make_assignment_kernel(const arrfunc_type_data *self, const arrfunc_type *af_tp,
                       void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                       const char *dst_arrmeta, const ndt::type &src_tp,
                       const char *src_arrmeta, kernel_request_t kernreq,
                       const eval::eval_context *ectx, const nd::array &kwds);

inline intptr_t
make_assignment_kernel(const arrfunc_type_data *self, const arrfunc_type *af_tp,
                       void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                       const char *dst_arrmeta, const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq,
                       const eval::eval_context *ectx, const nd::array &kwds)
{
  return make_assignment_kernel(self, af_tp, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, *src_tp, *src_arrmeta, kernreq,
                                ectx, kwds);
}

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
size_t make_pod_typed_data_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                             size_t data_size,
                                             size_t data_alignment,
                                             kernel_request_t kernreq);

/**
 * Creates an assignment kernel from the src to the dst built in
 * type ids.
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_type_id  The destination dynd type id.
 * \param src_type_id  The source dynd type id.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param errmode  The error mode to use for assignments.
 */
size_t make_builtin_type_assignment_kernel(void *ckb, intptr_t ckb_offset,
                                           type_id_t dst_type_id,
                                           type_id_t src_type_id,
                                           kernel_request_t kernreq,
                                           assign_error_mode errmode);

/**
 * When kernreq != kernel_request_single, adds an adapter to
 * the kernel which provides the requested kernel, and uses
 * a single kernel to fulfill the assignments. The
 * caller can use it like:
 *
 *  {
 *      ckb_offset = make_kernreq_to_single_kernel_adapter(
 *                      ckb, ckb_offset, kernreq);
 *      // Proceed to create 'single' kernel...
 */
size_t make_kernreq_to_single_kernel_adapter(void *ckb, intptr_t ckb_offset,
                                             int nsrc,
                                             kernel_request_t kernreq);

#ifdef DYND_CUDA
/**
 * Creates an assignment kernel when the src and the dst are the same, but
 * can be in a CUDA memory space, and are POD (plain old data).
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_device  If the destination data is on the CUDA device, true.
 *                    Otherwise false.
 * \param src_device  If the source data is on the CUDA device, true. Otherwise
 *                    false.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
size_t make_cuda_pod_typed_data_assignment_kernel(
    void *ckb, intptr_t ckb_offset, bool dst_device, bool src_device,
    size_t data_size, size_t data_alignment, kernel_request_t kernreq);

intptr_t make_cuda_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

intptr_t make_cuda_to_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

intptr_t make_cuda_from_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

#endif // DYND_CUDA
} // namespace dynd
