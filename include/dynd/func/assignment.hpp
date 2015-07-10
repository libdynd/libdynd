//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/multidispatch.hpp>

namespace dynd {
namespace nd {

  template <typename T, int N>
  struct declop;

  template <typename T>
  struct declop<T, 2> : declfunc<T> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc make()
    {
      for (const auto &pair : T::make_children()) {
        children[pair.first[0]][pair.first[1]] = pair.second;
      }

      return functional::multidispatch(dynd::ndt::type("(Any) -> Any"),
                                       children, default_child, {-1, 0});
    }
  };

  template <typename T>
  arrfunc declop<T, 2>::children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];

  template <typename T>
  arrfunc declop<T, 2>::default_child;

  /*
    template <typename F, template <type_id_t> class K>
    struct arithmetic_operator<F, K, 1> : declfunc<F> {
      static arrfunc children[DYND_TYPE_ID_MAX + 1];
      static arrfunc default_child;

      static arrfunc make()
      {
        const arrfunc self = functional::call<F>(ndt::type("(Any) -> Any"));

        for (const std::pair<const type_id_t, arrfunc> &pair :
             arrfunc::make_all<K, numeric_type_ids>()) {
          children[pair.first] = pair.second;
        }

        for (type_id_t i0 : dim_type_ids::vals()) {
          const ndt::type child_tp = ndt::arrfunc_type::make(
              {ndt::type(i0)}, self.get_type()->get_return_type());
          children[i0] = functional::elwise(child_tp, self);
        }

        return functional::multidispatch_by_type_id(
            self.get_array_type(), DYND_TYPE_ID_MAX + 1, children,
    default_child,
            false);
      }
    };

    template <typename F, template <type_id_t> class K>
    arrfunc arithmetic_operator<F, K, 1>::children[DYND_TYPE_ID_MAX + 1];
  */

  extern struct assign : declop<assign, 2> {
    static std::map<std::array<type_id_t, 2>, arrfunc> make_children();
  } assign;

} // namespace dynd::nd

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
intptr_t make_assignment_kernel(
    const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds);

inline intptr_t
make_assignment_kernel(const arrfunc_type_data *self,
                       const ndt::arrfunc_type *af_tp, void *ckb,
                       intptr_t ckb_offset, const ndt::type &dst_tp,
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
 * \param src_device  If the source data is on the CUDA device, true.
 *Otherwise
 *                    false.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
size_t make_cuda_pod_typed_data_assignment_kernel(
    void *ckb, intptr_t ckb_offset, bool dst_device, bool src_device,
    size_t data_size, size_t data_alignment, kernel_request_t kernreq);

intptr_t make_cuda_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

intptr_t make_cuda_to_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

intptr_t make_cuda_from_device_builtin_type_assignment_kernel(
    const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars);

#endif // DYND_CUDA

} // namespace dynd