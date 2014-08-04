//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ASSIGNMENT_KERNELS_HPP_
#define _DYND__ASSIGNMENT_KERNELS_HPP_

#include <stdexcept>

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {
/**
 * See the ckernel_builder class documentation
 * for details about how ckernels can be built and
 * used.
 *
 * This kernel type is for ckernels which assign one
 * data value from one type/arrmeta source to
 * a different type/arrmeta destination, using
 * the `unary_single_operation_t` function prototype.
 */
class unary_ckernel_builder : public ckernel_builder {
public:
    unary_ckernel_builder()
        : ckernel_builder()
    {
    }

    inline expr_single_t get_function() const {
        return get()->get_function<expr_single_t>();
    }

    /** Calls the function to do the assignment */
    inline void operator()(char *dst, const char *src) const {
        ckernel_prefix *kdp = get();
        expr_single_t fn = kdp->get_function<expr_single_t>();
        fn(dst, &src, kdp);
    }
};

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
class assignment_strided_ckernel_builder : public ckernel_builder {
public:
    assignment_strided_ckernel_builder()
        : ckernel_builder()
    {
    }

    inline expr_strided_t get_function() const {
        return get()->get_function<expr_strided_t>();
    }

    /** Calls the function to do the assignment */
    inline void operator()(char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride, size_t count) const {
        ckernel_prefix *kdp = get();
        expr_strided_t fn = kdp->get_function<expr_strided_t>();
        fn(dst, dst_stride, &src, &src_stride, count, kdp);
    }
};


namespace kernels {

    /**
     * A CRTP (curiously recurring template pattern) base class to help
     * create ckernels.
     */
    template<class CKT>
    struct unary_ck : public general_ck<CKT> {
        typedef CKT self_type;
        typedef general_ck<CKT> parent_type;

        /**
         * Initializes just the base.function member
         */
        inline void init_kernfunc(kernel_request_t kernreq)
        {
            switch (kernreq) {
            case kernel_request_single:
                this->base.template set_function<expr_single_t>(&self_type::single_wrapper);
                break;
            case kernel_request_strided:
                this->base.template set_function<expr_strided_t>(&self_type::strided_wrapper);
                break;
            default: {
                std::stringstream ss;
                ss << "assignment ckernel init: unrecognized ckernel request " << (int)kernreq;
                throw std::invalid_argument(ss.str());
            }
            }
        }

        static void single_wrapper(char *dst, const char *const *src,
                                   ckernel_prefix *rawself)
        {
            return parent_type::get_self(rawself)->single(dst, *src);
        }

        static void strided_wrapper(char *dst, intptr_t dst_stride,
                                    const char *const *src,
                                    const intptr_t *src_stride, size_t count,
                                    ckernel_prefix *rawself)
        {
            return parent_type::get_self(rawself)
                ->strided(dst, dst_stride, *src, *src_stride, count);
        }

        template<class R, class T0>
        inline void call_single_typed(char *dst, const char *src, R (self_type::*)(T0))
        {
            *reinterpret_cast<R *>(dst) =
                static_cast<self_type *>(this)
                    ->operator()(*reinterpret_cast<const T0 *>(src));
        }

        /**
         * Default single implementation calls operator(), which
         * should look similar to int32_t operator()(int64_t val).
         *
         * This can also be implemented directly in self_type to provide
         * more controlled behavior or for non-trivial types.
         */
        inline void single(char *dst, const char *src)
        {
            call_single_typed(dst, src, &self_type::operator());
        }

        /**
         * Default strided implementation calls single repeatedly.
         */
        inline void strided(char *dst, intptr_t dst_stride, const char *src,
                            intptr_t src_stride, size_t count)
        {
            self_type *self = parent_type::get_self(&this->base);
            for (size_t i = 0; i != count; ++i) {
                self->single(dst, src);
                dst += dst_stride;
                src += src_stride;
            }
        }
    };

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
size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp, const char *dst_arrmeta,
                              const ndt::type &src_tp, const char *src_arrmeta,
                              kernel_request_t kernreq,
                              const eval::eval_context *ectx);

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
size_t make_pod_typed_data_assignment_kernel(ckernel_builder *ckb,
                                             intptr_t ckb_offset,
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
size_t make_builtin_type_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, type_id_t dst_type_id,
    type_id_t src_type_id, kernel_request_t kernreq, assign_error_mode errmode);

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
size_t make_kernreq_to_single_kernel_adapter(ckernel_builder *ckb,
                                             intptr_t ckb_offset, int nsrc,
                                             kernel_request_t kernreq);

namespace kernels {
/**
 * Generic assignment kernel + destructor for a strided dimension.
 * This requires that the child kernel be created with the
 * kernel_request_strided type of kernel.
 */
struct strided_assign_ck : public kernels::unary_ck<strided_assign_ck> {
    intptr_t m_size;
    intptr_t m_dst_stride, m_src_stride;

    inline void single(char *dst, const char *src)
    {
        ckernel_prefix *child = get_child_ckernel();
        expr_strided_t child_fn = child->get_function<expr_strided_t>();
        child_fn(dst, m_dst_stride, &src, &m_src_stride, m_size, child);
    }

    inline void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count)
    {
        ckernel_prefix *child = get_child_ckernel();
        expr_strided_t child_fn = child->get_function<expr_strided_t>();
        intptr_t inner_size = m_size, inner_dst_stride = m_dst_stride,
                 inner_src_stride = m_src_stride;
        for (size_t i = 0; i != count; ++i) {
            child_fn(dst, inner_dst_stride, &src, &inner_src_stride, inner_size,
                     child);
            dst += dst_stride;
            src += src_stride;
        }
    }

    inline void destruct_children() {
        // Destroy the child ckernel
        get_child_ckernel()->destroy();
    }
};

} // namespace kernels

#ifdef DYND_CUDA
/**
 * Creates an assignment kernel for one data value from the
 * src type/arrmeta to the dst type/arrmeta. This adds the
 * kernel at the 'ckb_offset' position in 'ckb's data, as part
 * of a hierarchy matching the dynd type's hierarchy. At least
 * one of the types should be a CUDA type.
 *
 * This function should always be called with this == dst_tp first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_tp  The destination dynd type.
 * \param dst_arrmeta  Arrmeta for the destination data.
 * \param src_tp  The source dynd type.
 * \param src_arrmeta  Arrmeta for the source data
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param errmode  The error mode to use for assignments.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'ckb' immediately after the
 *           created kernel.
 */
size_t make_cuda_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta,
                kernel_request_t kernreq,
                const eval::eval_context *ectx);

/**
 * Creates an assignment kernel when the src and the dst are the same, but
 * can be in a CUDA memory space, and are POD (plain old data).
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_device  If the destination data is on the CUDA device, true. Otherwise false.
 * \param src_device  If the source data is on the CUDA device, true. Otherwise false.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
size_t make_cuda_pod_typed_data_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                bool dst_device, bool src_device,
                size_t data_size, size_t data_alignment,
                kernel_request_t kernreq);

/**
 * Creates an assignment kernel from the src to the dst built in
 * type ids. Either the src or the dst can be in a CUDA memory space.
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_device  If the destination data is on the CUDA device, true. Otherwise false.
 * \param dst_type_id  The destination dynd type id.
 * \param src_device  If the source data is on the CUDA device, true. Otherwise false.
 * \param src_type_id  The source dynd type id.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param errmode  The error mode to use for assignments.
 */
size_t make_cuda_builtin_type_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                bool dst_device, type_id_t dst_type_id,
                bool src_device, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode);
#endif
} // namespace dynd

#endif // _DYND__ASSIGNMENT_KERNELS_HPP_
