//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPR_KERNELS_HPP_
#define _DYND__EXPR_KERNELS_HPP_

#include <dynd/func/arrfunc.hpp>

namespace dynd {

typedef void (*expr_single_operation_t)(
                char *dst, const char * const *src,
                ckernel_prefix *extra);
typedef void (*expr_strided_operation_t)(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *extra);

namespace kernels {
    /**
     * A CRTP (curiously recurring template pattern) base class to help
     * create ckernels.
     */
    template<class CKT, int Nsrc>
    struct expr_ck {
        typedef CKT self_type;

        ckernel_prefix base;

        static self_type *get_self(ckernel_prefix *rawself) {
            return reinterpret_cast<self_type *>(rawself);
        }

        static const self_type *get_self(const ckernel_prefix *rawself) {
            return reinterpret_cast<const self_type *>(rawself);
        }

        static self_type *get_self(ckernel_builder *ckb, intptr_t ckb_offset) {
            return ckb->get_at<self_type>(ckb_offset);
        }

        static inline self_type *create(ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
        {
            ckb->ensure_capacity(ckb_offset + sizeof(self_type));
            ckernel_prefix *rawself = ckb->get_at<ckernel_prefix>(ckb_offset);
            return self_type::init(rawself, kernreq);
        }

        static inline self_type *create_leaf(ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
        {
            ckb->ensure_capacity_leaf(ckb_offset + sizeof(self_type));
            ckernel_prefix *rawself = ckb->get_at<ckernel_prefix>(ckb_offset);
            return self_type::init(rawself, kernreq);
        }

        /**
         * Initializes an instance of this ckernel in-place according to the
         * kernel request. This calls the constructor in-place, and initializes
         * the base function and destructor
         */
        static inline self_type *init(ckernel_prefix *rawself,
                                kernel_request_t kernreq)
        {
            // Alignment requirement of the type
            DYND_STATIC_ASSERT(
                (size_t)scalar_align_of<self_type>::value ==
                    (size_t)scalar_align_of<void *>::value,
                "ckernel types require alignment matching that of pointers");

            // Call the constructor in-place
            self_type *self = new (rawself) self_type();
            // Double check that the C++ struct layout is as we expect
            if (self != get_self(rawself)) {
                throw std::runtime_error("internal ckernel error: struct layout is not valid");
            }
            switch (kernreq) {
            case kernel_request_single:
                self->base.template set_function<expr_single_operation_t>(&self_type::single_wrapper);
                break;
            case kernel_request_strided:
                self->base.template set_function<expr_strided_operation_t>(&self_type::strided_wrapper);
                break;
            default: {
                std::stringstream ss;
                ss << "expr ckernel init: unrecognized ckernel request " << (int)kernreq;
                throw std::invalid_argument(ss.str());
            }
            }
            self->base.destructor = &self_type::destruct;
            return self;
        }

        /**
         * The ckernel destructor function, which is placed in
         * base.destructor.
         */
        static void destruct(ckernel_prefix *rawself) {
            self_type *self = get_self(rawself);
            self->destruct_children();
            self->~self_type();
        }

        /**
         * Default implementation of destruct_children does nothing.
         */
        inline void destruct_children()
        {
        }

        static void single_wrapper(char *dst, const char * const *src, ckernel_prefix *rawself) {
            return get_self(rawself)->single(dst, src);
        }

        static void strided_wrapper(char *dst, intptr_t dst_stride,
                                    const char *const *src,
                                    const intptr_t *src_stride, size_t count,
                                    ckernel_prefix *rawself)
        {
            return get_self(rawself)
                ->strided(dst, dst_stride, src, src_stride, count);
        }

        /**
         * Default strided implementation calls single repeatedly.
         */
        inline void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count)
        {
            self_type *self = get_self(&base);
            const char *src_copy[Nsrc];
            memcpy(src_copy, src, sizeof(src_copy));
            for (size_t i = 0; i != count; ++i) {
                self->single(dst, src_copy);
                dst += dst_stride;
                for (int j = 0; j < Nsrc; ++j) {
                    src_copy[j] += src_stride[j];
                }
            }
        }

        /**
         * Returns the child ckernel immediately following this one.
         */
        inline ckernel_prefix *get_child_ckernel() {
            return get_child_ckernel(sizeof(self_type));
        }

        /**
         * Returns the child ckernel at the specified offset.
         */
        inline ckernel_prefix *get_child_ckernel(intptr_t offset) {
            return base.get_child_ckernel(offset);
        }
    };
} // namespace kernels

class expr_kernel_generator;

/**
 * Evaluates any expression types in the array of
 * source types, passing the result non-expression
 * types on to the handler to build the rest of the
 * kernel.
 */
size_t make_expression_type_expr_kernel(ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *handler);

} // namespace dynd

#endif // _DYND__EXPR_KERNELS_HPP_
