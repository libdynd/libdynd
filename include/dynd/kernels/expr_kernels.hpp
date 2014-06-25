//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPR_KERNELS_HPP_
#define _DYND__EXPR_KERNELS_HPP_

#include <dynd/func/arrfunc.hpp>

namespace dynd {

namespace kernels {
    /**
     * A CRTP (curiously recurring template pattern) base class to help
     * create ckernels.
     */
    template<class CKT, int Nsrc>
    struct expr_ck : public general_ck<CKT> {
        typedef CKT self_type;
        typedef general_ck<CKT> parent_type;

        /**
         * Initializes just the base.function member
         */
        inline void init_kernfunc(kernel_request_t kernreq)
        {
            switch (kernreq) {
            case kernel_request_single:
                this->base.template set_function<expr_single_t>(
                    &self_type::single_wrapper);
                break;
            case kernel_request_strided:
                this->base.template set_function<expr_strided_t>(
                    &self_type::strided_wrapper);
                break;
            default: {
                std::stringstream ss;
                ss << "expr ckernel init: unrecognized ckernel request " << (int)kernreq;
                throw std::invalid_argument(ss.str());
            }
            }
        }

        static void single_wrapper(char *dst, const char *const *src,
                                   ckernel_prefix *rawself)
        {
            return parent_type::get_self(rawself)->single(dst, src);
        }

        static void strided_wrapper(char *dst, intptr_t dst_stride,
                                    const char *const *src,
                                    const intptr_t *src_stride, size_t count,
                                    ckernel_prefix *rawself)
        {
            return parent_type::get_self(rawself)
                ->strided(dst, dst_stride, src, src_stride, count);
        }

        /**
         * Default strided implementation calls single repeatedly.
         */
        inline void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count)
        {
            self_type *self = parent_type::get_self(&this->base);
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
    };
} // namespace kernels

class expr_kernel_generator;

/**
 * Evaluates any expression types in the array of
 * source types, passing the result non-expression
 * types on to the handler to build the rest of the
 * kernel.
 */
size_t make_expression_type_expr_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                size_t src_count, const ndt::type *src_dt, const char **src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *handler);

} // namespace dynd

#endif // _DYND__EXPR_KERNELS_HPP_
