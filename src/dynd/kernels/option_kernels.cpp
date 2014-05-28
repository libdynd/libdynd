//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/option_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

namespace {

struct bool_nafunc {
    arrfunc_type_data is_avail;
    arrfunc_type_data assign_na;

    static void is_avail_single(char *dst, const char *src,
                                ckernel_prefix *DYND_UNUSED(strided))
    {
        // Available if the value is 0 or 1
        *dst = *reinterpret_cast<const unsigned char *>(src) <= 1;
    }

    static void is_avail_strided(char *dst, intptr_t dst_stride,
                                 const char *src, intptr_t src_stride,
                                 size_t count,
                                 ckernel_prefix *DYND_UNUSED(strided))
    {
        // Available if the value is 0 or 1
        for (size_t i = 0; i != count;
                ++i, dst += dst_stride, src += src_stride) {
            *dst = *reinterpret_cast<const unsigned char *>(src) <= 1;
        }
    }

    static void assign_na_single(char *dst, const char *const* DYND_UNUSED(src),
                                ckernel_prefix *DYND_UNUSED(strided))
    {
        // NA is 2
        *dst = 2;
    }

    static void assign_na_strided(char *dst, intptr_t dst_stride,
                                  const char *const *DYND_UNUSED(src),
                                  const intptr_t *DYND_UNUSED(src_stride),
                                  size_t count,
                                  ckernel_prefix *DYND_UNUSED(strided))
    {
        // NA is 2
        if (dst_stride == 1) {
            memset(dst, 2, count);
        } else {
            for (size_t i = 0; i != count; ++i, dst += dst_stride) {
                *dst = 2;
            }
        }
    }

    static intptr_t instantiate_is_avail(
        const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        uint32_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
    {
        if (src_tp[0].get_type_id() != option_type_id ||
            src_tp[0].tcast<option_type>()->get_value_type().get_type_id() !=
                bool_type_id) {
            stringstream ss;
            ss << "Expected source type option[bool], got " << src_tp[0];
            throw type_error(ss.str());
        }
    }

    bool_nafunc() {
        // Use a typevar instead of option[bool] to avoid a circular dependency
        is_avail.func_proto =
            ndt::make_funcproto(ndt::make_typevar("T"), ndt::make_type<bool>());
        is_avail.ckernel_funcproto = unary_operation_funcproto;
        is_avail.data_ptr = NULL;
        is_avail.instantiate = &bool_nafunc::instantiate_is_avail;
        assign_na.func_proto =
            ndt::make_funcproto(0, NULL, ndt::make_typevar("T"));
        assign_na.ckernel_funcproto = expr_operation_funcproto;
        assign_na.data_ptr = NULL;
        assign_na.instantiate = &bool_nafunc::instantiate_assign_na;
    }
};

} // anonymous namespace

const nd::array &kernels::get_option_builtin_nafunc(type_id_t tid)
{

}
