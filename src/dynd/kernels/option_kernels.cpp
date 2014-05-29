//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/option_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

namespace {

struct bool_is_avail {
    static void single(char *dst, const char *src,
                       ckernel_prefix *DYND_UNUSED(self))
    {
        // Available if the value is 0 or 1
        *dst = *reinterpret_cast<const unsigned char *>(src) <= 1;
    }

    static void strided(char *dst, intptr_t dst_stride, const char *src,
                        intptr_t src_stride, size_t count,
                        ckernel_prefix *DYND_UNUSED(self))
    {
        // Available if the value is 0 or 1
        for (size_t i = 0; i != count;
                ++i, dst += dst_stride, src += src_stride) {
            *dst = *reinterpret_cast<const unsigned char *>(src) <= 1;
        }
    }
};

struct bool_assign_na {
    static void single(char *dst, const char *const* DYND_UNUSED(src),
                                ckernel_prefix *DYND_UNUSED(strided))
    {
        // NA is 2
        *dst = 2;
    }

    static void strided(char *dst, intptr_t dst_stride,
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
};

struct bool_nafunc {
    arrfunc_type_data is_avail;
    arrfunc_type_data assign_na;

    typedef dynd_bool base_type;

    static intptr_t instantiate_is_avail(
        const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp,
        const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
        const char *const *DYND_UNUSED(src_arrmeta), uint32_t kernreq,
        const eval::eval_context *DYND_UNUSED(ectx))
    {
        if (src_tp[0].get_type_id() != option_type_id ||
                src_tp[0].tcast<option_type>()->get_value_type().get_type_id() !=
                    (type_id_t)type_id_of<base_type>::value) {
            stringstream ss;
            ss << "Expected source type ?" << ndt::make_type<base_type>()
               << ", got " << src_tp[0];
            throw type_error(ss.str());
        }
        if (dst_tp.get_type_id() != bool_type_id) {
            throw type_error("Expected dst type bool");
        }
        ckernel_prefix *ckp = ckb->get_at<ckernel_prefix>(ckb_offset);
        ckp->set_unary_function<bool_is_avail>((kernel_request_t)kernreq);
        return ckb_offset + sizeof(ckernel_prefix);
    }

    static int resolve_is_avail_dst_type(const arrfunc_type_data *DYND_UNUSED(self),
                                  ndt::type &out_dst_tp,
                                  const ndt::type *DYND_UNUSED(src_tp),
                                  int DYND_UNUSED(throw_on_error))
    {
        out_dst_tp = ndt::make_type<dynd_bool>();
        return 1;
    }

    static intptr_t instantiate_assign_na(
        const arrfunc_type_data *DYND_UNUSED(self), dynd::ckernel_builder *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        uint32_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
    {
        if (src_tp[0].get_type_id() != option_type_id ||
                src_tp[0].tcast<option_type>()->get_value_type().get_type_id() !=
                    (type_id_t)type_id_of<base_type>::value) {
            stringstream ss;
            ss << "Expected source type ?" << ndt::make_type<base_type>()
               << ", got " << src_tp[0];
            throw type_error(ss.str());
        }
        if (dst_tp.get_type_id() != option_type_id ||
                dst_tp.tcast<option_type>()->get_value_type().get_type_id() !=
                    (type_id_t)type_id_of<base_type>::value) {
            stringstream ss;
            ss << "Expected dst type " << ndt::make_type<base_type>()
               << ", got " << dst_tp;
            throw type_error(ss.str());
        }
        ckernel_prefix *ckp = ckb->get_at<ckernel_prefix>(ckb_offset);
        ckp->set_expr_function<bool_assign_na>((kernel_request_t)kernreq);
        return ckb_offset + sizeof(ckernel_prefix);
    }

    bool_nafunc() {
        // Use a typevar instead of option[bool] to avoid a circular dependency
        is_avail.func_proto =
            ndt::make_funcproto(ndt::make_typevar("T"), ndt::make_type<dynd_bool>());
        is_avail.ckernel_funcproto = unary_operation_funcproto;
        is_avail.data_ptr = NULL;
        is_avail.instantiate = &bool_nafunc::instantiate_is_avail;
        is_avail.resolve_dst_type = &bool_nafunc::resolve_is_avail_dst_type;
        assign_na.func_proto =
            ndt::make_funcproto(0, NULL, ndt::make_typevar("T"));
        assign_na.ckernel_funcproto = expr_operation_funcproto;
        assign_na.data_ptr = NULL;
        assign_na.instantiate = &bool_nafunc::instantiate_assign_na;
    }

    nd::array get()
    {
        nd::array result(make_array_memory_block(option_type::make_nafunc_type(), 0, NULL));
        result.get_ndo()->m_data_pointer = reinterpret_cast<char *>(this);
        result.get_ndo()->m_flags = nd::default_access_flags;
        return result;
    }
};

} // anonymous namespace

const nd::array &kernels::get_option_builtin_nafunc(type_id_t tid)
{
    static bool_nafunc bna_data;
    static nd::array bna = bna_data.get();
    static nd::array nullarr;
    if (tid == bool_type_id) {
        return bna;
    } else {
        return nullarr;
    }
}
