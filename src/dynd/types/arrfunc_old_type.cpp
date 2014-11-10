//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/view.hpp>
#include <dynd/types/arrfunc_old_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

arrfunc_old_type::arrfunc_old_type()
    : base_type(arrfunc_old_type_id, custom_kind, sizeof(arrfunc_type_data),
                    scalar_align_of<uint64_t>::value,
                    type_flag_scalar|type_flag_zeroinit|type_flag_destructor,
                    0, 0, 0)
{
}

arrfunc_old_type::~arrfunc_old_type()
{
}

static void print_arrfunc(std::ostream& o, const arrfunc_type_data *af)
{
  o << "arrfunc_old deprecated";
}

void arrfunc_old_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta),
                              const char *data) const
{
  const arrfunc_type_data *af =
      reinterpret_cast<const arrfunc_type_data *>(data);
  print_arrfunc(o, af);
}

void arrfunc_old_type::print_type(std::ostream& o) const
{
    o << "arrfunc";
}

bool arrfunc_old_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == arrfunc_old_type_id;
}

void arrfunc_old_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                                             bool DYND_UNUSED(blockref_alloc))
    const
{
}

void arrfunc_old_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                const char *DYND_UNUSED(src_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void arrfunc_old_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_old_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_old_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
}

void arrfunc_old_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
    const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
    d->~arrfunc_type_data();
}

void arrfunc_old_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const arrfunc_type_data *d = reinterpret_cast<arrfunc_type_data *>(data);
        d->~arrfunc_type_data();
    }
}

/////////////////////////////////////////
// arrfunc to string assignment

namespace {
    struct arrfunc_to_string_ck : public kernels::unary_ck<arrfunc_to_string_ck> {
        ndt::type m_dst_string_dt;
        const char *m_dst_arrmeta;
        eval::eval_context m_ectx;

        inline void single(char *dst, const char *src)
        {
            const arrfunc_type_data *af =
                reinterpret_cast<const arrfunc_type_data *>(src);
            stringstream ss;
            print_arrfunc(ss, af);
            m_dst_string_dt.extended<base_string_type>()->set_from_utf8_string(
                m_dst_arrmeta, dst, ss.str(), &m_ectx);
        }
    };
} // anonymous namespace

static intptr_t make_arrfunc_to_string_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_string_dt,
    const char *dst_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
    typedef arrfunc_to_string_ck self_type;
    self_type *self = self_type::create_leaf(ckb, kernreq, ckb_offset);
    // The kernel data owns a reference to this type
    self->m_dst_string_dt = dst_string_dt;
    self->m_dst_arrmeta = dst_arrmeta;
    self->m_ectx = *ectx;
    return ckb_offset;
}

size_t arrfunc_old_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp,
    const char *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *ectx) const
{
    if (this == dst_tp.extended()) {
    } else {
        if (dst_tp.get_kind() == string_kind) {
            // Assignment to strings
            return make_arrfunc_to_string_assignment_kernel(
                ckb, ckb_offset, dst_tp, dst_arrmeta, kernreq, ectx);
        }
    }
    
    // Nothing can be assigned to/from arrfunc
    stringstream ss;
    ss << "Cannot assign from " << src_tp << " to " << dst_tp;
    throw dynd::type_error(ss.str());
}
