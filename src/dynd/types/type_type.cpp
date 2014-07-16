//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

type_type::type_type()
    : base_type(type_type_id, custom_kind, sizeof(const base_type *),
                sizeof(const base_type *),
                type_flag_scalar | type_flag_zeroinit | type_flag_destructor, 0,
                0, 0)
{
}

type_type::~type_type()
{
}

void type_type::print_data(std::ostream& o,
                const char *DYND_UNUSED(arrmeta), const char *data) const
{
    const type_type_data *ddd = reinterpret_cast<const type_type_data *>(data);
    // This tests avoids the atomic increment/decrement of
    // always constructing a type object
    if (is_builtin_type(ddd->tp)) {
        o << ndt::type(ddd->tp, true);
    } else {
        ddd->tp->print_type(o);
    }
}

void type_type::print_type(std::ostream& o) const
{
    o << "type";
}

bool type_type::operator==(const base_type& rhs) const
{
    return this == &rhs || rhs.get_type_id() == type_type_id;
}

void type_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta),
                intptr_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const
{
}

void type_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta),
                const char *DYND_UNUSED(src_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference)) const
{
}

void type_type::arrmeta_reset_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void type_type::arrmeta_finalize_buffers(char *DYND_UNUSED(arrmeta)) const
{
}

void type_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
}

void type_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *data) const
{
    const base_type *bd = reinterpret_cast<type_type_data *>(data)->tp;
    if (!is_builtin_type(bd)) {
        base_type_decref(bd);
    }
}

void type_type::data_destruct_strided(const char *DYND_UNUSED(arrmeta), char *data,
                intptr_t stride, size_t count) const
{
    for (size_t i = 0; i != count; ++i, data += stride) {
        const base_type *bd = reinterpret_cast<type_type_data *>(data)->tp;
        if (!is_builtin_type(bd)) {
            base_type_decref(bd);
        }
    }
}

static void
typed_data_assignment_kernel_single(char *dst, const char *const *src,
                                    ckernel_prefix *DYND_UNUSED(self))
{
    // Free the destination reference
    base_type_xdecref(reinterpret_cast<const type_type_data *>(dst)->tp);
    // Copy the pointer and count the reference
    const base_type *bd =
        (*reinterpret_cast<const type_type_data *const *>(src))->tp;
    reinterpret_cast<type_type_data *>(dst)->tp = bd;
    base_type_xincref(bd);
}

namespace {
    struct string_to_type_kernel_extra {
        typedef string_to_type_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *src_string_dt;
        const char *src_arrmeta;
        assign_error_mode errmode;

        static void single(char *dst, const char *const *src,
                           ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const string &s = e->src_string_dt->get_utf8_string(
                e->src_arrmeta, src[0], e->errmode);
            ndt::type(s).swap(reinterpret_cast<type_type_data *>(dst)->tp);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->src_string_dt);
        }
    };

    struct type_to_string_kernel_extra {
        typedef type_to_string_kernel_extra extra_type;

        ckernel_prefix base;
        const base_string_type *dst_string_dt;
        const char *dst_arrmeta;
        eval::eval_context ectx;

        static void single(char *dst, const char *const *src,
                           ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            const base_type *bd =
                (*reinterpret_cast<const type_type_data *const *>(src))->tp;
            stringstream ss;
            if (is_builtin_type(bd)) {
                ss << ndt::type(bd, true);
            } else {
                bd->print_type(ss);
            }
            e->dst_string_dt->set_from_utf8_string(e->dst_arrmeta, dst,
                                                   ss.str(), &e->ectx);
        }

        static void destruct(ckernel_prefix *extra)
        {
            extra_type *e = reinterpret_cast<extra_type *>(extra);
            base_type_xdecref(e->dst_string_dt);
        }
    };
} // anonymous namespace

size_t type_type::make_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx) const
{
  ckb_offset =
      make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);

  if (this == dst_tp.extended()) {
    if (src_tp.get_type_id() == type_type_id) {
      ckernel_prefix *e = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      e->set_function<expr_single_t>(typed_data_assignment_kernel_single);
      return ckb_offset;
    } else if (src_tp.get_kind() == string_kind) {
      // String to type
      string_to_type_kernel_extra *e =
          ckb->alloc_ck_leaf<string_to_type_kernel_extra>(ckb_offset);
      e->base.set_function<expr_single_t>(&string_to_type_kernel_extra::single);
      e->base.destructor = &string_to_type_kernel_extra::destruct;
      // The kernel data owns a reference to this type
      e->src_string_dt =
          static_cast<const base_string_type *>(ndt::type(src_tp).release());
      e->src_arrmeta = src_arrmeta;
      e->errmode = ectx->errmode;
      return ckb_offset;
    } else if (!src_tp.is_builtin()) {
      return src_tp.extended()->make_assignment_kernel(
          ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
          ectx);
    }
  } else {
    if (dst_tp.get_kind() == string_kind) {
      // Type to string
      type_to_string_kernel_extra *e =
          ckb->alloc_ck_leaf<type_to_string_kernel_extra>(ckb_offset);
      e->base.set_function<expr_single_t>(&type_to_string_kernel_extra::single);
      e->base.destructor = &type_to_string_kernel_extra::destruct;
      // The kernel data owns a reference to this type
      e->dst_string_dt =
          static_cast<const base_string_type *>(ndt::type(dst_tp).release());
      e->dst_arrmeta = dst_arrmeta;
      e->ectx = *ectx;
      return ckb_offset;
    }
  }

  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}

static int equal_comparison(const char *const *src,
                            ckernel_prefix *DYND_UNUSED(self))
{
    const ndt::type *da = reinterpret_cast<const ndt::type *const *>(src)[0];
    const ndt::type *db = reinterpret_cast<const ndt::type *const *>(src)[1];
    return *da == *db;
}

static int not_equal_comparison(const char *const *src,
                                ckernel_prefix *DYND_UNUSED(self))
{
    const ndt::type *da = reinterpret_cast<const ndt::type *const *>(src)[0];
    const ndt::type *db = reinterpret_cast<const ndt::type *const *>(src)[1];
    return *da != *db;
}

size_t type_type::make_comparison_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &src0_dt,
    const char *DYND_UNUSED(src0_arrmeta), const ndt::type &src1_dt,
    const char *DYND_UNUSED(src1_arrmeta), comparison_type_t comptype,
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  if (this == src0_dt.extended()) {
    if (*this == *src1_dt.extended()) {
      ckernel_prefix *e = ckb->alloc_ck_leaf<ckernel_prefix>(ckb_offset);
      if (comptype == comparison_type_equal) {
        e->set_function<expr_predicate_t>(equal_comparison);
      } else if (comptype == comparison_type_not_equal) {
        e->set_function<expr_predicate_t>(not_equal_comparison);
      } else {
        throw not_comparable_error(src0_dt, src1_dt, comptype);
      }
      return ckb_offset;
    }
  }

  throw not_comparable_error(src0_dt, src1_dt, comptype);
}

const ndt::type& ndt::make_type()
{
    // Static instance of type_type, which has a reference count > 0 for the
    // lifetime of the program. This static construction is inside a
    // function to ensure correct creation order during startup.
    static type_type stt;
    static const ndt::type static_instance(&stt, true);
    return static_instance;
}

const ndt::type& ndt::make_strided_of_type()
{
    static strided_dim_type sdt(ndt::make_type());
    static const ndt::type static_instance(&sdt, true);
    return static_instance;
}
