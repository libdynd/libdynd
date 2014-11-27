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

static void print_arrfunc(std::ostream &o,
                          const arrfunc_type_data *DYND_UNUSED(af))
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

size_t arrfunc_old_type::make_assignment_kernel(
    void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
    const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
    const ndt::type &src_tp, const char *DYND_UNUSED(src_arrmeta),
    kernel_request_t DYND_UNUSED(kernreq),
    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  // Nothing can be assigned to/from arrfunc
  stringstream ss;
  ss << "Cannot assign from " << src_tp << " to " << dst_tp;
  throw dynd::type_error(ss.str());
}
