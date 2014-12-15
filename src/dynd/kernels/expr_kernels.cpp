//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/expr_kernels.hpp>

using namespace std;
using namespace dynd;


namespace {
template<int N>
struct expression_type_expr_kernel_extra {
    typedef expression_type_expr_kernel_extra extra_type;

    bool is_expr[N];

    
};

} // anonymous namespace

size_t dynd::make_expression_type_expr_kernel(void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
                const ndt::type& DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                size_t DYND_UNUSED(src_count), const ndt::type *DYND_UNUSED(src_tp), const char **DYND_UNUSED(src_arrmeta),
                kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *DYND_UNUSED(ectx),
                const expr_kernel_generator *DYND_UNUSED(handler))
{
    throw runtime_error("TODO: make_expression_type_expr_kernel");
}