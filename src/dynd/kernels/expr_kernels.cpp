//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/expr_kernels.hpp>

using namespace std;
using namespace dynd;


namespace {
template<int N>
struct expression_dtype_expr_kernel_extra {
    typedef expression_dtype_expr_kernel_extra extra_type;

    bool is_expr[N];

    
};

} // anonymous namespace

size_t dynd::make_expression_dtype_expr_kernel(hierarchical_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *handler)
{
    throw runtime_error("TODO: make_expression_dtype_expr_kernel");
}
