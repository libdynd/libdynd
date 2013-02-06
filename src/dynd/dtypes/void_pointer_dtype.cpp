//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/void_pointer_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/exceptions.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

void dynd::void_pointer_dtype::print_data(std::ostream& o, const char *DYND_UNUSED(metadata), const char *data) const
{
    uintptr_t target_ptr = *reinterpret_cast<const uintptr_t *>(data);
    o << "0x";
    hexadecimal_print(o, target_ptr);
}

void dynd::void_pointer_dtype::print_dtype(std::ostream& o) const {

    o << "pointer<void>";
}

void dynd::void_pointer_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape)) const
{
}

void dynd::void_pointer_dtype::get_shape(size_t DYND_UNUSED(i),
                intptr_t *DYND_UNUSED(out_shape), const char *DYND_UNUSED(metadata)) const
{
}

bool dynd::void_pointer_dtype::is_lossless_assignment(const dtype& DYND_UNUSED(dst_dt), const dtype& DYND_UNUSED(src_dt)) const
{
    return false;
}

void dynd::void_pointer_dtype::get_single_compare_kernel(kernel_instance<compare_operations_t>& DYND_UNUSED(out_kernel)) const {
    throw std::runtime_error("void_pointer_dtype::get_single_compare_kernel not supported yet");
}

bool dynd::void_pointer_dtype::operator==(const base_dtype& rhs) const
{
    return rhs.get_type_id() == void_pointer_type_id;
}
