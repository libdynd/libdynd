//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifdef DYND_CUDA

#include <dynd/types/cuda_host_type.hpp>

using namespace std;
using namespace dynd;

cuda_host_type::cuda_host_type(const ndt::type& target_tp, unsigned int cuda_host_flags)
    : base_memory_type(cuda_host_type_id, memory_kind, 0, target_tp.get_data_alignment(),
        type_flag_none, 0), m_cuda_host_flags(cuda_host_flags)
{
}

cuda_host_type::~cuda_host_type()
{
}

void cuda_host_type::print_data(std::ostream& o, const char *metadata, const char *data) const
{
    m_target_tp.print_data(o, metadata, data);
}

void cuda_host_type::print_type(std::ostream& o) const
{
    o << "cuda host, " << m_target_tp;
}

bool cuda_host_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != cuda_host_type_id) {
        return false;
    } else {
        const cuda_host_type *dt = static_cast<const cuda_host_type*>(&rhs);
        return m_target_tp == dt->m_target_tp;
    }
}

void cuda_host_type::metadata_default_construct(char *metadata, intptr_t ndim, const intptr_t* shape) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_default_construct(metadata, ndim, shape);
    }
}

void cuda_host_type::metadata_destruct(char *metadata) const
{
    if (!m_target_tp.is_builtin()) {
        m_target_tp.extended()->metadata_destruct(metadata);
    }
}

#endif // DYND_CUDA
