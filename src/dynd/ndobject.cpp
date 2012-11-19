//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_iter.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

dynd::ndobject::ndobject()
    : m_memblock()
{
}

void dynd::ndobject::swap(ndobject& rhs)
{
    m_memblock.swap(rhs.m_memblock);
}

template<class T>
inline typename enable_if<is_dtype_scalar<T>::value, memory_block_ptr>::type
make_immutable_builtin_scalar_ndobject(const T& value)
{
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(0, sizeof(T), scalar_align_of<T>::value, &data_ptr);
    *reinterpret_cast<T *>(data_ptr) = value;
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = reinterpret_cast<extended_dtype *>(type_id_of<T>::value);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | immutable_access_flag;
    return result;
}

ndobject dynd::make_corder_ndobject(const dtype& uniform_dtype, int ndim, const intptr_t *shape, const void *data)
{
    // Determine the total data size
    intptr_t size = uniform_dtype.element_size();
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }

    dtype array_dtype = make_strided_array_dtype(uniform_dtype, ndim);

    // Allocate the ndobject metadata and data in one memory block
    char *data_ptr = NULL;
    memory_block_ptr result = make_ndobject_memory_block(array_dtype.extended()->get_metadata_size(),
                    size, uniform_dtype.alignment(), &data_ptr);
    DYND_MEMCPY(data_ptr, data, size);

    // Fill in the preamble metadata
    ndobject_preamble *ndo = reinterpret_cast<ndobject_preamble *>(result.get());
    ndo->m_dtype = array_dtype.extended();
    extended_dtype_incref(ndo->m_dtype);
    ndo->m_data_pointer = data_ptr;
    ndo->m_data_reference = NULL;
    ndo->m_flags = read_access_flag | write_access_flag;

    // Fill in the ndobject metadata with C-order strides
    strided_array_dtype_metadata *meta = reinterpret_cast<strided_array_dtype_metadata *>(ndo + 1);
    intptr_t stride = uniform_dtype.element_size();
    for (int i = ndim - 1; i >= 0; --i) {
        meta[i].stride = stride;
        meta[i].size = shape[i];
        stride *= shape[i];
    }

    return ndobject(result);
}

// Constructors from C++ scalars
dynd::ndobject::ndobject(dynd_bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(bool value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(dynd_bool(value)))
{
}
dynd::ndobject::ndobject(signed char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned char value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned short value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned int value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(unsigned long long value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(float value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(double value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(std::complex<float> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(std::complex<double> value)
    : m_memblock(make_immutable_builtin_scalar_ndobject(value))
{
}
dynd::ndobject::ndobject(const dtype& dt)
    : m_memblock(make_ndobject_memory_block(dt, 0, NULL))
{
}

dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0)
    : m_memblock(make_ndobject_memory_block(dt, 1, &dim0))
{
}
dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1)
{
    intptr_t dims[2] = {dim0, dim1};
    m_memblock = make_ndobject_memory_block(dt, 2, dims);
}
dynd::ndobject::ndobject(const dtype& dt, intptr_t dim0, intptr_t dim1, intptr_t dim2)
{
    intptr_t dims[3] = {dim0, dim1, dim2};
    m_memblock = make_ndobject_memory_block(dt, 3, dims);
}

void dynd::ndobject::val_assign(const ndobject& rhs, assign_error_mode errmode,
                    const eval::eval_context *ectx) const
{
    // Verify access permissions
    if (!(get_flags()&write_access_flag)) {
        throw runtime_error("tried to write to a dynd array that is not writeable");
    }
    if (!(rhs.get_flags()&read_access_flag)) {
        throw runtime_error("tried to read from a dynd array that is not readable");
    }

    if (rhs.is_scalar()) {
        unary_specialization_kernel_instance assign;
        const char *src_ptr = rhs.get_ndo()->m_data_pointer;

        ndobject_iter<1, 0> iter(*this);
        get_dtype_assignment_kernel(iter.get_uniform_dtype(), rhs.get_dtype(), errmode, ectx, assign);
        unary_operation_t assign_fn = assign.specializations[scalar_unary_specialization];
        if (!iter.empty()) {
            do {
                assign_fn(iter.data(), 0, src_ptr, 0, 1, assign.auxdata);
            } while (iter.next());
        }
    } else {
        throw runtime_error("TODO: finish ndobject::val_assign for non-scalar case");
    }
}

void ndobject::debug_dump(std::ostream& o, const std::string& indent) const
{
    o << indent << "------ ndobject\n";
    if (m_memblock.get()) {
        const ndobject_preamble *ndo = get_ndo();
        o << " address: " << (void *)m_memblock.get() << "\n";
        o << " refcount: " << ndo->m_memblockdata.m_use_count << "\n";
        o << " data pointer: " << (void *)ndo->m_data_pointer << "\n";
        o << " data reference: " << (void *)ndo->m_data_reference << "\n";
        o << " flags: " << ndo->m_flags << " (";
        if (ndo->m_flags & read_access_flag) o << "read_access ";
        if (ndo->m_flags & write_access_flag) o << "write_access ";
        if (ndo->m_flags & immutable_access_flag) o << "immutable ";
        o << ")\n";
        o << " dtype raw value: " << (void *)ndo->m_dtype << "\n";
        o << " dtype: " << get_dtype() << "\n";
        if (!ndo->is_builtin_dtype()) {
            o << " metadata:\n";
            ndo->m_dtype->metadata_debug_dump(get_ndo_meta(), o, indent + " ");
        }
    } else {
        o << indent << "NULL\n";
    }
    o << indent << "------" << endl;
}

std::ostream& dynd::operator<<(std::ostream& o, const ndobject& rhs)
{
    if (rhs.get_ndo() != NULL) {
        if (rhs.get_ndo()->is_builtin_dtype()) {
            print_builtin_scalar(rhs.get_ndo()->get_builtin_type_id(), o, rhs.get_ndo()->m_data_pointer);
        } else {
            rhs.get_ndo()->m_dtype->print_element(o, rhs.get_ndo()->m_data_pointer, rhs.get_ndo_meta());
        }
    } else {
        o << "<null>";
    }
    return o;
}