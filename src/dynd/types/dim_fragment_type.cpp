//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/typevar_type.hpp>
#include <dynd/func/make_callable.hpp>

using namespace std;
using namespace dynd;

dim_fragment_type::dim_fragment_type(intptr_t ndim, const intptr_t *tagged_dims)
    : base_dim_type(dim_fragment_type_id, ndt::make_type<void>(), 0, 1,
                            0, type_flag_symbolic, false),
      m_tagged_dims(ndim, tagged_dims)
{
    m_members.ndim = static_cast<uint8_t>(ndim);
}

static inline ndt::type get_tagged_dims_from_type(intptr_t ndim,
                                                  const ndt::type &tp,
                                                  intptr_t *out_tagged_dims)
{
    ndt::type dtp = tp;
    for (int i = 0; i < ndim; ++i) {
        switch (dtp.get_type_id()) {
            case fixed_dim_type_id:
                out_tagged_dims[i] = dtp.tcast<fixed_dim_type>()->get_fixed_dim_size();
                break;
            case cfixed_dim_type_id:
                out_tagged_dims[i] = tp.tcast<fixed_dim_type>()->get_fixed_dim_size();
                break;
            case strided_dim_type_id:
            case offset_dim_type_id:
                out_tagged_dims[i] = -2;
                break;
            case var_dim_type_id:
                out_tagged_dims[i] = -1;
                break;
            default: {
                stringstream ss;
                ss << "dim_fragment_type failed to get shape from type " << tp;
                throw type_error(ss.str());
            }    
        }
        dtp = dtp.tcast<base_dim_type>()->get_element_type();
    }
    return dtp;
}

static inline bool broadcast_tagged_dims_from_type(intptr_t ndim, ndt::type tp,
                                                   const intptr_t *tagged_dims,
                                                   intptr_t *out_tagged_dims)
{
    for (intptr_t i = 0; i < ndim; ++i) {
        intptr_t tagged_dim = tagged_dims[i], dim_size;
        switch (tp.get_type_id()) {
            case fixed_dim_type_id:
                dim_size = tp.tcast<fixed_dim_type>()->get_fixed_dim_size();
                if (tagged_dim < 0 || tagged_dim == 1) {
                    out_tagged_dims[i] = dim_size;
                } else if (tagged_dim != dim_size) {
                    return false;
                }
                break;
            case cfixed_dim_type_id:
                dim_size = tp.tcast<fixed_dim_type>()->get_fixed_dim_size();
                if (tagged_dim < 0 || tagged_dim == 1) {
                    out_tagged_dims[i] = dim_size;
                } else if (tagged_dim != dim_size) {
                    return false;
                }
                break;
            case strided_dim_type_id:
            case offset_dim_type_id:
                if (tagged_dim < 0) {
                    out_tagged_dims[i] = -2;
                }
                break;
            case var_dim_type_id:
                // All broadcasting is done dynamically for var
                break;
            default: {
                stringstream ss;
                ss << "dim_fragment_type failed to get shape from type " << tp;
                throw type_error(ss.str());
            }    
        }
        tp = tp.tcast<base_dim_type>()->get_element_type();
    }
    return true;
}

dim_fragment_type::dim_fragment_type(intptr_t ndim, const ndt::type &tp)
    : base_dim_type(dim_fragment_type_id, ndt::make_type<void>(), 0, 1,
                            0, type_flag_symbolic, false),
      m_tagged_dims(ndim)
{
    if (ndim > tp.get_ndim()) {
        stringstream ss;
        ss << "Tried to make a dimension fragment from type " << tp << " with "
           << ndim << " dimensions, but the type only has " << tp.get_ndim()
           << " dimensions";
        throw type_error(ss.str());
    }
    get_tagged_dims_from_type(ndim, tp, m_tagged_dims.get());
    m_members.ndim = static_cast<uint8_t>(ndim);
}

ndt::type dim_fragment_type::broadcast_with_type(intptr_t ndim,
                                                 const ndt::type &tp) const
{
    if (ndim == 0) {
        return ndt::type(this, true);
    }
    intptr_t this_ndim = get_ndim();
    // In each case, we fill in the leading dimensions from the
    // higher-dimension input, then broadcast the rest together
    if (ndim > this_ndim) {
        dimvector shape(ndim);
        ndt::type dtp =
            get_tagged_dims_from_type(ndim - this_ndim, tp, shape.get());
        if (!broadcast_tagged_dims_from_type(this_ndim, dtp, get_tagged_dims(),
                                             shape.get() +
                                                 (ndim - this_ndim))) {
            return ndt::type();
        }
        return ndt::make_dim_fragment(ndim, shape.get());
    } else if (ndim < this_ndim) {
        dimvector shape(this_ndim);
        memcpy(shape.get(), get_tagged_dims(),
               (this_ndim - ndim) * sizeof(intptr_t));
        if (!broadcast_tagged_dims_from_type(
                ndim, tp, get_tagged_dims() + (this_ndim - ndim),
                shape.get() + (this_ndim - ndim))) {
            return ndt::type();
        }
        return ndt::make_dim_fragment(this_ndim, shape.get());
    } else {
        dimvector shape(ndim);
        if (!broadcast_tagged_dims_from_type(ndim, tp, get_tagged_dims(),
                                             shape.get())) {
            return ndt::type();
        }
        return ndt::make_dim_fragment(ndim, shape.get());
    }
}

ndt::type dim_fragment_type::apply_to_dtype(const ndt::type& dtp) const
{
    intptr_t ndim = get_ndim();
    if (ndim > 0) {
        ndt::type tp = dtp;
        for (intptr_t i = ndim - 1; i >= 0; --i) {
            switch (m_tagged_dims[i]) {
                case dim_fragment_var:
                    tp = ndt::make_var_dim(tp);
                    break;
                case dim_fragment_strided:
                    tp = ndt::make_strided_dim(tp);
                    break;
                default:
                    tp = ndt::make_fixed_dim(m_tagged_dims[i], tp);
                    break;
            }
        }
        return tp;
    } else {
        return dtp;
    }
}

void dim_fragment_type::print_data(std::ostream &DYND_UNUSED(o),
                                const char *DYND_UNUSED(arrmeta),
                                const char *DYND_UNUSED(data)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

void dim_fragment_type::print_type(std::ostream& o) const
{
    o << "dim_fragment[";
    intptr_t ndim = get_ndim();
    for (intptr_t i = 0; i < ndim; ++i) {
        if (m_tagged_dims[i] == dim_fragment_var) {
            o << "var * ";
        } else if (m_tagged_dims[i] == dim_fragment_strided) {
            o << "strided * ";
        } else {
            o << "fixed[" << m_tagged_dims[i] << "]";
        }
    }
    o << "void]";
}

ndt::type dim_fragment_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    size_t DYND_UNUSED(current_i), const ndt::type &DYND_UNUSED(root_tp),
    bool DYND_UNUSED(leading_dimension)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

intptr_t dim_fragment_type::apply_linear_index(
    intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
    const char *DYND_UNUSED(arrmeta), const ndt::type &DYND_UNUSED(result_tp),
    char *DYND_UNUSED(out_arrmeta), memory_block_data *DYND_UNUSED(embedded_reference), size_t DYND_UNUSED(current_i),
    const ndt::type &DYND_UNUSED(root_tp), bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
    memory_block_data **DYND_UNUSED(inout_dataref)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

intptr_t dim_fragment_type::get_dim_size(const char *DYND_UNUSED(arrmeta),
                                        const char *DYND_UNUSED(data)) const
{
    if (get_ndim() > 0) {
        return m_tagged_dims[0] >= 0 ? m_tagged_dims[0] : -1;
    } else {
        return -1;
    }
}

bool dim_fragment_type::is_lossless_assignment(const ndt::type &DYND_UNUSED(dst_tp),
                                               const ndt::type &DYND_UNUSED(src_tp)) const
{
    return false;
}

bool dim_fragment_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != dim_fragment_type_id) {
        return false;
    } else {
        const dim_fragment_type *dft =
            static_cast<const dim_fragment_type *>(&rhs);
        return get_ndim() == rhs.get_ndim() &&
               memcmp(m_tagged_dims.get(), dft->m_tagged_dims.get(),
                      get_ndim() * sizeof(intptr_t)) == 0;
    }
}

void dim_fragment_type::arrmeta_default_construct(
    char *DYND_UNUSED(arrmeta), intptr_t DYND_UNUSED(ndim),
    const intptr_t *DYND_UNUSED(shape)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

void dim_fragment_type::arrmeta_copy_construct(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

size_t dim_fragment_type::arrmeta_copy_construct_onedim(
    char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
    memory_block_data *DYND_UNUSED(embedded_reference)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

void dim_fragment_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const
{
    throw type_error("Cannot store data of dim_fragment type");
}

const ndt::type& ndt::make_dim_fragment()
{
    static dim_fragment_type dft(0, NULL);
    static const ndt::type static_instance(&dft, true);
    return static_instance;
}
