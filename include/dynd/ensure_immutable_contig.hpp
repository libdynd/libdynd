//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_ENSURE_IMMUTABLE_CONTIG_HPP
#define DYND_ENSURE_IMMUTABLE_CONTIG_HPP

#include <dynd/array.hpp>
#include <dynd/types/strided_dim_type.hpp>

namespace dynd { namespace nd {

namespace detail {
    template<typename T>
    struct ensure_immutable_contig {
        inline static enable_if<is_dynd_scalar<T>::value, bool> run(nd::array& a)
        {
            const ndt::type& tp = a.get_type();
            if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                // It's immutable and "strided * <something>"
                const ndt::type &et =
                    tp.tcast<strided_dim_type>()->get_element_type();
                const strided_dim_type_arrmeta *md =
                    reinterpret_cast<const strided_dim_type_arrmeta *>(
                        a.get_arrmeta());
                if (et.get_type_id() == type_id_of<T>::value &&
                        md->stride == sizeof(T)) {
                    // It also has the right type and is contiguous,
                    // so no modification necessary.
                    return true;
                }
            }
            // We have to make a copy, check that it's a 1D array, and that
            // it has the same array kind as the requested type.
            if (tp.get_ndim() == 1) {
                // It's a 1D array
                const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                if(et.get_kind() == dynd_kind_of<T>::value) {
                    // It also has the same array kind as requested
                    nd::array tmp = nd::empty(
                        a.get_dim_size(),
                        ndt::make_strided_dim(ndt::make_type<T>()));
                    tmp.vals() = a;
                    tmp.flag_as_immutable();
                    a.swap(tmp);
                    return true;
                }
            }
            // It's not compatible, so return false
            return false;
        }
    };

    template<>
    struct ensure_immutable_contig<nd::string> {
        inline static bool run(nd::array& a)
        {
            const ndt::type& tp = a.get_type();
            if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                // It's immutable and "strided * <something>"
                const ndt::type &et =
                    tp.tcast<strided_dim_type>()->get_element_type();
                const strided_dim_type_arrmeta *md =
                    reinterpret_cast<const strided_dim_type_arrmeta *>(
                        a.get_arrmeta());
                if (et.get_type_id() == string_type_id &&
                        et.tcast<string_type>()->get_encoding() == string_encoding_utf_8 &&
                        md->stride == sizeof(string_type_data)) {
                    // It also has the right type and is contiguous,
                    // so no modification necessary.
                    return true;
                }
            }
            // We have to make a copy, check that it's a 1D array, and that
            // it has the same array kind as the requested type.
            if (tp.get_ndim() == 1) {
                // It's a 1D array
                const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                if(et.get_kind() == string_kind) {
                    // It also has the same array kind as requested
                    nd::array tmp = nd::empty(a.get_dim_size(), ndt::make_strided_of_string());
                    tmp.vals() = a;
                    tmp.flag_as_immutable();
                    a.swap(tmp);
                    return true;
                }
            }
            // It's not compatible, so return false
            return false;
        }
    };

    template<>
    struct ensure_immutable_contig<ndt::type> {
        inline static bool run(nd::array& a)
        {
            const ndt::type& tp = a.get_type();
            if (a.is_immutable() && tp.get_type_id() == strided_dim_type_id) {
                // It's immutable and "strided * <something>"
                const ndt::type &et =
                    tp.tcast<strided_dim_type>()->get_element_type();
                const strided_dim_type_arrmeta *md =
                    reinterpret_cast<const strided_dim_type_arrmeta *>(
                        a.get_arrmeta());
                if (et.get_type_id() == type_type_id &&
                        md->stride == sizeof(ndt::type)) {
                    // It also has the right type and is contiguous,
                    // so no modification necessary.
                    return true;
                }
            }
            // We have to make a copy, check that it's a 1D array, and that
            // it has the same array kind as the requested type.
            if (tp.get_ndim() == 1) {
                // It's a 1D array
                const ndt::type& et = tp.get_type_at_dimension(NULL, 1).value_type();
                if(et.get_type_id() == type_type_id) {
                    // It also has the same array type as requested
                    intptr_t dim_size = a.get_dim_size();
                    nd::array tmp = nd::typed_empty(1, &dim_size, ndt::make_strided_of_type());
                    tmp.vals() = a;
                    tmp.flag_as_immutable();
                    a.swap(tmp);
                    return true;
                }
            }
            // It's not compatible, so return false
            return false;
        }
    };
} // namespace detail

/**
  * Makes sure the array is an immutable, contiguous, "strided * T"
  * array, modifying it in place if necessary.
  */
template<typename T>
bool ensure_immutable_contig(nd::array& a)
{
    if (!a.is_null()) {
        return detail::ensure_immutable_contig<T>::run(a);
    } else {
        return false;
    }
}

}} // namespace dynd::nd

#endif // DYND_ENSURE_IMMUTABLE_CONTIG_HPP
