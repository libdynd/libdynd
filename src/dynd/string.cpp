//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>

#include <dynd/string.hpp>
#include <dynd/types/string_type.hpp>

using namespace std;
namespace nd = dynd::nd;
namespace ndt = dynd::ndt;
using dynd::string_type_data;

nd::string::string(const nd::array& rhs)
{
    if (!rhs.is_null()) {
        if (rhs.is_immutable() &&
                rhs.get_type().get_type_id() == string_type_id &&
                rhs.get_type().tcast<string_type>()->get_encoding() ==
                    string_encoding_utf_8) {
            // It's already immutable and the right type
            m_value = rhs;
        } else if (rhs.get_type().value_type().get_kind() == string_kind) {
            m_value = rhs.ucast(ndt::make_string()).eval_immutable();
        } else {
            stringstream ss;
            ss << "Cannot implicitly convert nd::array of type "
               << rhs.get_type().value_type() << " to  string";
            throw type_error(ss.str());
        }
    }
}

std::string nd::string::str() const
{
    if (!m_value.is_null()) {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(
                m_value.get_readonly_originptr());
        return std::string(std->begin, std->end);
    } else {
        throw std::invalid_argument("Cannot get the value of a NULL dynd string");
    }
}

bool nd::string::empty() const
{
    if (!m_value.is_null()) {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(
                m_value.get_readonly_originptr());
        return std->begin == std->end;
    } else {
        return true;
    }
}

bool nd::string::operator==(const nd::string& rhs) const
{
    if (!is_null() && !rhs.is_null()) {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(
                m_value.get_readonly_originptr());
        const string_type_data *rhs_std =
            reinterpret_cast<const string_type_data *>(
                rhs.m_value.get_readonly_originptr());
        size_t size = std->end - std->begin;
        size_t rhs_size = rhs_std->end - rhs_std->begin;
        return size == rhs_size &&
               memcmp(std->begin, rhs_std->begin, size) == 0;
    } else {
        return is_null() == rhs.is_null();
    }
}

static inline bool lex_comp(const string_type_data *lhs,
                            const string_type_data *rhs)
{
    return std::lexicographical_compare(lhs->begin, lhs->end, rhs->begin,
                                        rhs->end);
}
bool nd::string::operator<(const nd::string& rhs) const
{
    return !rhs.is_null() &&
            (is_null() ||
            lex_comp(reinterpret_cast<const string_type_data *>(
                            m_value.get_readonly_originptr()),
                        reinterpret_cast<const string_type_data *>(
                            rhs.m_value.get_readonly_originptr())));
}



const char *nd::string::begin() const
{
    if (!m_value.is_null()) {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(
                m_value.get_readonly_originptr());
        return std->begin;
    } else {
        return NULL;
    }
}

const char *nd::string::end() const
{
    if (!m_value.is_null()) {
        const string_type_data *std =
            reinterpret_cast<const string_type_data *>(
                m_value.get_readonly_originptr());
        return std->end;
    } else {
        return NULL;
    }
}
