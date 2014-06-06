//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_HPP_
#define _DYND__STRING_HPP_

#include <dynd/array.hpp>

namespace dynd { namespace nd {

/**
 * This is a container for an nd::array which must always be
 * an immutable "string", or NULL.
 */
class string {
    nd::array m_value;

public:
    inline string() : m_value() {}
    /**
     * Constructors from various string types. All of these assume the
     * source data is encoded as utf8.
     */
    inline string(const nd::string &rhs) : m_value(rhs.m_value) {}
    inline string(const char *rhs) : m_value(rhs) {}
    inline string(const char *begin, size_t size)
        : m_value(make_string_array(begin, size, string_encoding_utf_8,
                                    immutable_access_flag))
    {
    }
    inline string(const char *begin, const char *end)
        : m_value(make_string_array(begin, end - begin, string_encoding_utf_8,
                                    immutable_access_flag))
    {
    }
    inline string(const std::string& rhs) : m_value(rhs) {}
    /**
     * Constructor from an nd::array. Validates the input, and evaluates
     * to "string" type if it has the "string" kind.
     */
    string(const nd::array& rhs);

    /** Can implicitly convert to nd::array */
    inline operator const nd::array&() const {
        return m_value;
    }

    // TODO: Also need string_ref/string_view
    std::string str() const;

    bool empty() const;

    inline bool is_null() const {
        return m_value.is_null();
    }

    bool operator==(const nd::string& rhs) const;

    bool operator<(const nd::string& rhs) const;

    const char *begin() const;
    const char *end() const;
};

}} // namespace dynd::nd

#endif // _DYND__STRING_HPP_
