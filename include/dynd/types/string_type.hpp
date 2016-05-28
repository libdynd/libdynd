//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string type uses a 16-byte SSO representation
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>
#include <dynd/types/sso_bytestring.hpp>
#include <dynd/types/string_kind_type.hpp>

namespace dynd {

class DYNDT_API string : public sso_bytestring<1> {
public:
  /** Default-constructs to an empty string */
  string() {}

  string(const string &rhs) : sso_bytestring(rhs) {}

  string(const string &&rhs) : sso_bytestring(std::move(rhs)) {}

  string(const char *data, size_t size) : sso_bytestring(data, size) {}

  /** Construct from a std::string, assumed to be UTF-8 */
  string(const std::string &other) : sso_bytestring(other.data(), other.size()) {}

  /** Construct from a C string, assumed to be UTF-8 */
  string(const char *cstr) : sso_bytestring(cstr, strlen(cstr)) {}

  string &operator=(const string &rhs) {
    sso_bytestring::assign(rhs.data(), rhs.size());
    return *this;
  }

  string &operator=(string &&rhs) {
    sso_bytestring::operator=(std::move(rhs));
    return *this;
  }

  bool operator<(const string &rhs) const {
    return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator<=(const string &rhs) const {
    return !std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  bool operator>=(const string &rhs) const {
    return !std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator>(const string &rhs) const {
    return std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  string operator+(const string &rhs) {
    string result;
    result.resize(size() + rhs.size());
    DYND_MEMCPY(result.data(), data(), size());
    DYND_MEMCPY(result.data() + size(), rhs.data(), rhs.size());
    return result;
  }

  string &operator+=(const string &rhs) {
    sso_bytestring::append(rhs.data(), rhs.size());
    return *this;
  }
};

namespace ndt {

  class DYNDT_API string_type : public base_string_type {
  private:
    const string_encoding_t m_encoding{string_encoding_utf_8};
    const std::string m_encoding_repr{encoding_as_string(string_encoding_utf_8)};

  public:
    typedef string data_type;

    string_type(type_id_t id)
        : base_string_type(id, make_type<string_kind_type>(), sizeof(string), alignof(string),
                           type_flag_zeroinit | type_flag_destructor, 0) {}

    string_encoding_t get_encoding() const { return m_encoding; }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const { return string_encoding_char_size_table[string_encoding_utf_8]; }

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    type get_canonical_type() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;
  };

  template <>
  struct id_of<string> : std::integral_constant<type_id_t, string_id> {};

  template <>
  struct id_of<string_type> : std::integral_constant<type_id_t, string_id> {};

  template <>
  struct traits<string> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return make_type<string_type>(); }

    static string na() { return string(); }
  };

  template <>
  struct traits<std::string> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<string>(); }
  };

  template <>
  struct traits<const char *> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<string_type>(); }
  };

  template <size_t N>
  struct traits<char[N]> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<string_type>(); }
  };

  template <size_t N>
  struct traits<const char[N]> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<string_type>(); }
  };

} // namespace dynd::ndt
} // namespace dynd
