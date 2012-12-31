//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtype.hpp>
#include <dynd/gfunc/make_callable.hpp>

using namespace std;
using namespace dynd;


base_string_dtype::~base_string_dtype()
{
}

std::string base_string_dtype::get_utf8_string(const char *metadata, const char *data, assign_error_mode errmode) const
{
    const char *begin, *end;
    get_string_range(&begin, &end, metadata, data);
    return string_range_as_utf8_string(get_encoding(), begin, end, errmode);
}

size_t base_string_dtype::get_iterdata_size(int DYND_UNUSED(ndim)) const
{
    return 0;
}

static string get_extended_string_encoding(const dtype& dt) {
    const base_string_dtype *d = static_cast<const base_string_dtype *>(dt.extended());
    stringstream ss;
    ss << d->get_encoding();
    return ss.str();
}

static pair<string, gfunc::callable> base_string_dtype_properties[] = {
    pair<string, gfunc::callable>("encoding", gfunc::make_callable(&get_extended_string_encoding, "self"))
};

void base_string_dtype::get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const
{
    *out_properties = base_string_dtype_properties;
    *out_count = sizeof(base_string_dtype_properties) / sizeof(base_string_dtype_properties[0]);
}