//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/datashape_formatter.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/fixed_string_type.hpp>

using namespace std;
using namespace dynd;

static void format_datashape(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                             const std::string &indent, bool multiline);

static void format_struct_datashape(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                                    const std::string &indent, bool multiline)
{
  // The data requires arrmeta
  if (arrmeta == NULL) {
    data = NULL;
  }
  const ndt::struct_type *bsd = tp.extended<ndt::struct_type>();
  size_t field_count = bsd->get_field_count();
  const std::vector<uintptr_t> &arrmeta_offsets = bsd->get_arrmeta_offsets();
  const uintptr_t *data_offsets = NULL;
  if (data != NULL) {
    data_offsets = bsd->get_data_offsets(arrmeta);
  }
  o << (multiline ? "{\n" : "{");
  for (size_t i = 0; i < field_count; ++i) {
    if (multiline) {
      o << indent << "  ";
    }
    o << bsd->get_field_name(i) << ": ";
    format_datashape(o, bsd->get_field_type(i), arrmeta ? (arrmeta + arrmeta_offsets[i]) : NULL,
                     data ? (data + data_offsets[i]) : NULL, multiline ? (indent + "  ") : indent, multiline);
    if (multiline) {
      o << ",\n";
    }
    else if (i != field_count - 1) {
      o << ", ";
    }
  }
  o << indent << "}";
}

static void format_dim_datashape(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                                 const std::string &indent, bool multiline)
{
  switch (tp.get_id()) {
  case fixed_dim_id: {
    if (tp.is_symbolic()) {
      // A symbolic type, so arrmeta/data can't exist
      o << "Fixed * ";
      format_datashape(o, tp.extended<ndt::base_dim_type>()->get_element_type(), NULL, NULL, indent, multiline);
      break;
    }
    const ndt::fixed_dim_type *fad = tp.extended<ndt::fixed_dim_type>();
    intptr_t dim_size = fad->get_fixed_dim_size();
    o << dim_size << " * ";
    // Allow data to keep going only if the dimension size is 1
    if (dim_size != 1) {
      data = NULL;
    }
    format_datashape(o, fad->get_element_type(), arrmeta + (arrmeta ? sizeof(fixed_dim_type_arrmeta) : 0), data, indent,
                     multiline);
    break;
  }
  case var_dim_id: {
    const ndt::var_dim_type *vad = tp.extended<ndt::var_dim_type>();
    const char *child_data = NULL;
    if (data == NULL || arrmeta == NULL) {
      o << "var * ";
    }
    else {
      const ndt::var_dim_type::data_type *d = reinterpret_cast<const ndt::var_dim_type::data_type *>(data);
      if (d->begin == NULL) {
        o << "var * ";
      }
      else {
        o << d->size << " * ";
        if (d->size == 1) {
          const ndt::var_dim_type::metadata_type *md =
              reinterpret_cast<const ndt::var_dim_type::metadata_type *>(arrmeta);
          child_data = d->begin + md->offset;
        }
      }
    }
    format_datashape(o, vad->get_element_type(), arrmeta ? (arrmeta + sizeof(ndt::var_dim_type::metadata_type)) : NULL,
                     child_data, indent, multiline);
    break;
  }
  default: {
    stringstream ss;
    ss << "Datashape formatting for dynd type " << tp << " is not yet implemented";
    throw runtime_error(ss.str());
  }
  }
}

static void format_string_datashape(std::ostream &o, const ndt::type &tp)
{
  switch (tp.get_id()) {
  case string_id:
  case fixed_string_id:
    // data shape only has one kind of string
    o << "string";
    break;
  default: {
    stringstream ss;
    ss << "unrecognized string dynd type " << tp << " while formatting datashape";
    throw dynd::type_error(ss.str());
  }
  }
}

static void format_complex_datashape(std::ostream &o, const ndt::type &tp)
{
  switch (tp.get_id()) {
  case complex_float32_id:
    o << "complex[float32]";
    break;
  case complex_float64_id:
    o << "complex[float64]";
    break;
  default: {
    stringstream ss;
    ss << "unrecognized string complex type " << tp << " while formatting datashape";
    throw dynd::type_error(ss.str());
  }
  }
}

static void format_datashape(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data,
                             const std::string &indent, bool multiline)
{
  switch (tp.get_id()) {
  case struct_id:
    format_struct_datashape(o, tp, arrmeta, data, indent, multiline);
    break;
  case fixed_dim_id:
  case var_dim_id:
    format_dim_datashape(o, tp, arrmeta, data, indent, multiline);
    break;
  case fixed_string_id:
  case string_id:
    format_string_datashape(o, tp);
    break;
  case complex_float32_id:
  case complex_float64_id:
    format_complex_datashape(o, tp);
    break;
  default:
    o << tp;
    break;
  }
}

void dynd::format_datashape(std::ostream &o, const ndt::type &tp, const char *arrmeta, const char *data, bool multiline)
{
  ::format_datashape(o, tp, arrmeta, data, "", multiline);
}

std::string dynd::format_datashape(const ndt::type &tp, const std::string &prefix, bool multiline)
{
  stringstream ss;
  ss << prefix;
  ::format_datashape(ss, tp, NULL, NULL, "", multiline);
  return ss.str();
}
