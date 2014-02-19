//
// Copyright (C) 2010-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dynd_complex.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<float>& val)
{
    out << std::complex<float>(val.real(), val.imag());
    return out;
}

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<double>& val)
{
    out << std::complex<double>(val.real(), val.imag());
    return out;
}
