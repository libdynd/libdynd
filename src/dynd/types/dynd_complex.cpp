//
// Copyright (C) 2010-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dynd_complex.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<float>& val) {
    return (out << "(" << val.m_real << " + " << val.m_imag << "j)");
}

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<double>& val) {
    return (out << "(" << val.m_real << " + " << val.m_imag << "j)");
}
