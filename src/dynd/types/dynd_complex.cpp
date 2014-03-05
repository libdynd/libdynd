//
// Copyright (C) 2010-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/dynd_complex.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<float>& DYND_UNUSED(val))
{
    return (out << "<complex printing unimplemented>");}

std::ostream& dynd::operator<<(ostream& out, const dynd_complex<double>& DYND_UNUSED(val))
{
    return (out << "<complex printing unimplemented>");
}
